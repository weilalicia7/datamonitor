"""
NHS Open Data Ingestion Module

Automated downloading, validation, and processing of NHS open datasets
for continuous model recalibration.

Data Sources:
    1. Cancer Waiting Times (monthly CSV) - england.nhs.uk
       ⚠ DECOMMISSIONING June 2026. Final release = June 2026 data.
    2. SACT Activity Dashboard (annually) - digital.nhs.uk/ndrs
    3. NHSBSA Secondary Care Medicines Data with Indicative Price (monthly) - opendata.nhsbsa.net
    4. SACT v4.0 Patient-Level Data (monthly from Aug 2026) - digital.nhs.uk/ndrs
    5. NHS NDRS Cancer Data Hub / FDS (monthly, activates July 2026) - digital.nhs.uk/ndrs
       ↑ CWT SUCCESSOR SOURCE: automatically activated when CWT decommissions.

All sources are publicly available, free, and require no authentication.

CWT Decommission Timeline:
    Final CWT monthly data release: June 2026
    CWT system goes offline: ~ July 2026
    Successor: NHS NDRS Cancer Data Hub aggregate outputs
               + Faster Diagnosis Standard (FDS) statistics (28-day standard)
    Transition handled automatically: this ingester switches to
    'ndrs_cancer_data_hub' source on 1 July 2026.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - NHS data ingestion disabled")


@dataclass
class DataSourceConfig:
    """Configuration for an NHS data source."""
    name: str
    url_template: str
    format: str  # 'csv', 'excel', 'json_api'
    frequency: str  # 'monthly', 'quarterly', 'annual'
    description: str
    last_downloaded: Optional[datetime] = None
    last_hash: Optional[str] = None
    enabled: bool = True


@dataclass
class IngestionResult:
    """Result of a data ingestion attempt."""
    source: str
    success: bool
    timestamp: datetime
    records_count: int = 0
    file_path: Optional[str] = None
    is_new_data: bool = False
    error: Optional[str] = None
    data_hash: Optional[str] = None


class NHSDataIngester:
    """
    Manages automated ingestion of NHS open data sources.

    Usage:
        ingester = NHSDataIngester()
        results = ingester.check_and_download_all()
        for result in results:
            if result.is_new_data:
                # Trigger model recalibration
                recalibrator.update(result.source, result.file_path)
    """

    # NHS England Cancer Waiting Times base URL
    CWT_BASE_URL = "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2"

    # NHSBSA CKAN API
    NHSBSA_API_URL = "https://opendata.nhsbsa.net/api/3/action/datastore_search_sql"

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / 'datasets' / 'nhs_open_data')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.data_dir / 'ingestion_history.json'
        self.history: List[Dict] = self._load_history()

        self.sources = self._configure_sources()

        logger.info(f"NHSDataIngester initialized. Data dir: {self.data_dir}")

    # ── CWT decommission & successor URLs ─────────────────────────────────────
    # NHS England CWT system final release: June 2026 data.
    # From 1 July 2026 the 'ndrs_cancer_data_hub' source activates automatically.
    CWT_DECOMMISSION_DATE = datetime(2026, 7, 1)

    # NHS NDRS Cancer Data Hub — CWT successor (aggregate outputs)
    NDRS_CDH_URL = "https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub"

    # Faster Diagnosis Standard statistics (28-day referral-to-diagnosis target)
    FDS_STATS_URL = (
        "https://www.england.nhs.uk/statistics/statistical-work-areas/"
        "cancer-waiting-times/faster-diagnosis-standard/"
    )

    # SACT v4.0 data quality phases
    # Collection commenced 1 April 2026; rollout ends June 2026; full Aug 2026
    SACT_V4_PHASES = {
        'rollout_start':    (2026, 4, 1),   # Collection begins (partial only)
        'full_conformance': (2026, 7, 1),   # All trusts required to submit
        'first_complete':   (2026, 8, 1),   # First complete monthly dataset
    }

    def _configure_sources(self) -> Dict[str, DataSourceConfig]:
        """Configure all NHS data sources."""
        return {
            'cancer_waiting_times': DataSourceConfig(
                name='Cancer Waiting Times',
                url_template='https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/',
                format='csv',
                frequency='monthly',
                description='27,000+ rows CSV. Directly recalibrates no-show baseline rates and waiting time targets. Feeds Level 1 recalibration.'
            ),
            'sact_activity': DataSourceConfig(
                name='SACT Activity Dashboard',
                url_template='https://digital.nhs.uk/ndrs/data/data-sets/sact',
                format='metadata',
                frequency='annual',
                description=(
                    'SACT Activity Dashboard — annual aggregate outputs (confirmed April 2026; '
                    'previously published quarterly). '
                    'Metadata reference for schema validation. '
                    'Interactive Shiny dashboard at nhsd-ndrs.shinyapps.io/sact_activity/. '
                    'Data downloadable via dashboard UI — no direct CSV API available.'
                )
            ),
            'ndrs_cancer_data_hub': DataSourceConfig(
                name='NHS NDRS Cancer Data Hub (CWT Successor)',
                url_template='https://digital.nhs.uk/ndrs/data/data-outputs/cancer-data-hub',
                format='html_check',
                frequency='monthly',
                description=(
                    'CWT SUCCESSOR SOURCE — activates automatically on 1 July 2026 when the '
                    'NHS England Cancer Waiting Times system decommissions. '
                    'Polls the NDRS Cancer Data Hub for aggregate downloadable outputs '
                    'and the Faster Diagnosis Standard (FDS) statistics page (28-day referral-to-diagnosis). '
                    'FDS URL: england.nhs.uk/statistics/statistical-work-areas/'
                    'cancer-waiting-times/faster-diagnosis-standard/. '
                    'Data feeds Level 1 baseline recalibration (replaces CWT for no-show '
                    'baseline rates and waiting-time targets).'
                )
            ),
            'nhsbsa_prescribing': DataSourceConfig(
                name='NHSBSA Secondary Care Medicines Data with Indicative Price',
                url_template='https://opendata.nhsbsa.net/dataset/secondary-care-medicines-data-indicative-price',
                format='catalogue',
                frequency='monthly',
                description='Secondary Care Medicines Data with Indicative Price (SCMD-IP). Replaces original SCMD dataset retired June 2022. Monthly hospital prescribing data including chemotherapy drugs. Latest: January 2026.'
            ),
            'sact_v4_patient_data': DataSourceConfig(
                name='SACT v4.0 Patient-Level Data',
                url_template='https://digital.nhs.uk/ndrs/data/data-sets/sact',
                format='csv',
                frequency='monthly',
                description=(
                    'SACT v4.0 patient-level submissions (60 fields, 7 sections). '
                    'Collection commenced 1 April 2026 (rollout period April-June 2026 = partial/preliminary). '
                    'Full conformance July 2026. First complete dataset August 2026. '
                    'Rollout data usable for Level 1-2 recalibration. '
                    'Complete data (Aug 2026+) triggers Level 3 full retrain. '
                    'Auto-detects CSV files placed in datasets/nhs_open_data/sact_v4/.'
                )
            ),
        }

    def _load_history(self) -> List[Dict]:
        """Load ingestion history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_history(self):
        """Save ingestion history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history[-100:], f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save ingestion history: {e}")

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of data for change detection."""
        return hashlib.sha256(data).hexdigest()[:16]

    def check_and_download_all(self) -> List[IngestionResult]:
        """
        Check all sources for new data and download if available.

        Returns list of IngestionResult objects.
        """
        results = []

        for source_id, config in self.sources.items():
            if not config.enabled:
                continue

            if not self._is_due(config):
                logger.debug(f"Skipping {source_id}: not due for update")
                continue

            try:
                result = self._download_source(source_id, config)
                results.append(result)

                if result.success:
                    self.history.append({
                        'source': source_id,
                        'timestamp': datetime.now().isoformat(),
                        'success': True,
                        'records': result.records_count,
                        'is_new': result.is_new_data,
                        'hash': result.data_hash
                    })
            except Exception as e:
                logger.error(f"Failed to download {source_id}: {e}")
                results.append(IngestionResult(
                    source=source_id,
                    success=False,
                    timestamp=datetime.now(),
                    error=str(e)
                ))

        self._save_history()
        return results

    def _is_due(self, config: DataSourceConfig) -> bool:
        """Check if a data source is due for update."""
        if config.last_downloaded is None:
            return True

        now = datetime.now()
        freq_days = {
            'monthly': 28,
            'quarterly': 85,
            'annual': 350,
            'daily': 1
        }
        threshold = timedelta(days=freq_days.get(config.frequency, 30))
        return (now - config.last_downloaded) > threshold

    def _download_source(self, source_id: str, config: DataSourceConfig) -> IngestionResult:
        """Download data from a specific source."""
        if source_id == 'cancer_waiting_times':
            return self.download_cancer_waiting_times()
        elif source_id == 'nhsbsa_prescribing':
            return self.download_nhsbsa_prescribing()
        elif source_id == 'sact_activity':
            return self.download_sact_activity()
        elif source_id == 'sact_v4_patient_data':
            return self.download_sact_v4_data()
        elif source_id == 'ndrs_cancer_data_hub':
            return self.download_ndrs_cancer_data_hub()
        else:
            return IngestionResult(
                source=source_id, success=False,
                timestamp=datetime.now(), error=f"Unknown source: {source_id}"
            )

    def download_cancer_waiting_times(self, year_month: str = None) -> IngestionResult:
        """
        Download Cancer Waiting Times CSV from NHS England.

        Scrapes the statistics page for the latest CSV download link,
        then downloads the Combined CSV file.

        ⚠ DECOMMISSIONING: The NHS England CWT system publishes its final
        monthly dataset for June 2026. From 1 July 2026 this method returns
        an error and the 'ndrs_cancer_data_hub' source activates automatically.
        """
        # ── CWT Decommission Guard ─────────────────────────────────────────
        if datetime.now() >= self.CWT_DECOMMISSION_DATE:
            logger.warning(
                "CWT DECOMMISSIONED (1 July 2026): NHS England Cancer Waiting Times "
                "system has been retired. No new data is available from this source. "
                "The 'ndrs_cancer_data_hub' source is now active as the replacement. "
                f"Check: {self.NDRS_CDH_URL}"
            )
            # Disable this source to prevent future retries
            cwt_cfg = self.sources.get('cancer_waiting_times')
            if cwt_cfg:
                cwt_cfg.enabled = False
            return IngestionResult(
                source='cancer_waiting_times',
                success=False,
                timestamp=datetime.now(),
                error=(
                    'CWT system decommissioned June 2026. '
                    'Switch to ndrs_cancer_data_hub source. '
                    f'URL: {self.NDRS_CDH_URL}'
                )
            )

        if not REQUESTS_AVAILABLE:
            return IngestionResult(
                source='cancer_waiting_times', success=False,
                timestamp=datetime.now(), error='requests library not available'
            )

        output_dir = self.data_dir / 'cancer_waiting_times'
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import re

            # Scrape the statistics page for CSV links
            stats_url = 'https://www.england.nhs.uk/statistics/statistical-work-areas/cancer-waiting-times/monthly-data-and-summaries/2025-26-monthly-cancer-waiting-times-statistics/'
            logger.info(f"Scraping CWT page for download links...")
            page = requests.get(stats_url, timeout=30)

            csv_links = re.findall(r'href="([^"]*Combined-CSV[^"]*\.csv)"', page.text)

            if not csv_links:
                return IngestionResult(
                    source='cancer_waiting_times', success=False,
                    timestamp=datetime.now(), error='No CSV links found on CWT page'
                )

            # Download the latest (first) CSV
            latest_url = csv_links[0]
            filename = latest_url.split('/')[-1]
            output_file = output_dir / filename

            logger.info(f"Downloading CWT: {filename}")
            response = requests.get(latest_url, timeout=60)

            if response.status_code != 200:
                return IngestionResult(
                    source='cancer_waiting_times', success=False,
                    timestamp=datetime.now(), error=f'Download failed: HTTP {response.status_code}'
                )

            data_hash = self._compute_hash(response.content)
            config = self.sources['cancer_waiting_times']
            is_new = data_hash != config.last_hash

            with open(output_file, 'wb') as f:
                f.write(response.content)

            try:
                df = pd.read_csv(output_file)
                records = len(df)
            except Exception:
                records = len(response.content.decode('utf-8', errors='ignore').split('\n'))

            config.last_downloaded = datetime.now()
            config.last_hash = data_hash

            logger.info(f"CWT downloaded: {filename}, {records} records, new={is_new}")

            return IngestionResult(
                source='cancer_waiting_times', success=True,
                timestamp=datetime.now(), records_count=records,
                file_path=str(output_file), is_new_data=is_new, data_hash=data_hash
            )

        except Exception as e:
            logger.error(f"CWT download error: {e}")
            return IngestionResult(
                source='cancer_waiting_times', success=False,
                timestamp=datetime.now(), error=str(e)
            )

    def download_nhsbsa_prescribing(self) -> IngestionResult:
        """
        Download Secondary Care Medicines Data with Indicative Price (SCMD) from NHSBSA.

        NOTE: The original 'secondary-care-medicines-data' dataset was retired in June 2022.
        The replacement dataset 'secondary-care-medicines-data-indicative-price' has been
        available since then and includes indicative price data per item.

        Latest available: January 2026 (released ~20th March 2026, 2-month lag).

        Uses the CKAN package API to find the latest resource, then downloads metadata.
        """
        if not REQUESTS_AVAILABLE:
            return IngestionResult(
                source='nhsbsa_prescribing', success=False,
                timestamp=datetime.now(), error='requests library not available'
            )

        output_dir = self.data_dir / 'prescribing'
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get the latest SCMD with Indicative Price package info
            # NOTE: Old package 'secondary-care-medicines-data' retired June 2022.
            # New package: 'secondary-care-medicines-data-indicative-price'
            pkg_url = 'https://opendata.nhsbsa.net/api/3/action/package_show?id=secondary-care-medicines-data-indicative-price'
            logger.info("Fetching NHSBSA SCMD with Indicative Price package info...")
            response = requests.get(pkg_url, timeout=30)

            if response.status_code != 200:
                return IngestionResult(
                    source='nhsbsa_prescribing', success=False,
                    timestamp=datetime.now(), error=f'NHSBSA API returned {response.status_code}'
                )

            data = response.json()
            if not data.get('success'):
                return IngestionResult(
                    source='nhsbsa_prescribing', success=False,
                    timestamp=datetime.now(), error='NHSBSA API returned failure'
                )

            resources = data['result'].get('resources', [])
            if not resources:
                return IngestionResult(
                    source='nhsbsa_prescribing', success=False,
                    timestamp=datetime.now(), error='No resources found in SCMD package'
                )

            # Save resource metadata (list of all available datasets with URLs)
            resource_info = [{
                'name': r.get('name', ''),
                'url': r.get('url', ''),
                'format': r.get('format', ''),
                'created': r.get('created', ''),
                'id': r.get('id', '')
            } for r in resources]

            output_file = output_dir / f'nhsbsa_scmd_indicative_price_{datetime.now().strftime("%Y_%m")}.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'package': 'secondary-care-medicines-data-indicative-price',
                    'description': 'NHS hospital secondary care prescribing with indicative prices (replaces retired SCMD dataset, June 2022 onwards)',
                    'total_resources': len(resources),
                    'latest_resource': resource_info[-1]['name'] if resource_info else '',
                    'latest_url': resource_info[-1]['url'] if resource_info else '',
                    'fetched_at': datetime.now().isoformat(),
                    'resources': resource_info
                }, f, indent=2)

            data_hash = self._compute_hash(json.dumps(resource_info).encode())
            config = self.sources['nhsbsa_prescribing']
            is_new = data_hash != config.last_hash
            config.last_downloaded = datetime.now()
            config.last_hash = data_hash

            latest_name = resource_info[-1]['name'] if resource_info else 'unknown'
            logger.info(f"NHSBSA SCMD catalogue saved: {len(resources)} resources, latest={latest_name}")

            return IngestionResult(
                source='nhsbsa_prescribing', success=True,
                timestamp=datetime.now(), records_count=len(resources),
                file_path=str(output_file), is_new_data=is_new, data_hash=data_hash
            )

        except Exception as e:
            logger.error(f"NHSBSA download error: {e}")
            return IngestionResult(
                source='nhsbsa_prescribing', success=False,
                timestamp=datetime.now(), error=str(e)
            )

    def download_sact_activity(self) -> IngestionResult:
        """
        Download SACT Activity data.

        The SACT Activity dashboard is a Shiny app without direct CSV API.
        Instead, we fetch the SACT dataset specification page for the latest
        documentation and metadata, and check for any manually placed files.
        """
        if not REQUESTS_AVAILABLE:
            return IngestionResult(
                source='sact_activity', success=False,
                timestamp=datetime.now(), error='requests library not available'
            )

        output_dir = self.data_dir / 'sact_activity'
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Fetch the SACT dataset page for metadata
            sact_url = 'https://digital.nhs.uk/ndrs/data/data-sets/sact'
            logger.info("Fetching SACT dataset page metadata...")
            response = requests.get(sact_url, timeout=30)

            if response.status_code == 200:
                # Save the page content as reference
                import re
                # Extract key info from page
                title_match = re.search(r'<title>(.*?)</title>', response.text)
                title = title_match.group(1) if title_match else 'SACT Dataset'

                metadata = {
                    'source': 'SACT Dataset Page',
                    'url': sact_url,
                    'title': title,
                    'fetched_at': datetime.now().isoformat(),
                    'page_size': len(response.content),
                    'status': 'v4.0 collection commenced 1 April 2026 (3-month rollout period)',
                    'dashboard_url': 'https://nhsd-ndrs.shinyapps.io/sact_activity/',
                    'note': (
                        'SACT v4.0: Data collection commenced 1 April 2026. '
                        'Rollout period April-June 2026 (partial trust submissions). '
                        'Full conformance expected 1 July 2026. '
                        'First COMPLETE monthly dataset expected August 2026. '
                        'Data NOT available to external systems until after rollout (post June 2026). '
                        'Interactive Shiny dashboard - data downloadable via dashboard UI. '
                        'Auto-download not supported for Shiny apps.'
                    )
                }

                output_file = output_dir / f'sact_metadata_{datetime.now().strftime("%Y_%m")}.json'
                with open(output_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                data_hash = self._compute_hash(json.dumps(metadata).encode())
                config = self.sources['sact_activity']
                is_new = data_hash != config.last_hash
                config.last_downloaded = datetime.now()
                config.last_hash = data_hash

                logger.info("SACT metadata saved")

                return IngestionResult(
                    source='sact_activity', success=True,
                    timestamp=datetime.now(), records_count=1,
                    file_path=str(output_file), is_new_data=is_new, data_hash=data_hash
                )

            return IngestionResult(
                source='sact_activity', success=False,
                timestamp=datetime.now(), error=f'SACT page returned HTTP {response.status_code}'
            )

        except Exception as e:
            logger.error(f"SACT download error: {e}")
            return IngestionResult(
                source='sact_activity', success=False,
                timestamp=datetime.now(), error=str(e)
            )

    def check_sact_v4_availability(self) -> Dict[str, Any]:
        """
        Check SACT v4.0 data availability and assign a data quality phase.

        Phases:
            not_started        — before 1 April 2026
            rollout_partial    — April–June 2026: partial trust submissions
                                 USABLE for Level 1-2 recalibration (preliminary)
            full_conformance   — July 2026: all trusts submitting, month not yet processed
                                 USABLE for Level 2 recalibration
            first_complete     — August 2026+: first complete monthly dataset available
                                 USABLE for Level 3 full retrain

        Returns a dict with phase, quality, recommended recalibration level, and notes.
        """
        now = datetime.now()
        rs = self.SACT_V4_PHASES['rollout_start']
        fc = self.SACT_V4_PHASES['full_conformance']
        fco = self.SACT_V4_PHASES['first_complete']

        rollout_start = datetime(*rs)
        full_conformance = datetime(*fc)
        first_complete = datetime(*fco)

        sact_v4_dir = self.data_dir / 'sact_v4'
        local_csvs = list(sact_v4_dir.glob('*.csv')) if sact_v4_dir.exists() else []
        has_local = len(local_csvs) > 0

        if now < rollout_start:
            return {
                'phase': 'not_started',
                'quality': 'unavailable',
                'usable': False,
                'has_local_files': has_local,
                'local_file_count': len(local_csvs),
                'local_files': [f.name for f in local_csvs],
                'recommended_recalibration_level': None,
                'note': 'SACT v4.0 collection has not commenced yet (starts 1 April 2026).',
                'checked_at': now.isoformat(),
            }

        if now < full_conformance:
            return {
                'phase': 'rollout_partial',
                'quality': 'preliminary',
                'usable': True,  # Can use partial data with caveats
                'has_local_files': has_local,
                'local_file_count': len(local_csvs),
                'local_files': [f.name for f in local_csvs],
                'recommended_recalibration_level': 2 if has_local else 1,
                'note': (
                    'SACT v4.0 rollout period (April–June 2026). '
                    'Only partial trust submissions — nationally incomplete. '
                    'USABLE for preliminary analysis and Level 1-2 recalibration. '
                    'Results reflect submitting trusts only, not full England picture. '
                    'Do NOT trigger Level 3 full retrain until August 2026.'
                ),
                'checked_at': now.isoformat(),
            }

        if now < first_complete:
            return {
                'phase': 'full_conformance',
                'quality': 'conformance',
                'usable': True,
                'has_local_files': has_local,
                'local_file_count': len(local_csvs),
                'local_files': [f.name for f in local_csvs],
                'recommended_recalibration_level': 2 if has_local else None,
                'note': (
                    'SACT v4.0 full conformance period (July 2026). '
                    'All trusts required to submit. '
                    'First complete monthly dataset (July 2026 data) expected August 2026. '
                    'USABLE for Level 2 recalibration now. '
                    'Level 3 full retrain recommended once August 2026 data confirmed.'
                ),
                'checked_at': now.isoformat(),
            }

        # August 2026 and beyond
        return {
            'phase': 'first_complete',
            'quality': 'complete',
            'usable': True,
            'has_local_files': has_local,
            'local_file_count': len(local_csvs),
            'local_files': [f.name for f in local_csvs],
            'recommended_recalibration_level': 3 if has_local else 2,
            'note': (
                'First complete SACT v4.0 dataset period (August 2026+). '
                'Full national picture available. '
                'Level 3 full retrain triggered automatically when CSV files detected. '
                'Place monthly SACT v4.0 CSV in datasets/nhs_open_data/sact_v4/ to trigger.'
            ),
            'checked_at': now.isoformat(),
        }

    def download_sact_v4_data(self) -> IngestionResult:
        """
        Check for and ingest SACT v4.0 patient-level data.

        Strategy (SACT v4.0 is submitted by trusts to NDRS, not a public API):
        1. Check the local sact_v4/ directory for CSV files placed manually or
           downloaded from NDRS portal after authentication.
        2. Fetch the NDRS SACT page to detect any newly published download links.
        3. Assign quality metadata based on the current phase (rollout/complete).
        4. Return IngestionResult with quality_phase field for recalibration routing.

        Recalibration routing (done by auto_recalibration.py):
            rollout_partial / conformance → Level 1-2 (preliminary, no full retrain)
            first_complete               → Level 3 (full retrain, national picture)
        """
        sact_v4_dir = self.data_dir / 'sact_v4'
        sact_v4_dir.mkdir(parents=True, exist_ok=True)

        availability = self.check_sact_v4_availability()
        phase = availability['phase']
        quality = availability['quality']

        # ── 1. Check local directory for CSV files ────────────────────────
        local_csvs = sorted(sact_v4_dir.glob('*.csv'), key=os.path.getmtime, reverse=True)

        if local_csvs:
            latest_csv = local_csvs[0]
            try:
                df = pd.read_csv(latest_csv, nrows=5)  # quick header check
                records = sum(1 for _ in open(latest_csv)) - 1
            except Exception:
                records = 0

            raw_bytes = latest_csv.read_bytes()
            data_hash = self._compute_hash(raw_bytes)
            config = self.sources.get('sact_v4_patient_data')
            is_new = data_hash != (config.last_hash if config else None)

            if config:
                config.last_downloaded = datetime.now()
                config.last_hash = data_hash

            # Save enriched metadata alongside the CSV
            meta = {
                'file': latest_csv.name,
                'records': records,
                'quality_phase': phase,
                'quality': quality,
                'recommended_recalibration_level': availability['recommended_recalibration_level'],
                'note': availability['note'],
                'ingested_at': datetime.now().isoformat(),
            }
            with open(sact_v4_dir / 'latest_meta.json', 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(
                f"SACT v4.0 data: {latest_csv.name}, {records} records, "
                f"phase={phase}, quality={quality}, is_new={is_new}"
            )

            return IngestionResult(
                source='sact_v4_patient_data',
                success=True,
                timestamp=datetime.now(),
                records_count=records,
                file_path=str(latest_csv),
                is_new_data=is_new,
                data_hash=data_hash,
            )

        # ── 2. No local files — poll NDRS page for published links ────────
        if not REQUESTS_AVAILABLE:
            return IngestionResult(
                source='sact_v4_patient_data', success=False,
                timestamp=datetime.now(), error='requests not available'
            )

        try:
            ndrs_url = 'https://digital.nhs.uk/ndrs/data/data-sets/sact'
            logger.info(f"Checking NDRS SACT page for v4.0 data links (phase={phase})...")
            response = requests.get(ndrs_url, timeout=30)

            download_links: List[str] = []
            if response.status_code == 200:
                import re
                # Look for CSV/ZIP download links mentioning SACT
                raw_links = re.findall(r'href="([^"]*(?:\.csv|\.zip)[^"]*)"', response.text)
                download_links = [l for l in raw_links if 'sact' in l.lower()]

                if download_links:
                    logger.info(f"Found {len(download_links)} SACT download link(s) on NDRS page")

            # Save status / manifest regardless of links found
            manifest = {
                'checked_at': datetime.now().isoformat(),
                'phase': phase,
                'quality': quality,
                'note': availability['note'],
                'download_links_found': download_links,
                'instructions': (
                    f"SACT v4.0 phase: {phase}. "
                    "To use real data: download from NDRS portal and place CSV(s) in "
                    "datasets/nhs_open_data/sact_v4/. "
                    "The auto-learning scheduler will detect and ingest them within 24h."
                )
            }
            manifest_file = sact_v4_dir / f'sact_v4_status_{datetime.now().strftime("%Y_%m")}.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            return IngestionResult(
                source='sact_v4_patient_data',
                success=True,
                timestamp=datetime.now(),
                records_count=len(download_links),
                file_path=str(manifest_file),
                is_new_data=bool(download_links),  # new if links discovered
            )

        except Exception as e:
            logger.error(f"SACT v4.0 check error: {e}")
            return IngestionResult(
                source='sact_v4_patient_data', success=False,
                timestamp=datetime.now(), error=str(e)
            )

    def download_ndrs_cancer_data_hub(self) -> IngestionResult:
        """
        Poll the NHS NDRS Cancer Data Hub and Faster Diagnosis Standard (FDS)
        statistics pages for downloadable outputs.

        This is the automatic successor to Cancer Waiting Times (CWT), which
        decommissioned June 2026.  It activates from 1 July 2026 onward.

        Strategy:
            1. Check that today >= CWT_DECOMMISSION_DATE; if not, return early
               with an informational result (CWT still active).
            2. Fetch the NDRS Cancer Data Hub page and scan for CSV/XLSX links.
            3. Fetch the Faster Diagnosis Standard statistics page and scan for
               Combined-CSV or monthly-data links.
            4. Save a manifest JSON with all discovered links so the operator
               can review and download manually (or automate if links are direct).
            5. If any direct CSV links are found, attempt to download the latest.

        Data feeds:
            - Level 1 recalibration: no-show baseline rates and waiting-time targets
              (same role previously filled by CWT monthly CSV).
        """
        output_dir = self.data_dir / 'ndrs_cancer_data_hub'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Only activate after CWT decommissions
        if datetime.now() < self.CWT_DECOMMISSION_DATE:
            manifest = {
                'status': 'not_yet_active',
                'note': (
                    f'NDRS Cancer Data Hub source activates on {self.CWT_DECOMMISSION_DATE.date()} '
                    'when the NHS England CWT system decommissions. '
                    'CWT is still the active source until then.'
                ),
                'ndrs_cdh_url': self.NDRS_CDH_URL,
                'fds_url': self.FDS_STATS_URL,
                'checked_at': datetime.now().isoformat(),
            }
            manifest_file = output_dir / 'ndrs_cdh_status.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            # Mark as checked so _is_due() won't re-run this on every call
            cfg = self.sources.get('ndrs_cancer_data_hub')
            if cfg:
                cfg.last_downloaded = datetime.now()
            return IngestionResult(
                source='ndrs_cancer_data_hub',
                success=True,
                timestamp=datetime.now(),
                records_count=0,
                file_path=str(manifest_file),
                is_new_data=False,
                error=None,
            )

        if not REQUESTS_AVAILABLE:
            return IngestionResult(
                source='ndrs_cancer_data_hub', success=False,
                timestamp=datetime.now(), error='requests library not available'
            )

        import re

        discovered_links: List[Dict[str, str]] = []
        errors: List[str] = []

        # ── 1. Scan NDRS Cancer Data Hub page ────────────────────────────
        try:
            logger.info(f"Fetching NDRS Cancer Data Hub: {self.NDRS_CDH_URL}")
            resp_cdh = requests.get(self.NDRS_CDH_URL, timeout=30)
            if resp_cdh.status_code == 200:
                raw_links = re.findall(
                    r'href="([^"]*(?:\.csv|\.xlsx|\.zip)[^"]*)"', resp_cdh.text
                )
                for lnk in raw_links:
                    discovered_links.append({
                        'source_page': 'ndrs_cancer_data_hub',
                        'url': lnk,
                        'type': 'direct_download',
                    })
                logger.info(
                    f"NDRS CDH page scanned: {len(raw_links)} download link(s) found"
                )
            else:
                errors.append(f'NDRS CDH page: HTTP {resp_cdh.status_code}')
        except Exception as e:
            errors.append(f'NDRS CDH page error: {e}')

        # ── 2. Scan Faster Diagnosis Standard statistics page ─────────────
        try:
            logger.info(f"Fetching FDS statistics page: {self.FDS_STATS_URL}")
            resp_fds = requests.get(self.FDS_STATS_URL, timeout=30)
            if resp_fds.status_code == 200:
                fds_csv_links = re.findall(
                    r'href="([^"]*(?:Combined-CSV|monthly[^"]*|FDS[^"]*)'
                    r'(?:\.csv|\.xlsx|\.zip)[^"]*)"',
                    resp_fds.text,
                    re.IGNORECASE,
                )
                for lnk in fds_csv_links:
                    discovered_links.append({
                        'source_page': 'faster_diagnosis_standard',
                        'url': lnk,
                        'type': 'fds_monthly_data',
                    })
                logger.info(
                    f"FDS page scanned: {len(fds_csv_links)} CSV/XLSX link(s) found"
                )
            else:
                errors.append(f'FDS stats page: HTTP {resp_fds.status_code}')
        except Exception as e:
            errors.append(f'FDS page error: {e}')

        # ── 3. Attempt to download the first direct CSV found ─────────────
        downloaded_file: Optional[str] = None
        records = 0
        for link_info in discovered_links:
            url = link_info['url']
            if not url.startswith('http'):
                # Use the correct base domain per source page
                if link_info.get('source_page') == 'faster_diagnosis_standard':
                    url = 'https://www.england.nhs.uk' + url
                else:
                    url = 'https://digital.nhs.uk' + url
            try:
                logger.info(f"Attempting download: {url}")
                dl = requests.get(url, timeout=60)
                if dl.status_code == 200:
                    fname = url.split('/')[-1].split('?')[0] or 'ndrs_cdh_data.csv'
                    fpath = output_dir / fname
                    with open(fpath, 'wb') as f:
                        f.write(dl.content)
                    try:
                        df_tmp = pd.read_csv(fpath) if fpath.suffix == '.csv' else pd.read_excel(fpath)
                        records = len(df_tmp)
                    except Exception:
                        records = len(dl.content.decode('utf-8', errors='ignore').split('\n'))
                    downloaded_file = str(fpath)
                    logger.info(f"Downloaded NDRS CDH file: {fname}, {records} rows")
                    break
            except Exception as e:
                logger.warning(f"Download failed for {url}: {e}")

        # ── 4. Save manifest ──────────────────────────────────────────────
        manifest = {
            'checked_at': datetime.now().isoformat(),
            'cwt_decommission_date': self.CWT_DECOMMISSION_DATE.isoformat(),
            'status': 'active — CWT successor',
            'ndrs_cdh_url': self.NDRS_CDH_URL,
            'fds_url': self.FDS_STATS_URL,
            'discovered_links': discovered_links,
            'downloaded_file': downloaded_file,
            'errors': errors,
            'instructions': (
                'NHS CWT decommissioned. This source (NDRS Cancer Data Hub) is now active. '
                'If direct download links are listed above, the system attempted to download '
                'the first CSV/XLSX automatically. If none were downloaded, retrieve data '
                'manually from the NDRS Cancer Data Hub or FDS statistics page and place '
                'in datasets/nhs_open_data/ndrs_cancer_data_hub/. '
                'Data feeds Level 1 recalibration (no-show baselines, waiting-time targets).'
            ),
        }
        manifest_file = output_dir / f'ndrs_cdh_{datetime.now().strftime("%Y_%m")}.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        data_hash = self._compute_hash(json.dumps(discovered_links).encode())
        config = self.sources.get('ndrs_cancer_data_hub')
        is_new = data_hash != (config.last_hash if config else None)
        if config:
            config.last_downloaded = datetime.now()
            config.last_hash = data_hash

        # success = at least one page was reachable (< 2 errors means ≥ 1 page OK)
        # Finding no CSV links is valid — NDRS CDH may not expose direct downloads yet.
        both_pages_failed = len(errors) >= 2
        success = not both_pages_failed or downloaded_file is not None
        return IngestionResult(
            source='ndrs_cancer_data_hub',
            success=success,
            timestamp=datetime.now(),
            records_count=records or len(discovered_links),
            file_path=downloaded_file or str(manifest_file),
            is_new_data=is_new,
            data_hash=data_hash,
            error='; '.join(errors) if both_pages_failed else None,
        )

    def get_latest_data(self, source: str) -> Optional[pd.DataFrame]:
        """Load the most recent downloaded data for a source."""
        # Map source IDs to their actual on-disk directory names
        _dir_map = {
            'cancer_waiting_times': 'cancer_waiting_times',
            'nhsbsa_prescribing': 'prescribing',
            'sact_activity': 'sact_activity',
            'sact_v4_patient_data': 'sact_v4',
            'ndrs_cancer_data_hub': 'ndrs_cancer_data_hub',
        }
        dir_name = _dir_map.get(source, source)
        source_dir = self.data_dir / dir_name

        if not source_dir.exists():
            return None

        files = sorted(source_dir.glob('*.*'), key=os.path.getmtime, reverse=True)
        if not files:
            return None

        latest = files[0]
        try:
            if str(latest).endswith('.csv'):
                return pd.read_csv(latest)
            elif str(latest).endswith('.xlsx'):
                return pd.read_excel(latest)
            elif str(latest).endswith('.json'):
                with open(latest) as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to load {latest}: {e}")

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get status of all data sources including SACT v4.0 availability."""
        # Map source IDs to actual directory names
        dir_map = {
            'cancer_waiting_times': 'cancer_waiting_times',
            'nhsbsa_prescribing': 'prescribing',
            'sact_activity': 'sact_activity',
            'sact_v4_patient_data': 'sact_v4',
            'ndrs_cancer_data_hub': 'ndrs_cancer_data_hub',
        }

        status = {}
        for source_id, config in self.sources.items():
            dir_name = dir_map.get(source_id, source_id)
            source_dir = self.data_dir / dir_name
            files = list(source_dir.glob('*.*')) if source_dir.exists() else []
            file_count = len(files)

            # If last_downloaded is None but files exist, infer from newest file
            last_dl = config.last_downloaded
            if last_dl is None and files:
                newest = max(files, key=os.path.getmtime)
                last_dl = datetime.fromtimestamp(os.path.getmtime(newest))
                config.last_downloaded = last_dl  # Persist for is_due check

            entry = {
                'name': config.name,
                'format': config.format,
                'frequency': config.frequency,
                'enabled': config.enabled,
                'last_downloaded': last_dl.isoformat() if last_dl else None,
                'files_on_disk': file_count,
                'is_due': self._is_due(config),
                'description': config.description
            }

            # Enrich SACT v4 entry with quality phase info
            if source_id == 'sact_v4_patient_data':
                availability = self.check_sact_v4_availability()
                entry['sact_v4_availability'] = availability

            # Enrich NDRS CDH entry with CWT transition status
            if source_id == 'ndrs_cancer_data_hub':
                now = datetime.now()
                entry['cwt_transition'] = {
                    'cwt_decommission_date': self.CWT_DECOMMISSION_DATE.isoformat(),
                    'active': now >= self.CWT_DECOMMISSION_DATE,
                    'days_until_active': max(0, (self.CWT_DECOMMISSION_DATE - now).days),
                    'note': (
                        'ACTIVE — CWT successor'
                        if now >= self.CWT_DECOMMISSION_DATE
                        else (
                            f'Not yet active. CWT decommissions {self.CWT_DECOMMISSION_DATE.date()}. '
                            f'{max(0, (self.CWT_DECOMMISSION_DATE - now).days)} days remaining.'
                        )
                    ),
                }

            status[source_id] = entry

        return {
            'sources': status,
            'data_directory': str(self.data_dir),
            'total_ingestions': len(self.history),
            'last_check': self.history[-1]['timestamp'] if self.history else None
        }


def create_ingester(data_dir: str = None) -> NHSDataIngester:
    """Factory function for creating NHS data ingester."""
    return NHSDataIngester(data_dir)
