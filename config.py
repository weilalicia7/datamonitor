"""
SACT Intelligent Scheduling System - Configuration
===================================================

Central configuration file for all system settings.
"""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

# =============================================================================
# PATHS
# =============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Subdirectories
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"
SCHEDULES_DIR = OUTPUT_DIR / "schedules"
REPORTS_DIR = OUTPUT_DIR / "reports"
DATA_CACHE_DIR = BASE_DIR / "data_cache"

# Create directories if they don't exist
for directory in [MODELS_DIR, TEMPLATES_DIR, OUTPUT_DIR, SCHEDULES_DIR, REPORTS_DIR, DATA_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================

# Objective weights (must sum to 1.0)
OPTIMIZATION_WEIGHTS = {
    'priority': 0.30,       # Maximize priority-weighted assignments
    'utilization': 0.25,    # Maximize chair utilization (earlier start times)
    'noshow_risk': 0.15,    # Minimize no-show risk exposure
    'waiting_time': 0.15,   # Minimize patient waiting time
    'robustness': 0.10,     # Schedule robustness (buffer between appointments)
    'travel': 0.05,         # Minimize patient travel distance
}

# Pareto frontier: pre-defined weight sets for multi-objective exploration
PARETO_WEIGHT_SETS = [
    {'name': 'balanced',       'priority': 0.30, 'utilization': 0.25, 'noshow_risk': 0.15, 'waiting_time': 0.15, 'robustness': 0.10, 'travel': 0.05},
    {'name': 'max_throughput', 'priority': 0.15, 'utilization': 0.45, 'noshow_risk': 0.10, 'waiting_time': 0.10, 'robustness': 0.10, 'travel': 0.10},
    {'name': 'patient_first',  'priority': 0.40, 'utilization': 0.10, 'noshow_risk': 0.10, 'waiting_time': 0.25, 'robustness': 0.05, 'travel': 0.10},
    {'name': 'risk_averse',    'priority': 0.20, 'utilization': 0.15, 'noshow_risk': 0.30, 'waiting_time': 0.10, 'robustness': 0.20, 'travel': 0.05},
    {'name': 'robust',         'priority': 0.20, 'utilization': 0.15, 'noshow_risk': 0.15, 'waiting_time': 0.10, 'robustness': 0.30, 'travel': 0.10},
]

# Time slot duration (minutes)
TIME_SLOT_MINUTES = 30

# Solver settings
SOLVER_TIME_LIMIT_SECONDS = 300  # 5 minutes default
SOLVER_NUM_WORKERS = 4           # Parallel workers

# Column generation settings (large instances >50 patients)
COLUMN_GEN_THRESHOLD = 50           # Switch from monolithic CP-SAT to CG above this
CG_MAX_ITERATIONS = 100             # Maximum pricing iterations
CG_REDUCED_COST_TOLERANCE = 1e-4   # Stop when best reduced cost < this
CG_SUBPROBLEM_TIME_LIMIT = 5.0     # Per-chair subproblem time limit (seconds)

# Buffer times (minutes)
DEFAULT_BUFFER_MINUTES = 0
ELEVATED_MODE_BUFFER_MINUTES = 15
CRISIS_MODE_BUFFER_MINUTES = 30

# Operating hours (24-hour format)
OPERATING_HOURS = (8, 18)  # 8am to 6pm

# Data directories
DATA_DIR = BASE_DIR / "data_cache"
MODEL_SAVE_DIR = MODELS_DIR

# Event classification keywords (alias for compatibility)
EVENT_CLASSIFICATION_KEYWORDS = EVENT_KEYWORDS if 'EVENT_KEYWORDS' in dir() else {}

# Severity weights for NLP scoring
SEVERITY_WEIGHTS = {
    'sentiment': 0.3,
    'classification': 0.5,
    'entity': 0.2
}

# Priority definitions.
#
# 4-tier oncology simplification of the NHS FSSA *Clinical Guide to
# Surgical Prioritisation* (5-tier P1a/P1b/P2/P3/P4).  The collapse
# merges FSSA's P1a (< 24 h) and P1b (< 72 h) into a single P1/P2
# split, which is the convention used across the dissertation prose
# (main.tex Sections 1.2, 2, and Results) and reflects oncology
# practice where the same-day-vs-3-day distinction is less load-
# bearing than in surgery.  See dissertation/PRIORITY_TIER_NOTE.md.
#
# max_delay_days is informational only -- the optimiser enforces
# P1_MAX_START_MIN (90 min from opening) plus per-patient
# earliest_time / latest_time set by callers.  No object reads
# the max_delay_days field at runtime; it is kept here as a single
# source of truth for documentation and audit purposes.
PRIORITY_DEFINITIONS = {
    1: {'name': 'Critical', 'max_delay_days': 1},   # FSSA P1a   (< 24 h)
    2: {'name': 'High',     'max_delay_days': 3},   # FSSA P1b   (< 72 h)
    3: {'name': 'Standard', 'max_delay_days': 14},  # FSSA P2    (< 1 month, cycle window)
    4: {'name': 'Routine',  'max_delay_days': 30}   # FSSA P3    (< 3 months)
}

# =============================================================================
# OPERATING MODES
# =============================================================================

class OperatingMode(Enum):
    """System operating modes based on event severity"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"
    EMERGENCY = "emergency"

# Severity thresholds for mode transitions
MODE_THRESHOLDS = {
    'elevated': 0.3,
    'crisis': 0.5,
    'emergency': 0.8
}

# =============================================================================
# PATIENT PRIORITIES
# =============================================================================

class PatientPriority(Enum):
    """Patient priority levels"""
    P1_CRITICAL = 1    # Curative intent, delay causes clinical harm
    P2_HIGH = 2        # Adjuvant therapy, moderate tolerance
    P3_STANDARD = 3    # Maintenance therapy, stable disease
    P4_DEFERRABLE = 4  # Palliative, symptom control

# Maximum delay (days) by priority -- mirrors PRIORITY_DEFINITIONS above
# and tracks the same FSSA-aligned timeframes (1 / 3 / 14 / 30 days).
# Informational only; not currently read at runtime.  See
# dissertation/PRIORITY_TIER_NOTE.md for the authoritative source.
PRIORITY_MAX_DELAY = {
    PatientPriority.P1_CRITICAL:   1,
    PatientPriority.P2_HIGH:       3,
    PatientPriority.P3_STANDARD:  14,
    PatientPriority.P4_DEFERRABLE: 30
}

# Hard CP-SAT constraint: P1 (curative-intent, life-threatening) patients must
# start within this many minutes of the day's opening when assigned.  Acts as a
# clinical-safety floor enforced at the solver level rather than as a soft
# preference; the soft priority weight in the objective continues to incentivise
# the broader P2-P4 ordering.  90 min = 1.5 h matches NHS-Wales same-morning
# practice for P1 SACT bookings.  Set to None to disable for ablation studies.
P1_MAX_START_MIN = 90

# =============================================================================
# ML MODEL SETTINGS
# =============================================================================

# No-show prediction thresholds
NOSHOW_THRESHOLDS = {
    'low': 0.10,       # < 10%: Normal booking
    'medium': 0.25,    # 10-25%: Send reminder
    'high': 0.40,      # 25-40%: Phone confirmation
    'very_high': 0.40  # > 40%: Double-book
}

# Model file names
NOSHOW_MODEL_FILE = MODELS_DIR / "noshow_model.pkl"
DURATION_MODEL_FILE = MODELS_DIR / "duration_model.pkl"
EVENT_IMPACT_MODEL_FILE = MODELS_DIR / "event_impact_model.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Minimum training data requirements
MIN_TRAINING_SAMPLES_NOSHOW = 500
MIN_TRAINING_SAMPLES_DURATION = 500

# =============================================================================
# EXTERNAL API SETTINGS
# =============================================================================

# Weather API (Open-Meteo - free, no key required)
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_UPDATE_INTERVAL = 3600  # 1 hour in seconds

# Default coordinates (Cardiff)
DEFAULT_LATITUDE = 51.4816
DEFAULT_LONGITUDE = -3.1791

# Traffic API (TomTom - free tier: 2,500 requests/day)
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "")
TOMTOM_API_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"
TRAFFIC_UPDATE_INTERVAL = 300  # 5 minutes in seconds

# RSS Feeds for news monitoring
NEWS_RSS_FEEDS = [
    {
        "name": "BBC Wales",
        "url": "https://feeds.bbci.co.uk/news/wales/rss.xml",
        "priority": 1
    },
    {
        "name": "Wales Online",
        "url": "https://www.walesonline.co.uk/news/rss.xml",
        "priority": 2
    },
    {
        "name": "Gov.uk Alerts",
        "url": "https://www.gov.uk/search/news-and-communications.atom",
        "priority": 1
    }
]
NEWS_UPDATE_INTERVAL = 900  # 15 minutes in seconds

# =============================================================================
# EVENT CLASSIFICATION
# =============================================================================

class EventType(Enum):
    """Types of events that can affect scheduling"""
    WEATHER_EVENT = "weather"
    TRAFFIC_INCIDENT = "traffic"
    HEALTH_ALERT = "health"
    SOCIAL_EVENT = "social"
    INFRASTRUCTURE = "infrastructure"
    EMERGENCY = "emergency"
    POLITICAL_EVENT = "political"
    SPORTS_EVENT = "sports"
    OTHER = "other"

# Keywords for event classification
EVENT_KEYWORDS = {
    EventType.WEATHER_EVENT: [
        'rain', 'snow', 'storm', 'flood', 'flooding', 'wind', 'ice', 'icy',
        'fog', 'weather', 'thunder', 'lightning', 'hail', 'blizzard', 'freeze'
    ],
    EventType.TRAFFIC_INCIDENT: [
        'accident', 'crash', 'collision', 'closure', 'closed', 'roadwork',
        'roadworks', 'traffic', 'delay', 'congestion', 'blocked', 'm4', 'a470',
        'motorway', 'junction'
    ],
    EventType.HEALTH_ALERT: [
        'nhs', 'hospital', 'outbreak', 'pandemic', 'virus', 'infection',
        'health', 'medical', 'emergency', 'ambulance', 'critical incident'
    ],
    EventType.SOCIAL_EVENT: [
        'festival', 'carnival', 'parade', 'march', 'protest', 'demonstration',
        'gathering', 'celebration', 'concert', 'event'
    ],
    EventType.SPORTS_EVENT: [
        'rugby', 'football', 'match', 'game', 'stadium', 'principality',
        'cardiff city', 'swansea', 'wales vs', 'six nations'
    ],
    EventType.INFRASTRUCTURE: [
        'power', 'outage', 'blackout', 'water', 'gas', 'maintenance',
        'engineering', 'works', 'construction'
    ],
    EventType.EMERGENCY: [
        'terror', 'attack', 'explosion', 'evacuation', 'lockdown', 'fire',
        'major incident', 'emergency services'
    ],
    EventType.POLITICAL_EVENT: [
        'election', 'summit', 'visit', 'royal', 'minister', 'parliament',
        'senedd', 'council'
    ]
}

# Severity modifiers for sentiment analysis
SEVERITY_KEYWORDS = {
    # Increase severity
    'critical': 0.30,
    'emergency': 0.30,
    'severe': 0.25,
    'major': 0.20,
    'serious': 0.20,
    'significant': 0.15,
    'dangerous': 0.20,
    'avoid': 0.15,
    'cancel': 0.20,
    'cancelled': 0.20,
    'closed': 0.15,
    'closure': 0.15,
    'evacuate': 0.30,
    'evacuated': 0.30,
    'fatal': 0.25,
    'death': 0.20,
    'casualties': 0.25,

    # Decrease severity
    'minor': -0.15,
    'small': -0.10,
    'brief': -0.10,
    'expected': -0.05,
    'planned': -0.10,
    'cleared': -0.20,
    'resolved': -0.25,
    'reopened': -0.20,
    'normal': -0.15
}

# =============================================================================
# EVENT IMPACT MATRIX
# =============================================================================

# Impact on no-show probability and duration by event type and severity
EVENT_IMPACT_MATRIX = {
    # (event_type, severity_level): (noshow_adjustment, duration_adjustment_minutes)
    (EventType.WEATHER_EVENT, 'low'): (0.05, 5),
    (EventType.WEATHER_EVENT, 'medium'): (0.15, 15),
    (EventType.WEATHER_EVENT, 'high'): (0.30, 30),
    (EventType.WEATHER_EVENT, 'critical'): (0.50, 60),

    (EventType.TRAFFIC_INCIDENT, 'low'): (0.05, 15),
    (EventType.TRAFFIC_INCIDENT, 'medium'): (0.10, 25),
    (EventType.TRAFFIC_INCIDENT, 'high'): (0.20, 45),
    (EventType.TRAFFIC_INCIDENT, 'critical'): (0.30, 60),

    (EventType.SOCIAL_EVENT, 'low'): (0.05, 10),
    (EventType.SOCIAL_EVENT, 'medium'): (0.10, 20),
    (EventType.SOCIAL_EVENT, 'high'): (0.15, 30),

    (EventType.SPORTS_EVENT, 'low'): (0.05, 15),
    (EventType.SPORTS_EVENT, 'medium'): (0.10, 25),
    (EventType.SPORTS_EVENT, 'high'): (0.15, 35),

    (EventType.HEALTH_ALERT, 'medium'): (0.15, 15),
    (EventType.HEALTH_ALERT, 'high'): (0.25, 20),
    (EventType.HEALTH_ALERT, 'critical'): (0.35, 30),

    (EventType.INFRASTRUCTURE, 'medium'): (0.20, 20),
    (EventType.INFRASTRUCTURE, 'high'): (0.35, 40),
    (EventType.INFRASTRUCTURE, 'critical'): (0.50, 60),

    (EventType.EMERGENCY, 'high'): (0.40, 45),
    (EventType.EMERGENCY, 'critical'): (0.60, 90),
}

# =============================================================================
# GEOGRAPHIC SETTINGS
# =============================================================================

# South Wales postcode districts with approximate coordinates
POSTCODE_COORDINATES = {
    'CF10': {'lat': 51.4780, 'lon': -3.1780, 'name': 'Cardiff City Centre'},
    'CF11': {'lat': 51.4700, 'lon': -3.2000, 'name': 'Cardiff Canton'},
    'CF14': {'lat': 51.5200, 'lon': -3.2100, 'name': 'Whitchurch/Rhiwbina'},
    'CF15': {'lat': 51.5100, 'lon': -3.2500, 'name': 'Radyr/Llandaff'},
    'CF23': {'lat': 51.5100, 'lon': -3.1400, 'name': 'Pentwyn/Llanedeyrn'},
    'CF24': {'lat': 51.4900, 'lon': -3.1600, 'name': 'Roath/Cathays'},
    'CF3': {'lat': 51.5000, 'lon': -3.1000, 'name': 'Rumney/St Mellons'},
    'CF5': {'lat': 51.4800, 'lon': -3.2400, 'name': 'Ely/Fairwater'},
    'CF31': {'lat': 51.5100, 'lon': -3.5800, 'name': 'Bridgend'},
    'CF32': {'lat': 51.5400, 'lon': -3.6000, 'name': 'Bridgend North'},
    'NP10': {'lat': 51.5800, 'lon': -3.0200, 'name': 'Newport East'},
    'NP19': {'lat': 51.5900, 'lon': -3.0000, 'name': 'Newport Central'},
    'NP20': {'lat': 51.5700, 'lon': -3.0300, 'name': 'Newport West'},
    'NP44': {'lat': 51.6500, 'lon': -3.0200, 'name': 'Cwmbran'},
    'SA1': {'lat': 51.6200, 'lon': -3.9400, 'name': 'Swansea Central'},
    'SA2': {'lat': 51.6100, 'lon': -3.9800, 'name': 'Swansea West'},
    'SA3': {'lat': 51.5700, 'lon': -4.0500, 'name': 'Gower'},
    'SA4': {'lat': 51.6600, 'lon': -4.0600, 'name': 'Llanelli Area'},
}

# Default sites configuration — Real Velindre Cancer Centre
# Chairs only in scheduling pool. CIU inpatient beds are separate ward.
DEFAULT_SITES = [
    {
        'code': 'WC',
        'name': 'Velindre Whitchurch (Day Unit)',
        'chairs': 19,
        'recliners': 4,
        'operating_hours': '08:30-18:00',
        'lat': 51.5200,
        'lon': -3.2100,
        'nurses_am': 10,
        'nurses_pm': 8,
        'type': 'main'
    },
    {
        'code': 'PCH',
        'name': 'Prince Charles Hospital (Macmillan Unit)',
        'chairs': 11,
        'recliners': 2,
        'operating_hours': '09:00-17:00',
        'lat': 51.7490,
        'lon': -3.3780,
        'nurses_am': 4,
        'nurses_pm': 3,
        'type': 'satellite'
    },
    {
        'code': 'RGH',
        'name': 'Royal Glamorgan Hospital (Outreach)',
        'chairs': 6,
        'recliners': 1,
        'operating_hours': '09:00-17:00',
        'lat': 51.5728,
        'lon': -3.3868,
        'nurses_am': 3,
        'nurses_pm': 2,
        'type': 'satellite'
    },
    {
        'code': 'POW',
        'name': 'Princess of Wales Hospital (Outreach)',
        'chairs': 6,
        'recliners': 1,
        'operating_hours': '09:00-17:00',
        'lat': 51.5040,
        'lon': -3.5760,
        'nurses_am': 3,
        'nurses_pm': 2,
        'type': 'satellite'
    },
    {
        'code': 'CWM',
        'name': 'Cwmbran Mobile Unit (Tenovus)',
        'chairs': 3,
        'recliners': 0,
        'operating_hours': '09:00-16:00',
        'lat': 51.6530,
        'lon': -3.0210,
        'nurses_am': 2,
        'nurses_pm': 1,
        'type': 'mobile'
    }
]

# =============================================================================
# UI SETTINGS
# =============================================================================

# Streamlit page config
PAGE_TITLE = "SACT Intelligent Scheduler"
PAGE_ICON = "🏥"
LAYOUT = "wide"

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb33',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',

    # Mode colors
    'normal': '#28a745',
    'elevated': '#ffc107',
    'crisis': '#fd7e14',
    'emergency': '#dc3545',

    # Utilization heatmap
    'util_low': '#ffffff',
    'util_medium': '#4a90d9',
    'util_high': '#1a5490'
}

# Refresh intervals (seconds)
UI_REFRESH_INTERVAL = 60
EVENT_REFRESH_INTERVAL = 300

# =============================================================================
# LOGGING
# =============================================================================

import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / 'sact_scheduler.log'

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)
