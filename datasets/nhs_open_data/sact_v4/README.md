# SACT v4.0 Patient-Level Data

Place SACT v4.0 CSV files here to trigger automatic model recalibration.

## How it works

The auto-learning scheduler checks this directory every 24 hours.
When a CSV file is detected, it:

1. Reads the data quality phase from the current date:
   - **April–June 2026** (`preliminary`) → Level 1-2 recalibration (partial trusts only)
   - **July 2026** (`conformance`) → Level 2 recalibration
   - **August 2026+** (`complete`) → Level 3 full retrain of all 12 ML models

2. Merges SACT v4.0 data with historical training data
3. Triggers the appropriate recalibration level
4. Saves model version snapshot before retraining

## To trigger immediately (don't wait 24h)

```
POST /api/data/sact-v4/check
```

Or check status:
```
GET /api/data/sact-v4/status
```

## Expected CSV format

SACT v4.0 format: 60 fields across 7 sections.
See `docs/NHS_OPEN_DATA_INVENTORY.md` for field details.

Required columns for ML training:
- `Attended_Status` (Yes/No/Cancelled)
- `Planned_Duration` (minutes)
- `Actual_Duration` (minutes)
- `Patient_ID`
- `Cycle_Number`

## Timeline

| Phase | Date | Quality | ML Action |
|-------|------|---------|-----------|
| Rollout | Apr–Jun 2026 | `preliminary` | Level 1-2 |
| Full conformance | Jul 2026 | `conformance` | Level 2 |
| First complete | Aug 2026 | `complete` | Level 3 |

*Data collection commenced 1 April 2026. First complete dataset expected August 2026.*
