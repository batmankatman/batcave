# Activity Time Tracking Analysis

## Overview

This project analyzes personal time tracking data covering 8 weeks (4 pairs of two-week periods) of comprehensive activity logging from waking to sleeping hours. The analysis processes raw time-stamped activity logs and generates detailed metrics, visualizations, and quality reports.

## Data Structure

### Time Tracking Format
- **Granularity**: Activities logged with military time timestamps (HHMM format)
- **Coverage**: Complete daily coverage from wake-up to sleep
- **Duration**: 8 total weeks of data
- **Organization**: Sequential daily segments with header information

### Activity Categories

The system uses a structured categorization approach with the following primary categories:

- **GOD**: Spiritual and religious activities
- **P**: Program/academic work 
- **R**: Research activities
- **E**: Exercise and physical activities
- **S**: Social interactions and communication
- **F**: Fun and recreational activities
- **W**: Workout activities (distinct from general exercise)
- **LO**: Sleep/lights out marker

## Measurement Methodology

### Time Calculation
- Activities are measured in **minutes** with automatic duration calculation
- Durations computed based on timestamp intervals between consecutive entries
- Midnight crossovers handled automatically
- Sleep periods calculated from LO markers to day-end boundaries

### Data Processing
- **5-minute time slots**: Complete timeline broken into 5-minute increments
- **Weekly aggregation**: Data grouped into sequential 7-day blocks
- **Category totals**: Time spent per category tracked across all periods
- **Quality validation**: Automatic detection and flagging of timestamp inconsistencies

<!-- ## Output Artifacts

### Data Files
- `cleaned.csv`: Processed activity data with calculated durations
- `timesheet_5min.csv`: Expanded 5-minute slot timeline
- `weekly_category_totals.csv`: Category totals by week
- `metrics_summary.json`: Comprehensive time allocation metrics
- `quality_report.json`: Data validation and error reporting
-->
### Visualizations
- **Sequential block charts**: Stacked bar charts showing daily hours across 7-day periods
- **Category breakdown**: Visual representation of time allocation patterns
- **Weekly progression**: Tracking changes in activity patterns over time

## Key Metrics Tracked

- Total time per category across all weeks
- Weekly category distributions
- Daily activity patterns
- Sleep vs. awake time ratios
- Activity transition
---

*Analysis covers comprehensive personal time tracking with minute-level precision across 8 weeks of daily activity logging.*
