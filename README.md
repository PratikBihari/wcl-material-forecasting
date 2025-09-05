# WCL Material Procurement Forecasting System

*Western Coalfields Limited - Material Management Department*

A specialized web application for Western Coalfields Limited (WCL) to forecast material procurement needs on a Financial Year (FY) basis.

## Purpose

Helps WCL's Material Management Department:
- **Plan annual procurement budgets** with accurate FY-based forecasts
- **Ensure timely availability** of critical consumables
- **Predict demand** using historical consumption patterns
- **Avoid shortages, overstocking, and emergency purchases**

## Key Features

### A. Data Upload Module
- **Excel/CSV Support**: Upload 2-3 years of historical data
- **Auto-cleaning**: Automatic data validation and outlier removal
- **Format Support**: .xlsx, .xls, and .csv files
- **Data Validation**: Ensures required columns and data quality

### B. Advanced Forecasting Engine
- **Multiple Models**: Prophet (recommended), ARIMA, Linear Regression
- **FY-Based Predictions**: 12-month forecasts aligned with Indian Financial Year
- **Seasonality Detection**: Identifies patterns (e.g., HDPE pipes before monsoon)
- **Smart Alerts**: Seasonal demand warnings and procurement recommendations

### C. Comprehensive Dashboard & Reports
- **Interactive Charts**: Historical data with forecast visualization
- **Budget Planning**: Automatic cost calculations with safety buffers
- **Stock Analysis**: Current inventory status and consumption trends
- **Report Downloads**: Excel and PDF format reports
- **Alert System**: Critical stock levels and seasonal demand notifications

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open browser to `http://localhost:5000`

## Data Requirements

Upload file must contain:
- `date`: Date in YYYY-MM-DD format (FY cycle: April to March)
- `consumption`: Monthly material consumption (units)
- `stock`: Current stock levels (units)
- `pending_orders`: Pending order quantities (units)

## Usage

1. **Upload Data**: Select Excel/CSV file with 2-3 years of historical data
2. **Configure**: Set unit cost and choose forecasting model
3. **Generate Forecast**: Get 12-month FY predictions with budget analysis
4. **Review Alerts**: Check seasonal and stock level warnings
5. **Download Reports**: Export detailed Excel/PDF reports

## ML Models

### Prophet (Recommended)
- Best for seasonal patterns
- Handles missing data well
- Automatic trend detection

### ARIMA
- Traditional time series forecasting
- Good for stable trends
- Statistical approach

### Linear Regression
- Fast and simple
- Uses engineered features
- Baseline model

## Sample Data

Use `sample_data.xlsx` with 2+ years of HDPE pipe consumption data showing seasonal patterns (high demand in May-June before monsoon).

## Seasonal Intelligence

The system automatically detects:
- **Pre-monsoon demand** (May-June) for pipes and drainage materials
- **Winter maintenance** periods requiring higher consumption
- **Quarterly patterns** aligned with mining operations

## Report Features

### Excel Reports
- Historical data sheet
- 12-month forecast breakdown
- Budget summary with safety margins
- Monthly procurement schedule

### PDF Reports
- Executive summary
- Key alerts and recommendations
- Budget overview
- Visual charts and graphs