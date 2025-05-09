# Module 1: Data Ingestion & Cleaning

## Problem Statement
Retail stores collect massive amounts of sales data daily, but extracting meaningful insights requires:
1. Consolidating data from 1000+ stores
2. Cleaning inconsistencies (missing values, formatting issues)
3. Enriching with time-based features for forecasting

## Solution Approach
### Key Challenges Addressed
- **Data Heterogeneity**: Standardized formats across all stores
- **Missing Data**: Smart imputation strategies
- **Temporal Features**: Added business-relevant time dimensions

## Implementation

### File Structure

sales_forecasting_dashboard/
├── data/ # Raw data inputs
│ ├── stores.csv # Store metadata (primary file)
│ └── sales/ # Optional: Individual store sales data
│ ├── store_1.csv
│ └── store_2.csv
├── src/
│ ├── data_loader.py # Core data processing logic
│ └── test_data_loader.py # Validation script
└── outputs/
├── cleaned_data.csv # Processed output
└── reports/ # Data quality reports




### Key Components
1. **DataLoader Class** (`src/data_loader.py`)
   - `load_data()`: Robust CSV reader with error handling
   - `clean_store_data()`: Handles:
     - Missing CompetitionDistance → Median imputation
     - PromoInterval → List conversion
     - Date standardization
   - `add_time_features()`: Adds 25+ temporal features including:
     - DayOfWeek, IsWeekend, IsHoliday
     - Seasonality (Quarter/Month)
     - Business day flags

2. **Test Script** (`src/test_data_loader.py`)
   - Validates data pipeline
   - Generates quality reports
   - Saves processed data to `/outputs`

## Step-by-Step Execution

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
pandas==2.0.3
numpy==1.24.3
holidays==0.28
python-dateutil==2.8.2


2. Data Preparation
Place your input files:

Store metadata → data/stores.csv

Sales data (optional) → data/sales/store_*.csv

3. Run Processing Pipeline

python src/test_data_loader.py

Expected Output

✅ Successfully loaded 1115 stores
🔄 Processing time features...
✔ Added 25 temporal features
💾 Saved cleaned data to outputs/cleaned_data.csv