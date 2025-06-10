# src/data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from pathlib import Path

class DataLoader:
    def __init__(self, data_path: str = '../data/', output_path: str = '../outputs/'):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.country_holidays = holidays.CountryHoliday('DE')  # Germany
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_data(self, filename: str) -> pd.DataFrame:
        file_path = self.data_path / filename
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded data from {file_path}")
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return pd.DataFrame()

    def clean_store_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        # Handle missing values
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
            df['CompetitionDistance'].median()
        ).astype(int)
        
        # Process competition dates
        comp_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        for col in comp_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        # Create CompetitionOpenDate
        if all(col in df.columns for col in comp_cols):
            df['CompetitionOpenDate'] = pd.to_datetime(
                df['CompetitionOpenSinceYear'].astype(str) + '-' +
                df['CompetitionOpenSinceMonth'].astype(str) + '-01',
                errors='coerce'
            )
            df['CompetitionOpen'] = df['CompetitionOpenDate'].notna()
        
        # Process promotions
        if 'PromoInterval' in df.columns:
            df['PromoInterval'] = df['PromoInterval'].apply(
                lambda x: [m.strip() for m in x.split(',')] if pd.notna(x) and isinstance(x, str) else []
            )
            df['HasPromo'] = df['PromoInterval'].apply(len) > 0
        
        # Convert categorical data
        cat_cols = ['StoreType', 'Assortment']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df

    def generate_sales_data(self, store_df: pd.DataFrame) -> pd.DataFrame:
        sales_data = []
        np.random.seed(42)
        
        for store_id in store_df['Store']:
            store_info = store_df[store_df['Store'] == store_id].iloc[0]
            
            # Generate 2 years of daily data
            for days_back in range(730, 0, -1):
                date = datetime.now() - pd.Timedelta(days=days_back)
                
                # Base sales based on store type
                base_sales = {
                    'a': 8000, 
                    'b': 10000,
                    'c': 6000, 
                    'd': 4000
                }.get(store_info['StoreType'], 5000)
                
                # Sales modifiers
                comp_effect = max(0.7, 1 - (1000/store_info['CompetitionDistance'])) if store_info['CompetitionDistance'] > 0 else 1
                promo_effect = 1.3 if store_info['HasPromo'] and date.month in [1,4,7,10] else 1
                day_effect = 1.5 if date.weekday() in [5,6] else 1  # Weekends
                season_effect = 1.2 if date.month in [11, 12] else 1  # Holiday season
                
                sales = base_sales * comp_effect * promo_effect * day_effect * season_effect * np.random.uniform(0.9, 1.1)
                
                sales_data.append({
                    'Store': store_id,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Sales': sales,
                    'Customers': int(sales/10),
                    'Open': 1,
                    'Promo': 1 if promo_effect > 1 else 0
                })
        
        return pd.DataFrame(sales_data)

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'Date' not in df.columns:
            return df
            
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Time features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayName'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsHoliday'] = df['Date'].apply(lambda x: x in self.country_holidays).astype(int)
        
        return df

    def merge_data(self, store_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        if store_df.empty:
            return sales_df
        if sales_df.empty:
            return store_df
            
        return pd.merge(
            sales_df,
            store_df,
            on='Store',
            how='left'
        )

    def process_all_data(self) -> pd.DataFrame:
        # Load and clean store data
        store_df = self.load_data('stores.csv')
        store_df = self.clean_store_data(store_df)
        
        # Generate sales data
        sales_df = self.generate_sales_data(store_df)
        sales_df = self.add_time_features(sales_df)
        
        # Merge datasets
        combined_df = self.merge_data(store_df, sales_df)
        
        # Save results
        self.save_results(combined_df)
        
        return combined_df

    def save_results(self, df: pd.DataFrame, filename: str = 'cleaned_sales_data.csv') -> bool:
        if df.empty:
            return False
            
        output_file = self.output_path / filename
        df.to_csv(output_file, index=False)
        print(f"💾 Saved cleaned data to {output_file}")
        return True