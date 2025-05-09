# src/data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import os
from pathlib import Path
from typing import Optional, Dict, List

class DataLoader:
    def __init__(self, data_path: str = './data/', output_path: str = './outputs/'):
        """
        Initialize the DataLoader with paths.
        
        Args:
            data_path: Path to directory containing input data
            output_path: Path to directory for saving processed data
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.country_holidays = holidays.CountryHoliday('US')  # Change to your country
        
        # Create directories if they don't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file with robust error handling"""
        file_path = self.data_path / filename
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded data from {file_path}")
            return df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            return None

    def clean_store_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive cleaning of store metadata"""
        if df is None or df.empty:
            return df

        df = df.copy()

        # 1. Handle missing values
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
            df['CompetitionDistance'].median()).astype(int)

        # 2. Process competition dates
        comp_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        for col in comp_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        if all(col in df.columns for col in comp_cols):
            df['CompetitionOpenDate'] = pd.to_datetime(
                df['CompetitionOpenSinceYear'].astype(str) + '-' +
                df['CompetitionOpenSinceMonth'].astype(str) + '-01',
                errors='coerce'
            )
            df['CompetitionOpen'] = df['CompetitionOpenDate'].notna()

        # 3. Process promotions
        if 'PromoInterval' in df.columns:
            df['PromoInterval'] = df['PromoInterval'].apply(
                lambda x: [m.strip() for m in x.split(',')] if pd.notna(x) and isinstance(x, str) else [])
            df['HasPromo'] = df['PromoInterval'].apply(len) > 0

        # 4. Convert categorical data
        cat_cols = ['StoreType', 'Assortment']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    def add_time_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """Add comprehensive time-based features"""
        if df is None or date_col not in df.columns:
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # 1. Basic date features
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['DayName'] = df[date_col].dt.day_name()
        df['Month'] = df[date_col].dt.month
        df['Year'] = df[date_col].dt.year
        df['WeekOfYear'] = df[date_col].dt.isocalendar().week
        df['Quarter'] = df[date_col].dt.quarter

        # 2. Business features
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsHoliday'] = df[date_col].apply(lambda x: x in self.country_holidays).astype(int)
        df['IsBusinessDay'] = ((df['IsWeekend'] == 0) & (df['IsHoliday'] == 0)).astype(int)

        # 3. Promo features (if promo data exists)
        if 'Promo' in df.columns:
            df['IsPromoDay'] = df['Promo'].astype(int)

        return df

    def process_sales_data(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance sales data"""
        if sales_df is None or sales_df.empty:
            return sales_df

        sales_df = sales_df.copy()

        # 1. Handle missing values
        numeric_cols = ['Sales', 'Customers', 'Open', 'Promo']
        for col in numeric_cols:
            if col in sales_df.columns:
                if col in ['Open', 'Promo']:
                    sales_df[col] = sales_df[col].fillna(0).astype(int)
                else:
                    sales_df[col] = sales_df[col].fillna(sales_df[col].median())

        # 2. Add derived metrics
        if all(col in sales_df.columns for col in ['Sales', 'Customers']):
            sales_df['SalesPerCustomer'] = sales_df['Sales'] / sales_df['Customers']
            sales_df['SalesPerCustomer'] = sales_df['SalesPerCustomer'].replace([np.inf, -np.inf], np.nan)

        return sales_df

    def merge_data(self, store_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Merge store metadata with sales data"""
        if store_df is None or store_df.empty:
            return sales_df
        if sales_df is None or sales_df.empty:
            return store_df

        return pd.merge(
            sales_df,
            store_df,
            on='Store',
            how='left'
        )

    def process_all_data(self, store_file: str = 'stores.csv', sales_file: str = 'sales.csv') -> Optional[pd.DataFrame]:
        """Complete data processing pipeline"""
        # 1. Load and clean store data
        store_df = self.load_data(store_file)
        store_df = self.clean_store_data(store_df)

        # 2. Load and process sales data if available
        sales_df = self.load_data(sales_file)
        if sales_df is not None:
            sales_df = self.process_sales_data(sales_df)
            sales_df = self.add_time_features(sales_df)
            combined_df = self.merge_data(store_df, sales_df)
        else:
            combined_df = store_df
            print("ℹ️ No sales data found - processing store metadata only")

        # 3. Final processing
        if combined_df is not None:
            if 'Date' in combined_df.columns:
                combined_df = combined_df.sort_values(['Store', 'Date'])
            else:
                combined_df = combined_df.sort_values('Store')
            combined_df.reset_index(drop=True, inplace=True)

        return combined_df

    def save_results(self, df: pd.DataFrame, filename: str = 'cleaned_data.csv') -> bool:
        """Save processed data to output directory"""
        if df is None or df.empty:
            print("⚠️ No data to save")
            return False

        output_file = self.output_path / filename
        try:
            df.to_csv(output_file, index=False)
            print(f"💾 Data successfully saved to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save data: {str(e)}")
            return False