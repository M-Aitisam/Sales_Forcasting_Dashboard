import pandas as pd  # Add this import
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select relevant features
        features = df[['StoreType', 'Assortment', 'CompetitionDistance', 
                       'DayOfWeek', 'Month', 'Year', 'WeekOfYear', 'IsWeekend', 'IsHoliday']]
        
        # Define preprocessing
        categorical_cols = ['StoreType', 'Assortment']
        numerical_cols = ['CompetitionDistance', 'DayOfWeek', 'Month', 'Year', 'WeekOfYear', 'IsWeekend', 'IsHoliday']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Fit and transform
        processed = preprocessor.fit_transform(features)
        
        # Get feature names
        num_features = numerical_cols
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        self.feature_names = num_features + list(cat_features)
        
        self.preprocessor = preprocessor
        return pd.DataFrame(processed, columns=self.feature_names)