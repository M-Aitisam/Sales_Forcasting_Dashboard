# src/model_trainer.py
import pandas as pd  # Add this import at the top
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return X_test, y_test, y_pred, metrics
    
    def save_model(self, path: str):
        joblib.dump(self.model, path)
        
    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        return pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', ascending=False)