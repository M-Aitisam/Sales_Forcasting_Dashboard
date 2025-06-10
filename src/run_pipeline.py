# src/run_pipeline.py
import pandas as pd
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from visualizer import Visualizer
import joblib
import matplotlib.pyplot as plt  # Add this import

def main():
    # Initialize components
    loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Load and process data
    print("🚀 Loading and processing data...")
    df = loader.process_all_data()
    if df.empty:
        print("❌ Failed to process data")
        return
    
    # Feature engineering
    print("⚙️ Engineering features...")
    X = feature_engineer.prepare_features(df)
    y = df['Sales']
    
    # Train model
    print("🤖 Training model...")
    X_test, y_test, y_pred, metrics = model_trainer.train(X, y)
    
    # Print metrics
    print("\n📊 Model Performance:")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Get feature importance
    coef_df = model_trainer.get_coefficients(feature_engineer.feature_names)
    print("\n🔍 Top 5 Features:")
    print(coef_df.head())
    
    # Save model and artifacts
    model_trainer.save_model('../outputs/model.pkl')
    joblib.dump(feature_engineer.preprocessor, '../outputs/preprocessor.pkl')
    coef_df.to_csv('../outputs/feature_importance.csv', index=False)
    print("\n💾 Saved model and artifacts to outputs directory")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plt = visualizer.plot_actual_vs_predicted(y_test, y_pred)
    plt.savefig('../outputs/actual_vs_predicted.png')
    
    plt = visualizer.plot_feature_importance(coef_df)
    plt.savefig('../outputs/feature_importance.png')
    
    plt = visualizer.plot_sales_distribution(df)
    plt.savefig('../outputs/sales_distribution.png')
    
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()