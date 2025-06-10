# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        plt.grid(True)
        return plt
        
    @staticmethod
    def plot_feature_importance(coef_df):
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title('Feature Impact on Sales')
        plt.tight_layout()
        return plt
        
    @staticmethod
    def plot_sales_distribution(df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Sales'], bins=30, kde=True)
        plt.title('Sales Distribution')
        return plt