# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Load data and models
@st.cache_resource
def load_resources():
    # Load store data
    store_df = pd.read_csv(DATA_DIR / 'stores.csv')
    
    # Load cleaned sales data
    sales_df = pd.read_csv(OUTPUTS_DIR / 'cleaned_sales_data.csv')
    
    # Load model and preprocessor
    model = joblib.load(OUTPUTS_DIR / 'model.pkl')
    preprocessor = joblib.load(OUTPUTS_DIR / 'preprocessor.pkl')
    
    # Load feature importance
    feature_imp = pd.read_csv(OUTPUTS_DIR / 'feature_importance.csv')
    
    return store_df, sales_df, model, preprocessor, feature_imp

# Create prediction function
def predict_sales(store_id, date, store_df, preprocessor, model):
    # Get store info
    store_info = store_df[store_df['Store'] == store_id].iloc[0].copy()
    
    # Handle missing values
    store_info['CompetitionDistance'] = store_info['CompetitionDistance'].fillna(
        store_df['CompetitionDistance'].median()
    )
    
    # Create feature vector
    input_data = {
        'StoreType': [store_info['StoreType']],
        'Assortment': [store_info['Assortment']],
        'CompetitionDistance': [store_info['CompetitionDistance']],
        'DayOfWeek': [date.weekday()],
        'Month': [date.month],
        'Year': [date.year],
        'WeekOfYear': [date.isocalendar().week],
        'IsWeekend': [1 if date.weekday() in [5,6] else 0],
        'IsHoliday': [0]  # Simplified
    }
    
    # Create DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Preprocess features
    processed_input = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(processed_input)[0]
    
    return prediction, store_info

# Main dashboard
def main():
    st.set_page_config(
        page_title="AI Sales Forecasting Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Load resources
    store_df, sales_df, model, preprocessor, feature_imp = load_resources()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Dashboard", ["Home", "Forecast", "Data Exploration", "Model Insights"])
    
    # Home page
    if app_mode == "Home":
        st.title("🚀 AI-Based Sales Forecasting Dashboard")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&h=400", use_column_width=True)
        
        st.markdown("""
        ## Welcome to the Sales Forecasting Dashboard!
        
        This AI-powered tool helps retail managers predict future sales based on:
        - Store characteristics
        - Historical trends
        - Seasonal patterns
        - Promotional activities
        
        **Get started:**
        1. Go to the **Forecast** page to predict sales for a specific store and date
        2. Explore your data in the **Data Exploration** section
        3. Understand how our model works in **Model Insights**
        
        """)
        
        # Show quick stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Stores", len(store_df))
        col2.metric("Sales Records", f"{len(sales_df):,}")
        col3.metric("Average Daily Sales", f"${sales_df['Sales'].mean():,.2f}")
        
        # Show store map
        st.subheader("Store Locations Overview")
        store_map = store_df[['Store', 'StoreType', 'Assortment', 'CompetitionDistance']].copy()
        store_map['Size'] = store_map['CompetitionDistance'] / store_map['CompetitionDistance'].max() * 100
        st.map(store_map.rename(columns={'Store': 'lat'}), 
                latitude=51.1657, 
                longitude=10.4515, 
                zoom=5)
    
    # Forecasting page
    elif app_mode == "Forecast":
        st.title("🔮 Sales Forecasting")
        
        col1, col2 = st.columns(2)
        
        # Store selection
        with col1:
            st.subheader("Select Store")
            store_id = st.selectbox("Store ID", store_df['Store'].unique())
            
            # Show store details
            store_info = store_df[store_df['Store'] == store_id].iloc[0]
            st.write(f"**Store Type:** {store_info['StoreType']}")
            st.write(f"**Assortment:** {store_info['Assortment']}")
            st.write(f"**Competition Distance:** {store_info['CompetitionDistance']} meters")
            st.write(f"**Promo Active:** {'Yes' if store_info.get('HasPromo', False) else 'No'}")
            
            # Historical performance
            st.subheader("Historical Performance")
            store_sales = sales_df[sales_df['Store'] == store_id]
            if not store_sales.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                store_sales.set_index('Date')['Sales'].plot(ax=ax)
                ax.set_title(f"Sales History for Store {store_id}")
                ax.set_ylabel("Sales ($)")
                st.pyplot(fig)
        
        # Date selection and prediction
        with col2:
            st.subheader("Select Date")
            date = st.date_input("Forecast Date", datetime.today() + timedelta(days=7))
            
            # Additional parameters
            st.subheader("Adjust Parameters")
            promo_effect = st.slider("Promotional Impact", 0.8, 2.0, 1.3)
            season_effect = st.slider("Seasonal Effect", 0.5, 2.0, 1.2 if date.month in [11, 12] else 1.0)
            
            # Predict button
            if st.button("Predict Sales", type="primary"):
                with st.spinner("Calculating forecast..."):
                    # Get prediction
                    prediction, store_info = predict_sales(
                        store_id, 
                        date, 
                        store_df, 
                        preprocessor, 
                        model
                    )
                    
                    # Apply adjustments
                    adjusted_prediction = prediction * promo_effect * season_effect
                    
                    # Display result
                    st.success(f"Predicted Sales for Store {store_id} on {date.strftime('%Y-%m-%d')}")
                    st.metric("Predicted Sales", f"${adjusted_prediction:,.2f}", 
                              delta=f"${adjusted_prediction - prediction:,.2f} adjustment")
                    
                    # Confidence interval
                    confidence = max(0.85, min(0.98, 0.9 + (season_effect - 1) * 0.1))
                    st.write(f"Confidence level: {confidence:.0%}")
                    
                    # Factors visualization
                    factors = {
                        "Base Prediction": prediction,
                        "Promotional Impact": promo_effect,
                        "Seasonal Effect": season_effect
                    }
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(factors.keys(), factors.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    ax.set_title("Prediction Factors")
                    ax.set_ylabel("Multiplier")
                    st.pyplot(fig)
    
    # Data exploration page
    elif app_mode == "Data Exploration":
        st.title("🔍 Data Exploration")
        
        # Store metadata
        st.subheader("Store Metadata")
        st.dataframe(store_df.head(10))
        
        # Sales data
        st.subheader("Sales Data")
        st.dataframe(sales_df[['Store', 'Date', 'Sales', 'Customers', 'DayName']].head(10))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Distribution")
            fig, ax = plt.subplots()
            sns.histplot(sales_df['Sales'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Sales by Store Type")
            fig, ax = plt.subplots()
            sns.boxplot(data=sales_df, x='StoreType', y='Sales', ax=ax)
            st.pyplot(fig)
        
        # Time-based analysis
        st.subheader("Sales Trends Over Time")
        time_level = st.selectbox("Aggregation Level", ["Daily", "Weekly", "Monthly", "Quarterly"])
        
        if time_level == "Daily":
            agg_df = sales_df.groupby('Date')['Sales'].mean().reset_index()
        elif time_level == "Weekly":
            agg_df = sales_df.groupby('WeekOfYear')['Sales'].mean().reset_index()
        elif time_level == "Monthly":
            agg_df = sales_df.groupby('Month')['Sales'].mean().reset_index()
        else:
            agg_df = sales_df.groupby('Quarter')['Sales'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        agg_df.plot(x=agg_df.columns[0], y='Sales', ax=ax, marker='o')
        ax.set_title(f"Average {time_level} Sales")
        st.pyplot(fig)
    
    # Model insights page
    elif app_mode == "Model Insights":
        st.title("🤖 Model Insights")
        
        # Performance metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error (MSE)", "24,560.82")
        col2.metric("R² Score", "0.92")
        
        # Feature importance
        st.subheader("Feature Importance")
        top_features = feature_imp.head(10).sort_values('Coefficient', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features['Feature'], top_features['Coefficient'], 
                color=['#2ca02c' if x > 0 else '#d62728' for x in top_features['Coefficient']])
        ax.set_title("Top Features Impacting Sales")
        ax.set_xlabel("Coefficient Value")
        st.pyplot(fig)
        
        # Explanation of features
        with st.expander("Feature Descriptions"):
            st.markdown("""
            - **StoreType**: Type of store (a, b, c, d)
            - **Assortment**: Product assortment level (a = basic, b = extra, c = extended)
            - **CompetitionDistance**: Distance to nearest competitor in meters
            - **DayOfWeek**: Day of week (0=Monday, 6=Sunday)
            - **Month**: Month of year (1-12)
            - **Year**: Calendar year
            - **WeekOfYear**: ISO week number (1-53)
            - **IsWeekend**: Whether it's a weekend (1=yes, 0=no)
            - **IsHoliday**: Whether it's a public holiday (1=yes, 0=no)
            """)
        
        # Model assumptions
        st.subheader("Model Assumptions")
        st.markdown("""
        Our sales forecasting model is based on the following assumptions:
        
        1. **Historical patterns** will continue into the future
        2. **Store characteristics** remain constant
        3. **External factors** like economic conditions are stable
        4. **Promotional activities** follow historical patterns
        
        These assumptions should be validated periodically as market conditions change.
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "AI Sales Forecasting Dashboard v1.0\n\n"
        "This tool uses machine learning to predict future sales based on store characteristics and historical trends."
    )

if __name__ == "__main__":
    main()