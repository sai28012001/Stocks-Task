import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('Joined_Transactional_Customer_Product.csv')
    # Clean the ProductName column - replace None and 'None' with StockCode
    df['ProductName'] = df['ProductName'].replace(['None', None], np.nan)
    df['ProductName'] = df.apply(lambda x: f"Stock Code: {x['StockCode']}" 
                                if pd.isna(x['ProductName']) else x['ProductName'], axis=1)
    return df

# Prepare the data
def prepare_data(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate total revenue per product (Quantity * Price)
    df['Revenue'] = df['Quantity'] * df['Price']
    
    # Get top 10 products by revenue
    product_totals = df.groupby('StockCode').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'ProductName': 'first'  # Get the first product name for each stock code
    }).sort_values('Revenue', ascending=False).head(10)
    
    # Create a dictionary of top products with their descriptions
    top_product_info = {}
    for idx, row in product_totals.iterrows():
        total_revenue = row['Revenue']
        total_quantity = row['Quantity']
        product_name = row['ProductName']
        top_product_info[idx] = (
            f"Code: {idx} - {product_name}\n"
            f"Total Revenue: £{total_revenue:,.2f} | "
            f"Total Quantity: {total_quantity:,.0f}"
        )
    
    return list(product_totals.index), df, top_product_info

# Forecasting function
def forecast_demand(df, stock_code):
    """
    Generate demand forecast with adaptive model selection based on data availability
    """
    # Filter data for the selected stock code and prepare time series
    stock_data = df[df['StockCode'] == stock_code].copy()
    
    # Create weekly time series of quantity
    stock_data = stock_data.set_index('InvoiceDate')
    weekly_data = stock_data.resample('W')['Quantity'].sum().fillna(0)
    
    # Calculate the number of weeks in the data
    n_weeks = len(weekly_data)
    
    if n_weeks < 4:
        st.error(f"Not enough data points for stock code {stock_code}. Need at least 4 weeks of data.")
        return None, None, None, None
    
    try:
        # Split data into train and test sets
        train_size = int(len(weekly_data) * 0.8)
        train = weekly_data[:train_size]
        test = weekly_data[train_size:]
        
        # Determine appropriate model based on data length
        if n_weeks >= 104:  # If we have 2 years or more of data
            model = ExponentialSmoothing(
                train,
                trend='add',
                seasonal='add',
                seasonal_periods=52  # Weekly seasonality
            )
            st.info("Using seasonal model with weekly seasonality")
        elif n_weeks >= 52:  # If we have 1 year or more of data
            model = ExponentialSmoothing(
                train,
                trend='add',
                seasonal='add',
                seasonal_periods=12  # Monthly-like seasonality
            )
            st.info("Using seasonal model with monthly seasonality")
        else:  # For shorter periods
            model = ExponentialSmoothing(
                train,
                trend='add',
                seasonal=None  # No seasonality for short periods
            )
            st.info("Using trend-only model (insufficient data for seasonality)")
        
        # Fit model with optimal parameters
        model_fit = model.fit(optimized=True)
        
        # Generate forecasts
        forecast_horizon = min(15, n_weeks)  # Adjust forecast horizon based on data availability
        forecast = model_fit.forecast(forecast_horizon)
        
        return train, test, forecast, model_fit
    
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        st.info("Attempting alternative modeling approach...")
        
        try:
            # Fallback to simple exponential smoothing
            model = ExponentialSmoothing(
                train,
                trend=None,
                seasonal=None
            )
            model_fit = model.fit()
            forecast = model_fit.forecast(min(15, n_weeks))
            
            st.warning("Using simple exponential smoothing due to data limitations")
            return train, test, forecast, model_fit
            
        except Exception as e:
            st.error(f"Alternative approach failed: {str(e)}")
            return None, None, None, None

def plot_demand(train, test, forecast, product_info):
    """
    Enhanced plotting function with better error handling and visual improvements
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.plot(train.index, train, label='Historical Data (Training)', 
            color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot test data if available
    if len(test) > 0:
        ax.plot(test.index, test, label='Historical Data (Test)', 
                color='orange', linewidth=2, marker='o', markersize=4)
    
    # Plot forecast
    ax.plot(forecast.index, forecast, label=f'{len(forecast)}-Week Forecast', 
            color='green', linestyle='--', linewidth=2)
    
    # Add confidence intervals if available
    if hasattr(forecast, 'conf_int'):
        ci = forecast.conf_int()
        ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], 
                       color='green', alpha=0.1, label='95% Confidence Interval')
    
    # Customize the plot
    ax.set_title(f'Demand Forecast\n{product_info}', pad=20, wrap=True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Quantity')
    ax.grid(True, alpha=0.3)
    
    # Improve legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def plot_error_distributions(train, test, model_fit):
    """
    Enhanced error distribution plotting with better handling of limited data
    """
    # Calculate errors
    train_predictions = model_fit.fittedvalues
    train_errors = train - train_predictions
    
    # Create figure
    if len(test) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        test_predictions = model_fit.forecast(len(test))
        test_errors = test - test_predictions
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    # Plot training errors
    ax1.hist(train_errors, bins=min(30, len(train_errors)//2), 
             color='blue', alpha=0.7, density=True)
    ax1.set_title('Training Error Distribution')
    ax1.set_xlabel('Error (Actual - Predicted)')
    ax1.set_ylabel('Density')
    
    # Add training error statistics
    train_stats = (f'Mean Error: {train_errors.mean():.2f}\n'
                  f'Std Dev: {train_errors.std():.2f}\n'
                  f'RMSE: {np.sqrt(np.mean(train_errors**2)):.2f}')
    
    ax1.text(0.95, 0.95, train_stats, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot test errors if available
    if len(test) > 0:
        ax2.hist(test_errors, bins=min(30, len(test_errors)//2), 
                 color='orange', alpha=0.7, density=True)
        ax2.set_title('Test Error Distribution')
        ax2.set_xlabel('Error (Actual - Predicted)')
        ax2.set_ylabel('Density')
        
        # Add test error statistics
        test_stats = (f'Mean Error: {test_errors.mean():.2f}\n'
                     f'Std Dev: {test_errors.std():.2f}\n'
                     f'RMSE: {np.sqrt(np.mean(test_errors**2)):.2f}')
        
        ax2.text(0.95, 0.95, test_stats, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Streamlit app
def main():
    st.title('Product Demand Forecasting Dashboard')
    
    try:
        # Load and prepare data
        data = load_data()
        top_products, prepared_data, product_info = prepare_data(data)
        
        # Display data summary
        st.subheader('Data Overview')
        st.write(f"Date Range: {prepared_data['InvoiceDate'].min().date()} to {prepared_data['InvoiceDate'].max().date()}")
        st.write(f"Total Products: {len(prepared_data['StockCode'].unique()):,}")
        st.write(f"Total Transactions: {len(prepared_data['TransactionID'].unique()):,}")
        
        # Create a selectbox with product descriptions
        st.subheader('Select a Product')
        selected_product = st.selectbox(
            'Choose from top 10 products by revenue:',
            options=top_products,
            format_func=lambda x: product_info[x]
        )
        
        if selected_product:
            # Show loading message
            with st.spinner('Generating forecast...'):
                # Generate forecast
                train_data, test_data, forecast_data, model_fit = forecast_demand(
                    prepared_data, selected_product)
                
                if train_data is not None:
                    # Display demand forecast plot
                    st.subheader('Historical and Forecast Demand')
                    fig1 = plot_demand(train_data, test_data, forecast_data, 
                                     product_info[selected_product])
                    st.pyplot(fig1)
                    
                    # Display error distributions
                    st.subheader('Forecast Error Analysis')
                    fig2 = plot_error_distributions(train_data, test_data, model_fit)
                    st.pyplot(fig2)
                    
                    # Display forecast values
                    st.subheader('15-Week Forecast Values')
                    forecast_df = pd.DataFrame({
                        'Week Starting': forecast_data.index.strftime('%Y-%m-%d'),
                        'Forecasted Quantity': forecast_data.values.round(0)
                    })
                    st.dataframe(forecast_df)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure your data file is in the correct location and format.")

if __name__ == "__main__":
    main()