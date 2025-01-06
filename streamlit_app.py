import streamlit as st
import yfinance as yf
import pandas as pd

# App title
st.title("🎈 NVIDIA Financial History")
st.write(
    "This app displays NVIDIA's past two weeks of historical prices. For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# Function to fetch historical prices
@st.cache_data  # Cache the data to improve performance
def get_historical_prices(ticker: str, period: str = "14d", interval: str = "1d") -> pd.DataFrame:
    try:
        # Fetch historical data for the specified ticker
        ticker_data = yf.Ticker(ticker)
        historical_data = ticker_data.history(period=period, interval=interval)
        if historical_data.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        return historical_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# NVIDIA ticker
ticker = "NVDA"

# Fetch data
st.write(f"Fetching historical prices for ticker: **{ticker}**")
historical_data = get_historical_prices(ticker)

# Display data
if not historical_data.empty:
    st.write("### NVIDIA Historical Prices (Last 2 Weeks)")
    st.dataframe(historical_data)

    # Plot closing prices
    st.write("### Closing Price Chart")
    st.line_chart(historical_data['Close'], use_container_width=True)

    # Provide download option
    csv_data = historical_data.to_csv().encode('utf-8')
    st.download_button(
        label="Download Historical Data as CSV",
        data=csv_data,
        file_name=f"{ticker}_last_2_weeks.csv",
        mime="text/csv",
    )
else:
    st.error("No historical data available. Please check the ticker symbol or try again later.")

