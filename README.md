# 📈 NSE Data Toolkit

Downloads real NSE stock data, cleans it, analyses it,
and exports a full portfolio report to CSV and Excel.

## What it does
- Downloads 1 year of live NSE data via yfinance
- Cleans missing values and duplicates
- Calculates returns, Sharpe Ratio, correlations
- Finds most volatile days in the portfolio
- Monthly and weekday return pattern analysis
- 20-day rolling moving average and volatility
- Exports full report to CSV and Excel

## How to run
pip install numpy pandas yfinance openpyxl
python nse_data_toolkit.py

## Stocks tracked
RELIANCE, TCS, INFY, HDFCBANK

## Author
Shivam Tyagi | github.com/QuantShivam
