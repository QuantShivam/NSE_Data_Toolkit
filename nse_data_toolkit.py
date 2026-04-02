"""
NSE Data Toolkit — Portfolio Analysis Report Generator
=======================================================
Author  : Shivam Tyagi
GitHub  : github.com/QuantShivam
Module  : 3 — Data Handling with Python
Tools   : NumPy, pandas, yfinance

What this does
--------------
Downloads real NSE stock data, cleans it, runs analysis
(returns, volatility, correlations, monthly patterns),
and saves a complete report to CSV + Excel.

I built this to practice every pandas topic from Module 3
using stocks I actually follow on NSE.

How to run
----------
    python nse_data_toolkit.py

Topics used (Module 3)
----------------------
 1. NumPy arrays & vectorized math
 2. pandas Series
 3. pandas DataFrame
 4. Downloading live data (yfinance)
 5. Data inspection (info, describe, isnull)
 6. Data selection (loc, iloc, filtering)
 7. Data cleaning (ffill, dropna, drop_duplicates)
 8. Column operations (pct_change, rolling, abs)
 9. GroupBy (split-apply-combine)
10. Merge & Join
11. Pivot tables
12. Time series (resample, rolling windows)
13. String operations
14. Saving data (CSV, Excel)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# ── Settings ──────────────────────────────────────────────────

# These are stocks I track — mix of sectors for comparison
MY_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
PERIOD = "1y"                  # 1 year of daily data
RISK_FREE_RATE = 0.026         # 6.5% FD rate / 252 trading days
ROLLING_DAYS = 20              # 20-day moving average window


# ── Step 1: Download data (Topic 4) ──────────────────────────

def download_stock_data(tickers, period):
    """
    Download closing prices for all stocks from Yahoo Finance.
    Returns a DataFrame where each column is one stock.
    """
    print(f"\nDownloading {len(tickers)} stocks for {period}...")

    raw = yf.download(tickers, period=period, progress=False)

    # yfinance gives multi-level columns like (Close, RELIANCE.NS)
    # We only need the Close prices
    prices = raw["Close"].copy()

    # Topic 13 — clean up column names: remove .NS suffix
    # "RELIANCE.NS" becomes "RELIANCE" — looks cleaner in reports
    prices.columns = [col.replace(".NS", "") for col in prices.columns]

    print(f"  Got {len(prices)} trading days for {list(prices.columns)}")
    return prices


# ── Step 2: Inspect data (Topic 5) ───────────────────────────

def inspect_data(prices):
    """
    Check what the data looks like before doing anything with it.
    This is like looking at raw material before building something.
    """
    print("\n── DATA INSPECTION ──")
    print(f"  Shape       : {prices.shape[0]} rows × {prices.shape[1]} columns")
    print(f"  Date range  : {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Missing vals: {prices.isnull().sum().sum()}")

    # Topic 5 — describe() gives quick statistical summary
    print("\n  Quick stats (₹):")
    print(prices.describe().round(2).to_string())


# ── Step 3: Clean data (Topic 7) ─────────────────────────────

def clean_data(prices):
    """
    Fix problems in the raw data before analysis.

    Why ffill instead of filling with 0?
    If RELIANCE closed at ₹2,900 yesterday and has no data today,
    the best guess is still ₹2,900 — not ₹0. Filling with 0 would
    create a fake -100% crash that ruins every calculation after it.
    """
    missing_before = prices.isnull().sum().sum()

    # Topic 7 — forward fill: use previous day's price for gaps
    prices = prices.ffill()

    # Topic 7 — backward fill: handles gaps at the very start
    prices = prices.bfill()

    # Topic 7 — remove duplicate rows (same date twice)
    prices = prices.drop_duplicates()

    missing_after = prices.isnull().sum().sum()
    print(f"\n── CLEANING ──")
    print(f"  Fixed {missing_before - missing_after} missing values")
    print(f"  Remaining gaps: {missing_after}")

    return prices


# ── Step 4: Calculate returns (Topic 8) ──────────────────────

def calculate_returns(prices):
    """
    Daily return = how much the stock moved today vs yesterday.
    Formula: (today - yesterday) / yesterday × 100

    pct_change() does this calculation for every row at once —
    that's the power of vectorized operations (Topic 1).
    """
    # Topic 8 — pct_change gives decimal return, multiply by 100 for %
    daily_returns = prices.pct_change() * 100

    # First row will be NaN (no "yesterday" to compare with)
    daily_returns = daily_returns.dropna()

    print(f"\n── RETURNS ──")
    print(f"  Calculated daily returns for {len(daily_returns)} days")
    return daily_returns


# ── Step 5: Per-stock statistics (Topics 1, 2) ───────────────

def stock_statistics(returns):
    """
    Calculate key stats for each stock — these are the same
    formulas I learned by hand in Module 2 (Math Foundation).

    Mean    = average daily return
    Std Dev = how wildly the stock swings (risk measure)
    Sharpe  = return per unit of risk (higher = better)
    """
    print("\n══════════════════════════════════════════════════════")
    print("  SECTION 1: PER-STOCK STATISTICS")
    print("══════════════════════════════════════════════════════")

    stats_rows = []

    for stock in returns.columns:
        r = returns[stock].dropna()

        # Topic 1 — NumPy operations on arrays
        avg = np.mean(r)
        std = np.std(r)

        # Sharpe Ratio from Module 2: (mean - risk_free) / std × √252
        sharpe = 0
        if std > 0:
            sharpe = (avg - RISK_FREE_RATE) / std * np.sqrt(252)

        stats_rows.append({
            "Stock": stock,
            "Avg Return (%)": round(avg, 3),
            "Std Dev (%)": round(std, 3),
            "Sharpe Ratio": round(sharpe, 2),
            "Min Day (%)": round(r.min(), 2),
            "Max Day (%)": round(r.max(), 2),
        })

    # Topic 3 — build DataFrame from list of dictionaries
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    return stats_df


# ── Step 6: Correlation matrix (Topic 10) ────────────────────

def correlation_analysis(returns):
    """
    How do stocks move relative to each other?

    +1.0 = move together (TCS and INFY — both IT sector)
     0.0 = no relationship
    -1.0 = move opposite (rare in real markets)

    As a CMT holder I know: low correlation = better diversification.
    If all your stocks are +0.95 correlated, you basically own one stock.
    """
    print("\n══════════════════════════════════════════════════════")
    print("  SECTION 2: CORRELATION MATRIX")
    print("══════════════════════════════════════════════════════")

    # Topic 10 — .corr() computes pairwise Pearson correlation
    corr = returns.corr().round(2)
    print(corr.to_string())

    # Find the most and least correlated pairs
    # (excluding self-correlation which is always 1.0)
    pairs = []
    stocks = corr.columns.tolist()
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            pairs.append({
                "Pair": f"{stocks[i]} — {stocks[j]}",
                "Correlation": corr.iloc[i, j]
            })

    pairs_df = pd.DataFrame(pairs).sort_values("Correlation")
    print(f"\n  Most diversified pair : {pairs_df.iloc[0]['Pair']} "
          f"({pairs_df.iloc[0]['Correlation']})")
    print(f"  Most correlated pair  : {pairs_df.iloc[-1]['Pair']} "
          f"({pairs_df.iloc[-1]['Correlation']})")

    return corr


# ── Step 7: Extreme days finder (Topic 6) ────────────────────

def find_extreme_days(returns, top_n=5):
    """
    Find the biggest up and down days across the portfolio.

    In CMT terms: these are the days with highest absolute momentum.
    Understanding WHY these days happened (news, events) is what
    separates a quant from someone who just runs code.
    """
    print("\n══════════════════════════════════════════════════════")
    print(f"  SECTION 3: TOP {top_n} MOST VOLATILE DAYS")
    print("══════════════════════════════════════════════════════")

    # Topic 1 — mean across columns = portfolio average return
    portfolio_ret = returns.mean(axis=1)

    # Topic 6 — select using nlargest on absolute values
    biggest_moves = portfolio_ret.abs().nlargest(top_n)
    extreme_dates = biggest_moves.index

    # Topic 6 — loc selection to get all stock returns on those dates
    result = returns.loc[extreme_dates].copy()
    result.insert(0, "Portfolio Avg (%)", portfolio_ret.loc[extreme_dates])
    result = result.round(2)

    print(result.to_string())
    return result


# ── Step 8: Monthly return patterns (Topics 9, 11) ──────────

def monthly_analysis(returns):
    """
    Which months are historically strong or weak?

    GroupBy (Topic 9) splits data by month, calculates mean,
    then Pivot (Topic 11) reshapes it into a grid.

    For trading: if January is consistently positive, that's
    a seasonal pattern worth investigating further.
    """
    print("\n══════════════════════════════════════════════════════")
    print("  SECTION 4: MONTHLY AVERAGE RETURNS (%)")
    print("══════════════════════════════════════════════════════")

    r = returns.copy()
    r["Month"] = r.index.strftime("%B")

    # Topic 11 — melt from wide to tall format
    stock_cols = [c for c in r.columns if c != "Month"]
    tall = r.melt(
        id_vars="Month",
        value_vars=stock_cols,
        var_name="Stock",
        value_name="Return"
    )

    # Topic 11 — pivot into month × stock grid
    pivot = pd.pivot_table(
        tall,
        values="Return",
        index="Month",
        columns="Stock",
        aggfunc="mean"
    ).round(3)

    print(pivot.to_string())
    return pivot


# ── Step 9: Day-of-week volatility (Topic 9) ────────────────

def weekday_analysis(returns):
    """
    Are Mondays really more volatile than Fridays?

    This is a classic market microstructure question. Using GroupBy
    to calculate standard deviation per weekday tells us which days
    have the wildest swings — useful for timing entries and exits.
    """
    print("\n══════════════════════════════════════════════════════")
    print("  SECTION 5: VOLATILITY BY DAY OF WEEK (%)")
    print("══════════════════════════════════════════════════════")

    r = returns.copy()
    r["Weekday"] = r.index.strftime("%A")

    # Topic 9 — GroupBy weekday, apply std()
    # This is Split → Apply → Combine in action
    weekday_vol = {}
    stock_cols = [c for c in r.columns if c != "Weekday"]
    for stock in stock_cols:
        weekday_vol[stock] = r.groupby("Weekday")[stock].std()

    weekday_df = pd.DataFrame(weekday_vol).round(3)

    # Reorder days properly (Monday → Friday)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday_df = weekday_df.reindex(day_order)

    print(weekday_df.to_string())
    return weekday_df


# ── Step 10: Rolling indicators (Topic 12) ──────────────────

def rolling_analysis(prices, returns):
    """
    20-day rolling mean and rolling std dev.

    Rolling mean = moving average (the most basic CMT indicator).
    Rolling std = how volatile the stock has been recently.

    When current price > rolling mean → uptrend signal.
    When rolling std spikes → market uncertainty is increasing.
    """
    print("\n══════════════════════════════════════════════════════")
    print(f"  SECTION 6: {ROLLING_DAYS}-DAY ROLLING INDICATORS (latest)")
    print("══════════════════════════════════════════════════════")

    # Topic 12 — rolling window calculations
    rolling_mean = prices.rolling(window=ROLLING_DAYS).mean()
    rolling_vol = returns.rolling(window=ROLLING_DAYS).std()

    # Show the most recent values
    latest_price = prices.iloc[-1]
    latest_ma = rolling_mean.iloc[-1]
    latest_vol = rolling_vol.iloc[-1]

    summary_rows = []
    for stock in prices.columns:
        price = latest_price[stock]
        ma = latest_ma[stock]
        vol = latest_vol[stock]

        # Simple trend check: price above or below 20-day MA
        trend = "ABOVE MA" if price > ma else "BELOW MA"

        summary_rows.append({
            "Stock": stock,
            "Price (₹)": round(price, 2),
            "20-Day MA (₹)": round(ma, 2),
            "Trend": trend,
            "20-Day Vol (%)": round(vol, 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    return rolling_mean, rolling_vol


# ── Step 11: Weekly resample (Topic 12) ──────────────────────

def weekly_summary(prices, returns):
    """
    Compress daily data to weekly — reduces noise, shows trend.

    resample('W') groups all days in each week, then:
      - .last() picks the Friday closing price
      - .mean() averages all daily returns that week
    """
    # Topic 12 — resample to weekly frequency
    weekly_prices = prices.resample("W").last()
    weekly_returns = returns.resample("W").mean().round(3)

    print(f"\n  Weekly data: {len(weekly_prices)} weeks from daily data")
    return weekly_prices, weekly_returns


# ── Step 12: Save everything (Topic 14) ──────────────────────

def save_reports(prices, returns, stats, corr, pivot):
    """
    Export to CSV and Excel — the formats clients actually want.

    CSV  = universal, works everywhere
    Excel = what finance people expect to receive
    """
    print("\n══════════════════════════════════════════════════════")
    print("  SAVING REPORTS")
    print("══════════════════════════════════════════════════════")

    today = datetime.now().strftime("%Y%m%d")

    # Build a combined sheet with prices + returns side by side
    combined = prices.copy()
    for stock in returns.columns:
        combined[stock + "_return%"] = returns[stock]

    # Topic 14 — save to CSV
    csv_file = f"nse_report_{today}.csv"
    combined.to_csv(csv_file)
    print(f"  Saved: {csv_file}")

    # Topic 14 — save pivot table separately
    pivot_file = f"nse_monthly_pivot_{today}.csv"
    pivot.to_csv(pivot_file)
    print(f"  Saved: {pivot_file}")

    # Topic 14 — save to Excel with multiple sheets
    excel_file = f"nse_report_{today}.xlsx"
    try:
        with pd.ExcelWriter(excel_file) as writer:
            combined.to_excel(writer, sheet_name="Prices & Returns")
            stats.to_excel(writer, sheet_name="Statistics", index=False)
            corr.to_excel(writer, sheet_name="Correlations")
            pivot.to_excel(writer, sheet_name="Monthly Pivot")
        print(f"  Saved: {excel_file} (4 sheets)")
    except ImportError:
        print("  Excel skipped — install openpyxl: pip install openpyxl")

    return csv_file, excel_file


# ── Main: Run everything in order ────────────────────────────

def main():
    print("=" * 55)
    print("  NSE DATA TOOLKIT")
    print("  Author: Shivam Tyagi | github.com/QuantShivam")
    print("=" * 55)

    # 1. Download
    prices = download_stock_data(MY_STOCKS, PERIOD)

    # 2. Inspect
    inspect_data(prices)

    # 3. Clean
    prices = clean_data(prices)

    # 4. Returns
    returns = calculate_returns(prices)

    # 5. Statistics
    stats = stock_statistics(returns)

    # 6. Correlations
    corr = correlation_analysis(returns)

    # 7. Extreme days
    find_extreme_days(returns)

    # 8. Monthly patterns
    pivot = monthly_analysis(returns)

    # 9. Weekday patterns
    weekday_analysis(returns)

    # 10. Rolling indicators
    rolling_analysis(prices.loc[returns.index], returns)

    # 11. Weekly summary
    weekly_summary(prices.loc[returns.index], returns)

    # 12. Save
    save_reports(prices.loc[returns.index], returns, stats, corr, pivot)

    print("\n" + "=" * 55)
    print("  REPORT COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()