import pandas as pd 
# install gdown if not already installed also for later requirements install xgboost if not already installed
import gdown
file_id = "1QNJ3hfm7zLfJ21lizliWbqaDB8VqgaVb"  # replace with your actual file ID
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "panel_data_final.xlsx", quiet=False)
# https://docs.google.com/spreadsheets/d/1QNJ3hfm7zLfJ21lizliWbqaDB8VqgaVb/edit?usp=sharing&ouid=100032466353511784065&rtpof=true&sd=true
df = pd.read_excel('panel_data_final.xlsx', sheet_name='Panel Data')

print(df.shape)  
print(df.head())

print("="*80)
print("BASIC INFORMATION")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")



total_cells = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()
total_missing_pct = (total_missing / total_cells) * 100

print(f"Total cells: {total_cells:,}")
print(f"Total missing values: {total_missing:,}")
print(f"Overall missing percentage: {total_missing_pct:.2f}%")



missing_by_col = df.isnull().sum()
missing_pct_by_col = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'Column': missing_by_col.index,
    'Missing_Count': missing_by_col.values,
    'Missing_Percentage': missing_pct_by_col.values
}).sort_values('Missing_Percentage', ascending=False)

print(missing_summary[missing_summary['Missing_Count'] > 0].to_string(index=False))



print(f"Missing in 'Year': {df['Year'].isnull().sum()}")
print(f"Missing in 'Company': {df['Company'].isnull().sum()}")
print(f"Missing in 'Sector': {df['Sector'].isnull().sum()}")


missing_by_company = df.groupby('Company').apply(
    lambda x: x.isnull().sum().sum()
).sort_values(ascending=False)

print(missing_by_company.to_string())

missing_by_year = df.groupby('Year').apply(
    lambda x: x.isnull().sum().sum()
).sort_values(ascending=False)

print(missing_by_year.to_string())


missing_by_sector = df.groupby('Sector').apply(
    lambda x: x.isnull().sum().sum()
).sort_values(ascending=False)

print(missing_by_sector.to_string())


expected_obs = df['Company'].nunique() * df['Year'].nunique()
actual_obs = len(df)
missing_combinations = expected_obs - actual_obs

print(f"Number of unique companies: {df['Company'].nunique()}")
print(f"Number of unique years: {df['Year'].nunique()}")
print(f"Expected observations (balanced): {expected_obs}")
print(f"Actual observations: {actual_obs}")
print(f"Missing company-year combinations: {missing_combinations}")
print(f"Panel is balanced: {expected_obs == actual_obs}")


financial_cols = [
    'Net Sales (Revenue)', 'Cost of Sales', 'Gross Profit',
    'Operating Income (Loss)', 'Interest Income', 'Interest Expense',
    'Other Income (Expense), net', 'Pre-Tax Income (Loss)',
    'Income Tax Expense (Benefit)', 'Net Income (Loss)', 'EPS — Diluted ($)',
    'Diluted Shares Outstanding (M)', 'Net Cash from Operations (CFO)',
    'Capital Expenditures (CapEx)', 'Acquisitions', 'Net Cash from Investing (CFI)',
    'Proceeds from Debt Issuance', 'Debt Repayment', 'Share Buybacks',
    'Dividends Paid', 'Net Cash from Financing (CFF)', 'Net Change in Cash',
    'Depreciation & Amortization', 'Stock-Based Compensation', 'Changes in Working Capital',
    'Cash & Cash Equivalents', 'Accounts Receivable, net', 'Inventories',
    'Other Current Assets', 'Total Current Assets', 'PP&E, net', 'Goodwill',
    'Other Long-Term Assets', 'Total Assets', 'Accounts Payable', 'Accrued Liabilities',
    'Other Current Liabilities', 'Total Current Liabilities', 'Long-Term Debt',
    'Other Long-Term Liabilities', 'Total Liabilities', "Total Stockholders' Equity",
    'Operating Expenses (SG&A + R&D)', 'Marketable Securities', 'Other Investing Activities',
    'Other Financing Activities', 'Stock Price (USD)', 'Sector Benchmark (%)',
    'US Inflation Rate (%)', 'US Federal Funds Rate (%)', 'US GDP Growth Rate (%)',
    'S&P 500 Annual Return (%)'
]

print("="*80)
print("FINANCIAL METRICS - MISSING VALUES ANALYSIS")
print("="*80)
print(f"{'Column Name':<40} | {'Missing':<8} | {'Percentage'}")
print("-"*80)

for col in financial_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"{col:<40} | {missing_count:>6}    | {missing_pct:>6.2f}%")
    else:
        print(f"{col:<40} | {'N/A':>6}    | {'N/A':>6}")

print("="*80)
total_missing_fin = sum([df[col].isnull().sum() for col in financial_cols if col in df.columns])
total_cells_fin = len([col for col in financial_cols if col in df.columns]) * len(df)
print(f"TOTAL missing across financial columns: {total_missing_fin} out of {total_cells_fin} "
      f"({total_missing_fin/total_cells_fin*100:.2f}%)")


missing_cols = ['Net Income (Loss)', 'Total Assets', 'Stock Price (USD)']
missing_df = df[missing_cols].isnull().astype(int)

if len(missing_df.columns) > 1:
    missing_corr = missing_df.corr()
    print("Correlation of missing values between key variables:")
    print(missing_corr.to_string())


cols_to_drop = [
    'Other Financing Activities',  # 80% missing - captured by CFF components
    'Other Investing Activities'   # 76% missing - captured by CFI components
]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
print(f"Dropped: {cols_to_drop}")
print("   Reason: Already captured by other cash flow components\n")


import numpy as np

moderate_missing_cols = {
    'Marketable Securities': 60.0,
    'Interest Income': 52.0
}

print("Handling moderate missing columns (50-70%):")
for col, pct in moderate_missing_cols.items():
    if col in df.columns:
        df[col] = df.groupby('Company')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        if df[col].isnull().any():
            sector_year_median = df.groupby(['Sector', 'Year'])[col].transform('median')
            df[col] = df[col].fillna(sector_year_median)
            if col == 'Marketable Securities' and df[col].isnull().any():
                df[col] = df[col].fillna(df['Cash & Cash Equivalents'] * 0.15)
            elif col == 'Interest Income' and df[col].isnull().any():
                df[col] = df[col].fillna(df['Cash & Cash Equivalents'] * 0.025)
        print(f"   Handled {col} ({pct}% missing) - kept for analysis")

print("\nInterpolating remaining columns by company...")

remaining_cols = [
    'Other Income (Expense), net', 'Operating Expenses (SG&A + R&D)',
    'Acquisitions', 'Debt Repayment', 'Stock-Based Compensation',
    'Other Current Assets', 'Other Long-Term Assets', 'Accrued Liabilities',
    'Other Long-Term Liabilities', 'EPS — Diluted ($)',
    'Diluted Shares Outstanding (M)', 'Proceeds from Debt Issuance',
    'Share Buybacks', 'Dividends Paid', 'Net Change in Cash',
    'Changes in Working Capital', 'Accounts Payable', 'Other Current Liabilities',
    'Sector Benchmark (%)', 'Goodwill', 'Stock Price (USD)'
]

for col in remaining_cols:
    if col in df.columns:
        df[col] = df.groupby('Company')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
print(f"   Interpolated {len([c for c in remaining_cols if c in df.columns])} columns")

print("\nFinal cleanup for any remaining missing values...")

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df.groupby('Company')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())


print("\n" + "="*80)
print("FINAL DATA QUALITY CHECK")
print("="*80)
remaining = df.isnull().sum().sum()
print(f"Total missing values remaining: {remaining}")
if remaining == 0:
    print("Data is clean and ready for the 7-layer framework!")

print("\n" + "="*80)
print("FEATURE AVAILABILITY BY LAYER")
print("="*80)

layers = {
    'Layer 1 - Feature Intelligence': ['Net Sales (Revenue)', 'Gross Profit',
                                        'Operating Income (Loss)', 'Total Assets',
                                        'Total Liabilities', 'Stock Price (USD)'],
    'Layer 2 - Macro Transmission':   ['Interest Expense', 'Long-Term Debt',
                                        'US Inflation Rate (%)', 'US Federal Funds Rate (%)',
                                        'US GDP Growth Rate (%)'],
    'Layer 3 - Regime Detection':      ['Sector Benchmark (%)', 'S&P 500 Annual Return (%)',
                                        'US GDP Growth Rate (%)', 'US Inflation Rate (%)'],
    'Layer 4 - Core Prediction':       ['Net Income (Loss)', 'EPS — Diluted ($)',
                                        'Operating Income (Loss)'],
    'Layer 5 - Scenario Simulation':   ['Net Cash from Operations (CFO)',
                                        'Capital Expenditures (CapEx)'],
    'Layer 6 - Sensitivity':           ['Stock Price (USD)', 'Net Income (Loss)', 'Total Assets'],
    'Layer 7 - Risk Dashboard':        ['Total Liabilities', 'Long-Term Debt', 'Net Income (Loss)']
}

for layer, features in layers.items():
    available = [f for f in features if f in df.columns]
    missing_feat = [f for f in features if f not in df.columns]
    print(f"\n{layer}:")
    print(f"   Available: {len(available)}/{len(features)} features")
    if missing_feat:
        print(f"   Missing: {missing_feat}")


df.to_excel('panel_data_cleaned_for_stress_testing.xlsx', index=False)
print("\n" + "="*80)
print("CLEANED DATA SAVED: panel_data_cleaned_for_stress_testing.xlsx")
print("="*80)


import pandas as pd
import numpy as np

df = pd.read_excel('panel_data_cleaned_for_stress_testing.xlsx')
df = df.sort_values(['Company', 'Year'])

print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Missing values: {df.isnull().sum().sum()}")

print("="*80)
print("FINANCIAL METRICS - POST-CLEANING VERIFICATION")
print("="*80)
print(f"{'Column Name':<40} | {'Missing':<8} | {'Percentage'}")
print("-"*80)

for col in financial_cols:   
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"{col:<40} | {missing_count:>6}    | {missing_pct:>6.2f}%")
    else:
        print(f"{col:<40} | {'N/A':>6}    | {'N/A':>6}")

print("="*80)
total_missing_check = sum([df[col].isnull().sum() for col in financial_cols if col in df.columns])
total_cells_check = len([col for col in financial_cols if col in df.columns]) * len(df)
print(f"TOTAL missing: {total_missing_check} out of {total_cells_check} "
      f"({total_missing_check/total_cells_check*100:.2f}%)")

df_interpolated = df.copy()


TRAIN_END_YEAR   = 2019   # train: 2000–2019
VAL_START_YEAR   = 2020   # validation: 2020–2021
TEST_START_YEAR  = 2022   # test: 2022–2024

all_years = sorted(df['Year'].unique())
train_years = [y for y in all_years if y <= TRAIN_END_YEAR]
val_years   = [y for y in all_years if VAL_START_YEAR <= y < TEST_START_YEAR]
test_years  = [y for y in all_years if y >= TEST_START_YEAR]

print(f"Train years  : {train_years[0]}–{train_years[-1]} ({len(train_years)} years)")
print(f"Val years    : {val_years[0]}–{val_years[-1]} ({len(val_years)} years)")
print(f"Test years   : {test_years[0]}–{test_years[-1]} ({len(test_years)} years)")



print("\n" + "="*70)
print("LAYER 1: FEATURE INTELLIGENCE ENGINE")
print("="*70)

df_features = df_interpolated.copy()

print("\nCalculating Profitability Ratios...")

df_features['roa']              = df_features['Net Income (Loss)'] / df_features['Total Assets']
df_features['roe']              = df_features['Net Income (Loss)'] / df_features["Total Stockholders' Equity"]
df_features['profit_margin']    = df_features['Net Income (Loss)'] / df_features['Net Sales (Revenue)']
df_features['gross_margin']     = df_features['Gross Profit']       / df_features['Net Sales (Revenue)']
df_features['operating_margin'] = df_features['Operating Income (Loss)'] / df_features['Net Sales (Revenue)']

print(f"   ROA range: {df_features['roa'].min():.4f} to {df_features['roa'].max():.4f}")
print(f"   ROE range: {df_features['roe'].min():.4f} to {df_features['roe'].max():.4f}")
print(f"   Profit Margin range: {df_features['profit_margin'].min():.4f} to {df_features['profit_margin'].max():.4f}")


print("\nCalculating Leverage Ratios...")

df_features['debt_ratio']      = df_features['Total Liabilities'] / df_features['Total Assets']
df_features['debt_to_equity']  = df_features['Total Liabilities'] / df_features["Total Stockholders' Equity"]
df_features['equity_ratio']    = df_features["Total Stockholders' Equity"] / df_features['Total Assets']

print(f"   Debt Ratio range: {df_features['debt_ratio'].min():.4f} to {df_features['debt_ratio'].max():.4f}")
print(f"   Debt-to-Equity range: {df_features['debt_to_equity'].min():.4f} to {df_features['debt_to_equity'].max():.4f}")

print("\nCalculating Liquidity Ratios...")

df_features['current_ratio'] = df_features['Total Current Assets']  / df_features['Total Current Liabilities']
df_features['quick_ratio']   = (df_features['Total Current Assets'] - df_features['Inventories']) / df_features['Total Current Liabilities']

print(f"   Current Ratio range: {df_features['current_ratio'].min():.2f} to {df_features['current_ratio'].max():.2f}")
print(f"   Quick Ratio range: {df_features['quick_ratio'].min():.2f} to {df_features['quick_ratio'].max():.2f}")


print("\nCalculating Efficiency Ratios...")

df_features['asset_turnover'] = df_features['Net Sales (Revenue)'] / df_features['Total Assets']
print(f"   Asset Turnover range: {df_features['asset_turnover'].min():.2f} to {df_features['asset_turnover'].max():.2f}")

print("\nCalculating Cash Flow Ratios...")

df_features['cfo_to_assets'] = df_features['Net Cash from Operations (CFO)'] / df_features['Total Assets']
df_features['cfo_to_sales']  = df_features['Net Cash from Operations (CFO)'] / df_features['Net Sales (Revenue)']
df_features['capex_to_sales']= df_features['Capital Expenditures (CapEx)'].abs() / df_features['Net Sales (Revenue)']

print(f"   CFO to Assets range: {df_features['cfo_to_assets'].min():.4f} to {df_features['cfo_to_assets'].max():.4f}")
print(f"   CFO to Sales range: {df_features['cfo_to_sales'].min():.4f} to {df_features['cfo_to_sales'].max():.4f}")


print("\nFirm Size - SCALE MEASURE")
df_features['firm_size'] = np.log(df_features['Total Assets'].clip(lower=1))
print(f"   Range: {df_features['firm_size'].min():.2f} to {df_features['firm_size'].max():.2f}")
print(f"   Mean: {df_features['firm_size'].mean():.2f}")


print("\nStock Returns - MARKET MEASURES")

if 'Stock Price (USD)' in df_features.columns:
    df_features['stock_return_company'] = df_features.groupby('Company')['Stock Price (USD)'].pct_change() * 100
    print(f"   Company stock return range: {df_features['stock_return_company'].min():.2f}% to {df_features['stock_return_company'].max():.2f}%")

if 'Sector Benchmark (%)' in df_features.columns:
    df_features['stock_return_sector'] = df_features['Sector Benchmark (%)']
    print(f"   Sector return range: {df_features['stock_return_sector'].min():.2f}% to {df_features['stock_return_sector'].max():.2f}%")


print("\nGDP Growth Rate - MACRO VARIABLE")
if 'US GDP Growth Rate (%)' in df_features.columns:
    df_features['gdp_growth'] = df_features['US GDP Growth Rate (%)']
    print(f"   Range: {df_features['gdp_growth'].min():.2f}% to {df_features['gdp_growth'].max():.2f}%")


df_features = df_features.replace([np.inf, -np.inf], np.nan)

ratio_cols_to_impute = [
    'roa', 'roe', 'profit_margin', 'gross_margin', 'operating_margin',
    'debt_ratio', 'debt_to_equity', 'equity_ratio',
    'current_ratio', 'quick_ratio', 'asset_turnover',
    'cfo_to_assets', 'cfo_to_sales', 'capex_to_sales', 'firm_size'
]

print("\nImputing ratio NaN values (company median -> sector-year median -> global median)...")
for col in ratio_cols_to_impute:
    if col in df_features.columns and df_features[col].isnull().any():
        
        df_features[col] = df_features.groupby('Company')[col].transform(
            lambda x: x.fillna(x.median())
        )
       
        if df_features[col].isnull().any():
            sy_median = df_features.groupby(['Sector', 'Year'])[col].transform('median')
            df_features[col] = df_features[col].fillna(sy_median)
       
        if df_features[col].isnull().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())
print("   Done.")

print("\n" + "="*70)
print("STEP 3: LAG TRANSFORMATIONS")
print("="*70)

lag_variables = ['roa', 'roe', 'profit_margin', 'debt_ratio', 'current_ratio',
                 'asset_turnover', 'firm_size']

for var in lag_variables:
    if var in df_features.columns:
        df_features[f'{var}_lag1'] = df_features.groupby('Company')[var].shift(1)
        df_features[f'{var}_lag2'] = df_features.groupby('Company')[var].shift(2)
        print(f"   Created {var}_lag1 and {var}_lag2")

for var in lag_variables:
    for sfx in ['_lag1', '_lag2']:
        col = f'{var}{sfx}'
        if col in df_features.columns:
            df_features[col] = df_features[col].fillna(0)

print("\n" + "="*70)
print("STEP 4: ROLLING AVERAGES")
print("="*70)

rolling_variables = ['roa', 'roe', 'profit_margin', 'debt_ratio', 'asset_turnover']

for var in rolling_variables:
    if var in df_features.columns:
        df_features[f'{var}_rolling3'] = df_features.groupby('Company')[var].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        print(f"   Created {var}_rolling3 (3-year average)")


print("\n" + "="*70)
print("STEP 5: TREND FEATURES")
print("="*70)

for var in rolling_variables:
    if var in df_features.columns and f'{var}_lag1' in df_features.columns:
        df_features[f'{var}_trend'] = (
            (df_features[var] - df_features[f'{var}_lag1'])
            / (df_features[f'{var}_lag1'].abs() + 1e-6)
        )
        print(f"   Created {var}_trend")


print("\n" + "="*70)
print("STEP 6: MACRO-FIRM INTERACTION TERMS")
print("="*70)

if 'US Inflation Rate (%)' in df_features.columns and 'debt_ratio' in df_features.columns:
    df_features['inflation_debt_interaction'] = df_features['US Inflation Rate (%)'] * df_features['debt_ratio']
    print("   Created inflation x debt_ratio")

if 'US Federal Funds Rate (%)' in df_features.columns and 'debt_ratio' in df_features.columns:
    df_features['rate_debt_interaction'] = df_features['US Federal Funds Rate (%)'] * df_features['debt_ratio']
    print("   Created interest_rate x debt_ratio")

if 'gdp_growth' in df_features.columns and 'asset_turnover' in df_features.columns:
    df_features['gdp_efficiency_interaction'] = df_features['gdp_growth'] * df_features['asset_turnover']
    print("   Created GDP_growth x asset_turnover")

if 'stock_return_company' in df_features.columns and 'roa' in df_features.columns:
    df_features['market_fundamental_interaction'] = df_features['stock_return_company'] * df_features['roa']
    print("   Created stock_return x roa")

if 'US Federal Funds Rate (%)' in df_features.columns and 'current_ratio' in df_features.columns:
    df_features['rate_liquidity_interaction'] = (
        df_features['US Federal Funds Rate (%)'] * (1 / df_features['current_ratio'].clip(lower=0.1))
    )
    print("   Created rate x (1/current_ratio)")

if 'US Inflation Rate (%)' in df_features.columns and 'gross_margin' in df_features.columns:
    df_features['inflation_margin_interaction'] = (
        df_features['US Inflation Rate (%)'] * (1 / df_features['gross_margin'].clip(lower=0.01))
    )
    print("   Created inflation x (1/gross_margin)")

if 'gdp_growth' in df_features.columns and 'debt_to_equity' in df_features.columns:
    df_features['gdp_leverage_interaction'] = df_features['gdp_growth'] * df_features['debt_to_equity']
    print("   Created GDP_growth x debt_to_equity")


df_features['roa_volatility'] = df_features.groupby('Company')['roa'].transform(
    lambda x: x.rolling(5, min_periods=3).std()
)

if 'roa_volatility' in df_features.columns and 'debt_ratio' in df_features.columns:
    df_features['volatility_debt_interaction'] = df_features['roa_volatility'] * df_features['debt_ratio']
    print("   Created volatility x debt_ratio")

if 'firm_size' in df_features.columns and 'gdp_growth' in df_features.columns:
    df_features['size_gdp_interaction'] = df_features['firm_size'] * df_features['gdp_growth']
    print("   Created firm_size x GDP_growth")

if 'profit_margin' in df_features.columns and 'US Federal Funds Rate (%)' in df_features.columns:
    df_features['margin_rate_interaction'] = df_features['profit_margin'] * df_features['US Federal Funds Rate (%)']
    print("   Created profit_margin x interest_rate")

if 'asset_turnover' in df_features.columns and 'US Inflation Rate (%)' in df_features.columns:
    df_features['turnover_inflation_interaction'] = df_features['asset_turnover'] * df_features['US Inflation Rate (%)']
    print("   Created asset_turnover x inflation")

# Cash ratio
if 'Cash & Cash Equivalents' in df_features.columns and 'Total Current Liabilities' in df_features.columns:
    df_features['cash_ratio'] = df_features['Cash & Cash Equivalents'] / df_features['Total Current Liabilities']
    if 'US Federal Funds Rate (%)' in df_features.columns:
        df_features['cash_rate_interaction'] = df_features['cash_ratio'] * df_features['US Federal Funds Rate (%)']
        print("   Created cash_ratio x interest_rate")

if 'firm_size' in df_features.columns and 'roa_volatility' in df_features.columns:
    df_features['size_volatility_interaction'] = df_features['firm_size'] * df_features['roa_volatility']
    print("   Created firm_size x volatility")

if 'quick_ratio' in df_features.columns and 'gdp_growth' in df_features.columns:
    df_features['liquidity_gdp_interaction'] = df_features['quick_ratio'] * df_features['gdp_growth']
    print("   Created quick_ratio x GDP_growth")


print("\nCleaning up invalid values in interaction terms...")
df_features = df_features.replace([np.inf, -np.inf], np.nan)

interaction_cols = [c for c in df_features.columns if 'interaction' in c or 'volatility' in c]
for col in interaction_cols:
    if df_features[col].isnull().any():
        df_features[col] = df_features.groupby('Company')[col].transform(
            lambda x: x.fillna(x.median())
        )
        df_features[col] = df_features[col].fillna(df_features[col].median())
        df_features[col] = df_features[col].fillna(0)  # absolute last resort
print("   Done.")


all_new_features = [
    'roa', 'roe', 'profit_margin', 'gross_margin', 'operating_margin',
    'debt_ratio', 'debt_to_equity', 'equity_ratio', 'current_ratio', 'quick_ratio',
    'asset_turnover', 'cfo_to_assets', 'cfo_to_sales', 'capex_to_sales',
    'firm_size', 'stock_return_company', 'stock_return_sector', 'gdp_growth',
    'roa_volatility', 'cash_ratio'
] + [f'{v}_lag1' for v in lag_variables] + [f'{v}_lag2' for v in lag_variables] \
  + [f'{v}_rolling3' for v in rolling_variables] \
  + [f'{v}_trend' for v in rolling_variables] \
  + interaction_cols

existing_features = [col for col in all_new_features if col in df_features.columns]

print("\n" + "="*70)
print("LAYER 1 COMPLETE - FINAL SUMMARY")
print("="*70)
print(f"\nTotal rows: {len(df_features):,}")
print(f"Total columns: {len(df_features.columns)}")
print(f"New features created: {len(existing_features)}")
print(f"\nNew Features by Category:")
print(f"   Profitability: 5  |  Leverage: 3  |  Liquidity: 2  |  Efficiency: 1")
print(f"   Cash Flow: 3  |  Firm Size: 1  |  Market: 2  |  Macro: 1  |  Volatility: 2")
print(f"   Lags: {len([c for c in existing_features if '_lag' in c])}  "
      f"|  Rolling: {len([c for c in existing_features if '_rolling' in c])}  "
      f"|  Trends: {len([c for c in existing_features if '_trend' in c])}  "
      f"|  Interactions: {len([c for c in existing_features if 'interaction' in c])}")

print("\nSample data (first 5 rows of key features):")
display_cols = ['Company', 'Year', 'roa', 'debt_ratio', 'current_ratio',
                'asset_turnover', 'firm_size', 'roa_lag1', 'roa_rolling3', 'roa_trend']
available_cols = [col for col in display_cols if col in df_features.columns]
print(df_features[available_cols].head(5).to_string(index=False))
print("\nLayer 1 COMPLETE! Ready for Layer 2 (Macro Transmission Layer)")


new_features_display = [
    'roa', 'roe', 'profit_margin', 'gross_margin', 'operating_margin',
    'debt_ratio', 'debt_to_equity', 'equity_ratio', 'current_ratio',
    'quick_ratio', 'asset_turnover', 'cfo_to_assets', 'cfo_to_sales', 'capex_to_sales'
]
existing_display = [col for col in new_features_display if col in df_features.columns]

print("\n" + "="*70)
print("LAYER 1 RESULTS")
print("="*70)
print("\nHEAD (First 5 rows):")
print(df_features[existing_display].head().to_string(index=False))
print("\nSUMMARY STATISTICS:")
stats = df_features[existing_display].describe().round(4)
print(stats)
print(f"\nTotal: {len(existing_display)} features | {len(df_features)} rows")
print("="*70)


df = df_features.copy()


print("\nCreating Regime Detection Features...")
if 'gdp_growth' in df.columns:
    df['recession'] = (df['gdp_growth'] < 0).astype(int)
    print("   Created recession dummy")


print("\nCreating 3-Way Interactions...")
three_way_created = []

if all(x in df.columns for x in ['US Inflation Rate (%)', 'debt_ratio', 'US Federal Funds Rate (%)']):
    df['perfect_storm_risk'] = df['US Inflation Rate (%)'] * df['debt_ratio'] * df['US Federal Funds Rate (%)']
    three_way_created.append('perfect_storm_risk')
    print("   Created perfect_storm_risk")

if all(x in df.columns for x in ['debt_ratio', 'US Federal Funds Rate (%)', 'recession']):
    df['default_risk_triple'] = df['debt_ratio'] * df['US Federal Funds Rate (%)'] * df['recession']
    three_way_created.append('default_risk_triple')
    print("   Created default_risk_triple")

if all(x in df.columns for x in ['US Inflation Rate (%)', 'profit_margin', 'debt_ratio']):
    df['inflation_squeeze'] = (
        df['US Inflation Rate (%)'] * (1 / (df['profit_margin'].abs() + 0.01)) * df['debt_ratio']
    )
    three_way_created.append('inflation_squeeze')
    print("   Created inflation_squeeze")

if all(x in df.columns for x in ['gdp_growth', 'debt_to_equity', 'current_ratio']):
    df['liquidity_crisis'] = (
        (df['gdp_growth'].clip(upper=0).abs()) * df['debt_to_equity']
        * (1 / (df['current_ratio'] + 0.01))
    )
    three_way_created.append('liquidity_crisis')
    print("   Created liquidity_crisis")

if all(x in df.columns for x in ['roa_volatility', 'debt_ratio', 'US Federal Funds Rate (%)']):
    df['rollover_risk'] = df['roa_volatility'] * df['debt_ratio'] * df['US Federal Funds Rate (%)']
    three_way_created.append('rollover_risk')
    print("   Created rollover_risk")

if all(x in df.columns for x in ['US Inflation Rate (%)', 'asset_turnover', 'profit_margin']):
    df['pricing_power'] = df['US Inflation Rate (%)'] * df['asset_turnover'] * df['profit_margin']
    three_way_created.append('pricing_power')
    print("   Created pricing_power")

if all(x in df.columns for x in ['stock_return_company', 'debt_ratio', 'recession']):
    df['distress_signal'] = (
        (df['stock_return_company'].clip(upper=0).abs()) * df['debt_ratio'] * df['recession']
    )
    three_way_created.append('distress_signal')
    print("   Created distress_signal")

print(f"\n3-WAY INTERACTIONS CREATED: {len(three_way_created)}/7")
for f in three_way_created:
    non_zero = (df[f] != 0).sum()
    print(f"   {f} ({non_zero} non-zero values)")


if 'US Inflation Rate (%)' in df.columns:
    df['regime_deflation']          = (df['US Inflation Rate (%)'] < 0).astype(int)
    df['regime_low_inflation']      = ((df['US Inflation Rate (%)'] >= 0) & (df['US Inflation Rate (%)'] <= 2)).astype(int)
    df['regime_moderate_inflation'] = ((df['US Inflation Rate (%)'] > 2) & (df['US Inflation Rate (%)'] <= 5)).astype(int)
    df['regime_high_inflation']     = (df['US Inflation Rate (%)'] > 5).astype(int)
    print("   Created 4 inflation regimes")

if 'US Federal Funds Rate (%)' in df.columns:
    df['regime_emergency_rates'] = ((df['US Federal Funds Rate (%)'] >= 0) & (df['US Federal Funds Rate (%)'] <= 1)).astype(int)
    df['regime_low_rates']       = ((df['US Federal Funds Rate (%)'] > 1) & (df['US Federal Funds Rate (%)'] <= 3)).astype(int)
    df['regime_moderate_rates']  = ((df['US Federal Funds Rate (%)'] > 3) & (df['US Federal Funds Rate (%)'] <= 5)).astype(int)
    df['regime_high_rates']      = (df['US Federal Funds Rate (%)'] > 5).astype(int)
    print("   Created 4 interest rate regimes")

if 'gdp_growth' in df.columns:
    df['regime_severe_recession'] = (df['gdp_growth'] < -2).astype(int)
    df['regime_mild_recession']   = ((df['gdp_growth'] >= -2) & (df['gdp_growth'] < 0)).astype(int)
    df['regime_slow_growth']      = ((df['gdp_growth'] >= 0) & (df['gdp_growth'] < 2)).astype(int)
    df['regime_strong_growth']    = (df['gdp_growth'] >= 2).astype(int)
    print("   Created 4 growth regimes")

if 'US Federal Funds Rate (%)' in df.columns:
    df['fed_rate_change']   = df.groupby('Company')['US Federal Funds Rate (%)'].diff()
    df['regime_easing']     = (df['fed_rate_change'] < -0.25).astype(int)
    df['regime_neutral']    = ((df['fed_rate_change'] >= -0.25) & (df['fed_rate_change'] <= 0.25)).astype(int)
    df['regime_tightening'] = (df['fed_rate_change'] > 0.25).astype(int)
    print("   Created 3 monetary policy regimes")

if 'regime_high_inflation' in df.columns and 'regime_mild_recession' in df.columns:
    df['stagflation'] = (
        (df['regime_high_inflation'] == 1) &
        ((df['regime_mild_recession'] == 1) | (df['regime_severe_recession'] == 1))
    ).astype(int)
    print("   Created stagflation indicator")


if all(c in df.columns for c in ['regime_deflation', 'regime_low_inflation',
                                   'regime_moderate_inflation', 'regime_high_inflation',
                                   'regime_severe_recession', 'regime_mild_recession',
                                   'regime_slow_growth', 'regime_strong_growth']):
    df['regime_code'] = (
        df['regime_deflation'] * 1000 +
        df['regime_low_inflation'] * 100 +
        df['regime_moderate_inflation'] * 10 +
        df['regime_high_inflation'] * 1
    ) * 10 + (
        df['regime_severe_recession'] * 1000 +
        df['regime_mild_recession'] * 100 +
        df['regime_slow_growth'] * 10 +
        df['regime_strong_growth'] * 1
    )
    print("   Created regime_code")

print("\nCreating Stress Intensity Score...")
stress_cols = []
if 'US Inflation Rate (%)' in df.columns:
    df['inflation_stress'] = (df['US Inflation Rate (%)'] / 10).clip(0, 0.25)
    stress_cols.append('inflation_stress')
if 'US Federal Funds Rate (%)' in df.columns:
    df['rate_stress'] = (df['US Federal Funds Rate (%)'] / 8).clip(0, 0.25)
    stress_cols.append('rate_stress')
if 'gdp_growth' in df.columns:
    df['gdp_stress'] = ((-df['gdp_growth']).clip(0, 5) / 20).clip(0, 0.25)
    stress_cols.append('gdp_stress')
if 'recession' in df.columns:
    df['recession_stress'] = df['recession'] * 0.25
    stress_cols.append('recession_stress')

if stress_cols:
    df['stress_intensity'] = df[stress_cols].sum(axis=1) * 100
    print(f"   stress_intensity (0-100): {df['stress_intensity'].min():.1f} to {df['stress_intensity'].max():.1f}")

print("\nCleaning up invalid values...")
df = df.replace([np.inf, -np.inf], np.nan)
numeric_cols_df = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols_df:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)
print("   Cleaned all invalid values")

regime_features = [col for col in df.columns
                   if 'regime_' in col or col in ['stagflation', 'stress_intensity', 'recession']]
layer2_new = three_way_created + regime_features

print(f"\nREGIME FEATURES CREATED: {len(regime_features)}")
print(f"TOTAL LAYER 2 FEATURES ADDED: {len(layer2_new)}")
print(f"\nFINAL DATAFRAME SHAPE: {df.shape}")
print("Layer 2 COMPLETE! Ready for Layer 3 (Regime Detection Engine)")


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("\n" + "="*70)
print("LAYER 3: REGIME DETECTION ENGINE")
print("="*70)

macro_vars = []
macro_candidates = {
    'inflation':    ['US Inflation Rate (%)'],
    'interest_rate':['US Federal Funds Rate (%)'],
    'gdp_growth':   ['gdp_growth', 'US GDP Growth Rate (%)'],
    'stress':       ['stress_intensity'],
}
for name, candidates in macro_candidates.items():
    for col in candidates:
        if col in df.columns:
            macro_vars.append(col)
            print(f"   Found {name}: '{col}'")
            break
    else:
        print(f"   {name} not found")

print(f"\nUsing {len(macro_vars)} macro variables: {macro_vars}")

macro_data = df[['Year'] + macro_vars].drop_duplicates().sort_values('Year').dropna()
print(f"\nYears with complete macro data: {len(macro_data)}")
print(f"   Range: {macro_data['Year'].min()} to {macro_data['Year'].max()}")
print(f"\nMacro data summary:")
print(macro_data[macro_vars].describe().round(2))

fig, axes = plt.subplots(len(macro_vars), 1, figsize=(12, 3 * len(macro_vars)))
if len(macro_vars) == 1:
    axes = [axes]
for i, var in enumerate(macro_vars):
    axes[i].plot(macro_data['Year'], macro_data[var], marker='o', linewidth=2)
    axes[i].axhline(y=macro_data[var].mean(), color='r', linestyle='--',
                    label=f'Mean: {macro_data[var].mean():.2f}')
    axes[i].set_ylabel(var)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Year')
plt.suptitle('Macroeconomic Variables Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('macro_variables_timeseries.png', dpi=150)
print("Saved: macro_variables_timeseries.png")
plt.close()

scaler = StandardScaler()
macro_scaled = scaler.fit_transform(macro_data[macro_vars])
print("\nMacro variables standardized (mean=0, std=1)")

max_clusters = min(6, len(macro_data) - 1)
kmeans_inertia = []
gmm_bic = []

for k in range(1, max_clusters + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(macro_scaled)
    kmeans_inertia.append(km.inertia_)
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(macro_scaled)
    gmm_bic.append(gm.bic(macro_scaled))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, max_clusters + 1), kmeans_inertia, 'bo-')
ax1.set_xlabel('Number of Regimes (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, max_clusters + 1), gmm_bic, 'ro-')
ax2.set_xlabel('Number of Regimes (k)')
ax2.set_ylabel('BIC (lower is better)')
ax2.set_title('BIC for Optimal k')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimal_regimes.png', dpi=150)
print("Saved: optimal_regimes.png")
plt.close()

if len(kmeans_inertia) >= 3:
    diffs = np.diff(kmeans_inertia)
    elbow_point = np.argmin(diffs[:2]) + 2 if len(diffs) >= 2 else 3
    optimal_k = min(elbow_point, 4)
else:
    optimal_k = 3
n_regimes = optimal_k
print(f"\nRecommended number of regimes: {n_regimes}")

print(f"\nApplying K-Means ({n_regimes} regimes)...")
kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
regime_labels_kmeans = kmeans.fit_predict(macro_scaled)
macro_data = macro_data.copy()
macro_data['regime_kmeans'] = regime_labels_kmeans

regime_map_kmeans = dict(zip(macro_data['Year'], macro_data['regime_kmeans']))
df['regime'] = df['Year'].map(regime_map_kmeans)

print("K-Means regime distribution:")
for r in range(n_regimes):
    years = macro_data[macro_data['regime_kmeans'] == r]['Year'].tolist()
    print(f"   Regime {r}: {len(years)} years ({years})")
print("\nApplying GMM (Probabilistic Regime Assignment)...")
gmm = GaussianMixture(n_components=n_regimes, random_state=42)
gmm.fit(macro_scaled)
regime_labels_gmm = gmm.predict(macro_scaled)
regime_probs_gmm  = gmm.predict_proba(macro_scaled)

macro_data['regime_gmm'] = regime_labels_gmm
for i in range(n_regimes):
    macro_data[f'regime_prob_{i}'] = regime_probs_gmm[:, i]

regime_map_gmm = dict(zip(macro_data['Year'], macro_data['regime_gmm']))
df['regime_gmm'] = df['Year'].map(regime_map_gmm)
df['regime_confidence'] = df['Year'].map(
    dict(zip(macro_data['Year'], regime_probs_gmm.max(axis=1)))
)

print("GMM regime distribution:")
for r in range(n_regimes):
    years = macro_data[macro_data['regime_gmm'] == r]['Year'].tolist()
    print(f"   Regime {r}: {len(years)} years ({years})")

regime_characteristics = []
for r in range(n_regimes):
    regime_data = macro_data[macro_data['regime_kmeans'] == r]
    char = {'regime': r, 'years': regime_data['Year'].tolist()}
    for var in macro_vars:
        char[f'avg_{var}'] = regime_data[var].mean()
    regime_characteristics.append(char)

regime_names = {}
for r in range(n_regimes):
    avg_inflation = regime_characteristics[r].get('avg_US Inflation Rate (%)', 0)
    avg_interest  = regime_characteristics[r].get('avg_US Federal Funds Rate (%)', 0)
    avg_gdp       = regime_characteristics[r].get('avg_gdp_growth', 0)
    avg_stress    = regime_characteristics[r].get('avg_stress_intensity', 0)

    if avg_stress > 60:
        regime_names[r] = "HIGH_STRESS"
    elif avg_inflation > 5 and avg_interest > 3:
        regime_names[r] = "HIGH_INFLATION_TIGHTENING"
    elif avg_inflation > 5:
        regime_names[r] = "HIGH_INFLATION"
    elif avg_gdp < 0:
        regime_names[r] = "RECESSION"
    elif avg_interest > 4:
        regime_names[r] = "HIGH_RATES"
    elif avg_interest <= 1:
        regime_names[r] = "ZERO_RATES"
    elif avg_gdp > 3:
        regime_names[r] = "STRONG_GROWTH"
    else:
        regime_names[r] = "NORMAL"

print("\nRegime Characteristics:")
for r in range(n_regimes):
    print(f"\n  Regime {r}: {regime_names[r]}")
    print(f"   Years: {regime_characteristics[r]['years']}")
    for var in macro_vars[:4]:
        val = regime_characteristics[r].get(f'avg_{var}', 0)
        print(f"   {var}: {val:.2f}")

df['regime_name'] = df['regime'].map(regime_names)

pca = PCA(n_components=2)
macro_pca = pca.fit_transform(macro_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label_col, title in zip(axes,
                                 ['regime_kmeans', 'regime_gmm'],
                                 ['K-Means Regime Clusters', 'GMM Regime Clusters (Probabilistic)']):
    scatter = ax.scatter(macro_pca[:, 0], macro_pca[:, 1],
                         c=macro_data[label_col], cmap='viridis', s=100, alpha=0.7)
    for i, year in enumerate(macro_data['Year']):
        ax.annotate(str(year), (macro_pca[i, 0], macro_pca[i, 1]), fontsize=8, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Regime')
plt.tight_layout()
plt.savefig('regime_clusters.png', dpi=150)
print("\nSaved: regime_clusters.png")
plt.close()

fig, ax = plt.subplots(figsize=(14, 6))
colors_palette = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
for r in range(n_regimes):
    mask = macro_data['regime_kmeans'] == r
    ax.scatter(macro_data[mask]['Year'], [r] * mask.sum(),
               c=colors_palette[r % len(colors_palette)], s=100, alpha=0.7,
               label=f'Regime {r}: {regime_names[r]}')

if 'stress_intensity' in macro_data.columns:
    ax2 = ax.twinx()
    ax2.plot(macro_data['Year'], macro_data['stress_intensity'],
             'o-', alpha=0.5, color='gray', label='Stress Intensity')
    ax2.set_ylabel('Stress Intensity', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Regime')
ax.set_yticks(range(n_regimes))
ax.set_yticklabels([f'R{r}' for r in range(n_regimes)])
ax.set_title('Regime Transitions Over Time')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regime_timeline.png', dpi=150)
print("Saved: regime_timeline.png")
plt.close()

print("\nRegime Transition Years:")
prev_regime = None
for _, row in macro_data.sort_values('Year').iterrows():
    if prev_regime is not None and row['regime_kmeans'] != prev_regime:
        print(f"   {int(row['Year'])}: {regime_names[prev_regime]} -> {regime_names[row['regime_kmeans']]}")
    prev_regime = row['regime_kmeans']

silhouette_km  = silhouette_score(macro_scaled, macro_data['regime_kmeans'])
silhouette_gmm = silhouette_score(macro_scaled, macro_data['regime_gmm'])

print(f"\nSilhouette Scores (higher = better separation):")
print(f"   K-Means: {silhouette_km:.4f}")
print(f"   GMM:     {silhouette_gmm:.4f}")

if silhouette_km > 0.5:
    print("   Good separation - regimes are distinct")
elif silhouette_km > 0.3:
    print("   Moderate separation - regimes somewhat distinct")
else:
    print("   Poor separation - regimes may not be well-defined")

print(f"\nRegime Distribution (percentage of observations):")
for r in range(n_regimes):
    count = (df['regime'] == r).sum()
    pct = count / len(df) * 100
    print(f"   Regime {r} ({regime_names[r]}): {count} observations ({pct:.1f}%)")

print("\nCreating Regime Dummy Variables & Interactions...")
for r in range(n_regimes):
    df[f'regime_{r}_dummy'] = (df['regime'] == r).astype(int)
    print(f"   Created regime_{r}_dummy")

key_ratios = ['roa', 'roe', 'debt_ratio', 'profit_margin', 'current_ratio', 'asset_turnover']
available_ratios = [r for r in key_ratios if r in df.columns]

for var in available_ratios:
    for r in range(n_regimes):
        df[f'{var}_x_regime_{r}'] = df[var] * df[f'regime_{r}_dummy']

three_way_vars = ['perfect_storm_risk', 'default_risk_triple', 'inflation_squeeze',
                  'liquidity_crisis', 'rollover_risk', 'pricing_power', 'distress_signal']
available_threeway = [v for v in three_way_vars if v in df.columns]

for var in available_threeway:
    for r in range(n_regimes):
        df[f'{var}_x_regime_{r}'] = df[var] * df[f'regime_{r}_dummy']

print(f"\nFINAL DATASET: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("Layer 3 COMPLETE! Ready for Layer 4 (XGBoost Prediction)")

# Quick economic validation
if 'roa' in df.columns:
    print("\nAverage ROA by Regime:")
    print(df.groupby('regime_name')['roa'].agg(['mean', 'std', 'count']).round(4))

if available_threeway:
    print("\nAverage Perfect Storm Risk by Regime:")
    print(df.groupby('regime_name')['perfect_storm_risk'].agg(['mean', 'std']).round(4))

print("\nAvailable columns in dataframe:")
for i, col in enumerate(df.columns.tolist()):
    print(f"   {i}: {col}")

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LAYER 4: XGBoost PREDICTION MODEL")
print("Predicting: ROA | Profit Margin  (ROE excluded — poor generalization)")
print("="*70)

# [FIX 2] Only reliable targets — ROE removed
RELIABLE_TARGETS = ['roa', 'profit_margin']
available_targets = [t for t in RELIABLE_TARGETS if t in df.columns]
print(f"\nTargets: {available_targets}")
print(f"Excluded: ROE (expected Test R2 ~0.24 — unreliable for stress testing)")

if len(available_targets) == 0:
    raise ValueError("No target variables found. Ensure Layer 1 ran successfully.")

print("\nDefining features from Layers 1-3...")
feature_cols = []

layer1_ratios = ['gross_margin', 'operating_margin', 'debt_ratio', 'debt_to_equity',
                 'equity_ratio', 'current_ratio', 'quick_ratio', 'asset_turnover',
                 'cfo_to_assets', 'cfo_to_sales', 'capex_to_sales', 'firm_size',
                 'roa_volatility', 'cash_ratio']
feature_cols.extend([f for f in layer1_ratios if f in df.columns])
feature_cols.extend([col for col in df.columns if '_lag' in col and 'x_regime' not in col])
feature_cols.extend([col for col in df.columns if '_trend' in col])
feature_cols.extend([col for col in df.columns if '_rolling' in col])
feature_cols.extend([col for col in df.columns if 'interaction' in col and 'x_regime' not in col])
feature_cols.extend([f for f in three_way_vars if f in df.columns])
if 'stress_intensity' in df.columns:
    feature_cols.append('stress_intensity')
if 'regime' in df.columns:
    feature_cols.append('regime')
if 'regime_confidence' in df.columns:
    feature_cols.append('regime_confidence')
feature_cols.extend([col for col in df.columns if 'regime_' in col and 'dummy' in col])
feature_cols.extend([col for col in df.columns if 'x_regime_' in col])

macro_base = ['US Inflation Rate (%)', 'US Federal Funds Rate (%)', 'gdp_growth',
              'S&P 500 Annual Return (%)', 'Sector Benchmark (%)']
feature_cols.extend([f for f in macro_base if f in df.columns])

feature_cols = list(set(feature_cols))
for target in available_targets:
    if target in feature_cols:
        feature_cols.remove(target)
# Also ensure roe is not a feature (since it's a target-family variable)
if 'roe' in feature_cols:
    feature_cols.remove('roe')

print(f"   Total features: {len(feature_cols)}")

model_df = df[['Year', 'Company'] + feature_cols + available_targets].copy()
model_df = model_df.dropna(subset=available_targets)
model_df = model_df.sort_values('Year')
model_df = model_df.replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    if model_df[col].isnull().any():
        model_df[col] = model_df[col].fillna(model_df[col].median())

print(f"   Final shape: {model_df.shape}")

train_idx = model_df['Year'].isin(train_years)
val_idx   = model_df['Year'].isin(val_years)
test_idx  = model_df['Year'].isin(test_years)

X_train = model_df.loc[train_idx, feature_cols]
X_val   = model_df.loc[val_idx,   feature_cols]
X_test  = model_df.loc[test_idx,  feature_cols]

y_train = model_df.loc[train_idx, available_targets]
y_val   = model_df.loc[val_idx,   available_targets]
y_test  = model_df.loc[test_idx,  available_targets]

print(f"\nSplit:")
print(f"   Train: {len(X_train)} rows ({train_years[0]}-{train_years[-1]})")
print(f"   Validation: {len(X_val)} rows ({val_years[0]}-{val_years[-1]})")
print(f"   Test: {len(X_test)} rows ({test_years[0]}-{test_years[-1]})")

print("\nTraining XGBoost models...")

xgb_models  = {}
xgb_results = {}

for target in available_targets:
    print(f"\n   Training for {target}...")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        random_state=42, early_stopping_rounds=30,
        eval_metric='rmse', verbosity=0, n_jobs=-1
    )
    model.fit(X_train, y_train[target],
              eval_set=[(X_val, y_val[target])], verbose=False)
    xgb_models[target] = model

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    xgb_results[target] = {
        'train_pred': y_pred_train, 'val_pred': y_pred_val, 'test_pred': y_pred_test,
        'train_r2':   r2_score(y_train[target], y_pred_train),
        'val_r2':     r2_score(y_val[target],   y_pred_val),
        'test_r2':    r2_score(y_test[target],   y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train[target], y_pred_train)),
        'val_rmse':   np.sqrt(mean_squared_error(y_val[target],   y_pred_val)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test[target],   y_pred_test)),
        'train_mae':  mean_absolute_error(y_train[target], y_pred_train),
        'val_mae':    mean_absolute_error(y_val[target],   y_pred_val),
        'test_mae':   mean_absolute_error(y_test[target],   y_pred_test),
    }
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"\n{'Target':<18} {'Set':<10} {'R2':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 58)
for target in available_targets:
    for set_name in ['train', 'val', 'test']:
        r2   = xgb_results[target][f'{set_name}_r2']
        rmse = xgb_results[target][f'{set_name}_rmse']
        mae  = xgb_results[target][f'{set_name}_mae']
        print(f"{target:<18} {set_name:<10} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}")
    print("-" * 58)

print("\nOverfitting Check:")
for target in available_targets:
    gap = xgb_results[target]['train_r2'] - xgb_results[target]['test_r2']
    flag = "Possible overfitting" if gap > 0.1 else "Good generalization"
    print(f"   {target}: {flag} (train-test gap: {gap:.4f})")

print("\n" + "="*70)
print("FEATURE IMPORTANCE BY TARGET")
print("="*70)
importance_dict = {}
for target in available_targets:
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': xgb_models[target].feature_importances_
    }).sort_values('importance', ascending=False)
    importance_dict[target] = importance_df
    print(f"\nTop 10 features for {target.upper()}:")
    for idx, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {idx+1:2d}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*70)
print("LINEAR REGRESSION BASELINE")
print("="*70)
for target in available_targets:
    lr = LinearRegression()
    lr.fit(X_train, y_train[target])
    y_pred_lr = lr.predict(X_test)
    lr_r2   = r2_score(y_test[target], y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_lr))
    xgb_rmse = xgb_results[target]['test_rmse']
    improvement = (lr_rmse - xgb_rmse) / lr_rmse * 100
    print(f"   {target}: LR R2={lr_r2:.4f}, LR RMSE={lr_rmse:.4f} "
          f"(XGBoost improves RMSE by {improvement:.1f}%)")

if 'regime_name' in model_df.columns:
    print("\n" + "="*70)
    print("REGIME-SPECIFIC PERFORMANCE (Test Set)")
    print("="*70)
    test_data = model_df.loc[test_idx].copy()
    for target in available_targets:
        test_data[f'pred_{target}'] = xgb_results[target]['test_pred']

    print(f"\n{'Regime':<20} {'Target':<15} {'R2':<10} {'n':<8}")
    print("-" * 55)
    for regime in test_data['regime_name'].unique() if 'regime_name' in test_data.columns else []:
        if pd.notna(regime):
            regime_data = test_data[test_data['regime_name'] == regime]
            if len(regime_data) >= 5:
                for target in available_targets:
                    r2_regime = r2_score(regime_data[target], regime_data[f'pred_{target}'])
                    print(f"{regime:<20} {target:<15} {r2_regime:<10.4f} {len(regime_data):<8}")
                print("-" * 55)

print("\nStoring predictions in main dataframe...")
X_all = df[feature_cols].fillna(0)
for target in available_targets:
    df[f'{target}_predicted']  = xgb_models[target].predict(X_all)
    df[f'{target}_error']      = df[target] - df[f'{target}_predicted']
    df[f'{target}_error_pct']  = (df[f'{target}_error'] / df[target].abs().clip(lower=0.01)) * 100

# Cross-target error correlation
if len(available_targets) >= 2:
    print("\nCross-Target Error Correlation:")
    error_cols = [f'{t}_error' for t in available_targets]
    print(df[error_cols].corr().round(4))

try:
    fig, axes = plt.subplots(1, len(available_targets), figsize=(6 * len(available_targets), 5))
    if len(available_targets) == 1:
        axes = [axes]
    for i, target in enumerate(available_targets):
        axes[i].scatter(y_test[target], xgb_results[target]['test_pred'], alpha=0.5, s=30)
        min_val = min(y_test[target].min(), xgb_results[target]['test_pred'].min())
        max_val = max(y_test[target].max(), xgb_results[target]['test_pred'].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[i].set_xlabel(f'Actual {target.upper()}')
        axes[i].set_ylabel(f'Predicted {target.upper()}')
        axes[i].set_title(f'{target.upper()} (Test R2={xgb_results[target]["test_r2"]:.3f})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('layer4_predictions.png', dpi=150)
    print("Saved: layer4_predictions.png")
    plt.close()
except Exception as e:
    print(f"Visualization skipped: {e}")


layer4_results = {
    'models':            xgb_models,
    'features':          feature_cols,
    'targets':           available_targets,
    'predictions':       xgb_results,
    'feature_importance':importance_dict,
    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
    'df': df
}

print("\n" + "="*70)
print("LAYER 4 COMPLETE!")
print(f"   Targets Predicted: {len(available_targets)} ({', '.join(available_targets).upper()})")
print(f"   Features Used: {len(feature_cols)}")
print(f"   Training Period: {train_years[0]}-{train_years[-1]} ({len(X_train)} rows)")
print(f"   Test Period: {test_years[0]}-{test_years[-1]} ({len(X_test)} rows)")
print("\nTEST PERFORMANCE:")
for target in available_targets:
    r2   = xgb_results[target]['test_r2']
    rmse = xgb_results[target]['test_rmse']
    status = "EXCELLENT" if r2 > 0.9 else "GOOD" if r2 > 0.7 else "MODERATE"
    print(f"   {target.upper()}: R2={r2:.4f}, RMSE={rmse:.4f} -> {status}")
print("Layer 4 COMPLETE! Ready for Layer 5 (Scenario Simulation Engine)")


print("="*70)
print("LAYER 5: SCENARIO SIMULATION ENGINE")
print(f"Focus: {', '.join([t.upper() for t in available_targets])}")
print("="*70)

models      = layer4_results['models']
feature_cols= layer4_results['features']

latest_year   = df['Year'].max()
baseline_data = df[df['Year'] == latest_year].iloc[0]

baseline_inflation = baseline_data.get('US Inflation Rate (%)', 3.0)
baseline_interest  = baseline_data.get('US Federal Funds Rate (%)', 4.5)
baseline_gdp       = baseline_data.get('gdp_growth',
                        baseline_data.get('US GDP Growth Rate (%)', 2.5))

print(f"\nBASELINE (Latest available: {latest_year}):")
print(f"   Inflation Rate : {baseline_inflation:.2f}%")
print(f"   Fed Funds Rate : {baseline_interest:.2f}%")
print(f"   GDP Growth     : {baseline_gdp:.2f}%")

scenarios = {
    "BASE":             {"name": "Base Case",               "inflation": baseline_inflation,       "interest": baseline_interest,       "gdp": baseline_gdp,       "color": "blue"},
    "INFLATION_SHOCK":  {"name": "Inflation Shock",         "inflation": baseline_inflation + 8.0, "interest": baseline_interest + 2.0, "gdp": baseline_gdp - 1.5, "color": "red"},
    "RATE_HIKE":        {"name": "Aggressive Rate Hikes",   "inflation": baseline_inflation + 2.0, "interest": baseline_interest + 5.0, "gdp": baseline_gdp - 2.0, "color": "orange"},
    "RECESSION":        {"name": "Severe Recession",        "inflation": baseline_inflation - 2.0, "interest": baseline_interest - 3.0, "gdp": baseline_gdp - 5.0, "color": "darkred"},
    "RECOVERY":         {"name": "Strong Recovery",         "inflation": baseline_inflation + 1.0, "interest": baseline_interest + 1.0, "gdp": baseline_gdp + 3.0, "color": "green"},
    "STAGFLATION":      {"name": "Stagflation",             "inflation": baseline_inflation +10.0, "interest": baseline_interest + 4.0, "gdp": baseline_gdp - 3.0, "color": "purple"},
    "SOFT_LANDING":     {"name": "Soft Landing",            "inflation": baseline_inflation - 2.0, "interest": baseline_interest - 1.0, "gdp": baseline_gdp - 0.5, "color": "teal"},
}

print(f"\nDefined {len(scenarios)} stress scenarios:")
for key, sc in scenarios.items():
    print(f"\n   {key}: {sc['name']}")
    print(f"      Inflation: {sc['inflation']:.1f}% | Interest: {sc['interest']:.1f}% | GDP: {sc['gdp']:.1f}%")


def update_interaction_terms(df_scenario, scenario_values):
    """
    Update all Layer 1-2 interaction terms to reflect new macro values.
    Must be called before prediction to avoid stale features.
    """
    df_s = df_scenario.copy()
    inflation = scenario_values['inflation']
    interest  = scenario_values['interest']
    gdp       = scenario_values['gdp']

    if 'US Inflation Rate (%)'  in df_s.columns: df_s['US Inflation Rate (%)']  = inflation
    if 'US Federal Funds Rate (%)'in df_s.columns:df_s['US Federal Funds Rate (%)']= interest
    if 'gdp_growth'             in df_s.columns: df_s['gdp_growth']              = gdp
    if 'US GDP Growth Rate (%)'  in df_s.columns: df_s['US GDP Growth Rate (%)']  = gdp

    if 'debt_ratio' in df_s.columns:
        df_s['inflation_debt_interaction'] = inflation * df_s['debt_ratio']
        df_s['rate_debt_interaction']      = interest  * df_s['debt_ratio']
        if 'debt_to_equity' in df_s.columns:
            df_s['gdp_leverage_interaction'] = gdp * df_s['debt_to_equity']
    if 'current_ratio' in df_s.columns:
        df_s['rate_liquidity_interaction'] = interest * (1 / df_s['current_ratio'].clip(lower=0.1))
    if 'gross_margin' in df_s.columns:
        df_s['inflation_margin_interaction'] = inflation * (1 / df_s['gross_margin'].clip(lower=0.01))
    if 'asset_turnover' in df_s.columns:
        df_s['gdp_efficiency_interaction']  = gdp       * df_s['asset_turnover']
        df_s['turnover_inflation_interaction']= inflation * df_s['asset_turnover']
    if 'profit_margin' in df_s.columns:
        df_s['margin_rate_interaction']     = df_s['profit_margin'] * interest
    if 'firm_size' in df_s.columns:
        df_s['size_gdp_interaction']        = df_s['firm_size'] * gdp
    if 'quick_ratio' in df_s.columns:
        df_s['liquidity_gdp_interaction']   = df_s['quick_ratio'] * gdp

    if all(x in df_s.columns for x in ['roa_volatility', 'debt_ratio']):
        df_s['volatility_debt_interaction'] = df_s['roa_volatility'] * df_s['debt_ratio']
        df_s['rollover_risk']               = df_s['roa_volatility'] * df_s['debt_ratio'] * interest
    if all(x in df_s.columns for x in ['debt_ratio', 'profit_margin']):
        df_s['inflation_squeeze']           = inflation * (1 / df_s['profit_margin'].abs().clip(lower=0.01)) * df_s['debt_ratio']
    if all(x in df_s.columns for x in ['debt_to_equity', 'current_ratio']):
        df_s['liquidity_crisis']            = max(0, -gdp) * df_s['debt_to_equity'] * (1 / df_s['current_ratio'].clip(lower=0.01))
    if all(x in df_s.columns for x in ['asset_turnover', 'profit_margin']):
        df_s['pricing_power']               = inflation * df_s['asset_turnover'] * df_s['profit_margin']

    # Stress intensity
    inf_stress  = min(0.25, max(0, inflation) / 10)
    rate_stress = min(0.25, max(0, interest) / 8)
    gdp_stress  = min(0.25, max(0, -gdp) / 20)
    df_s['stress_intensity'] = (inf_stress + rate_stress + gdp_stress) * 100

    return df_s


def simulate_scenario(df, scenario_values, models, feature_cols, target):
    """Simulate firm performance under a given macro scenario."""
    latest_data = df.groupby('Company').last().reset_index()
    sim_data = update_interaction_terms(latest_data, scenario_values)
    for col in [c for c in feature_cols if c not in sim_data.columns]:
        sim_data[col] = 0
    X_sim = sim_data[feature_cols].fillna(0)
    if target in models:
        sim_data[f'predicted_{target}'] = models[target].predict(X_sim)
    else:
        sim_data[f'predicted_{target}'] = 0
    return sim_data


scenario_results = {}
print("\nRunning simulations...")
for target in available_targets:
    print(f"\nSimulating for {target.upper()}:")
    scenario_results[target] = {}
    for scenario_key, scenario_vals in scenarios.items():
        sv = {'inflation': scenario_vals['inflation'],
              'interest':  scenario_vals['interest'],
              'gdp':       scenario_vals['gdp']}
        sim_result = simulate_scenario(df, sv, models, feature_cols, target)
        scenario_results[target][scenario_key] = sim_result
        avg_pred = sim_result[f'predicted_{target}'].mean()
        if target == 'profit_margin':
            print(f"   {scenario_key:20s}: Avg {target.upper()} = {avg_pred:.4f} ({avg_pred*100:.2f}%)")
        else:
            print(f"   {scenario_key:20s}: Avg {target.upper()} = {avg_pred:.4f}")

print("\n" + "="*70)
print("STRESS IMPACT ANALYSIS")
print("="*70)
stress_impacts = {}
for target in available_targets:
    print(f"\n{target.upper()} Stress Impacts:")
    print("-" * 50)
    baseline_pred = scenario_results[target]['BASE'][f'predicted_{target}']
    for scenario_key in ['INFLATION_SHOCK', 'RATE_HIKE', 'RECESSION', 'STAGFLATION', 'SOFT_LANDING']:
        if scenario_key in scenario_results[target]:
            sp = scenario_results[target][scenario_key][f'predicted_{target}']
            impact_pct = ((sp - baseline_pred) / baseline_pred.abs().clip(lower=0.001) * 100).mean()
            stress_impacts.setdefault(target, []).append({
                'scenario':      scenario_key,
                'scenario_name': scenarios[scenario_key]['name'],
                'impact_pct':    impact_pct,
            })
            arrow = "UP" if impact_pct > 0 else "DOWN"
            print(f"   {arrow} {scenario_key:20s}: {impact_pct:+.1f}% change from baseline")

print("\n" + "="*70)
print("FIRM-LEVEL RISK SCORES (0-100)")
print("="*70)
risk_scores = []

for target in available_targets:
    stag_df = scenario_results[target]['STAGFLATION']
    base_df = scenario_results[target]['BASE']

    for _, row in stag_df.iterrows():
        company = row['Company']
        sector  = row.get('Sector', 'Unknown')
        base_row = base_df[base_df['Company'] == company]
        if len(base_row) == 0:
            continue

        base_pred   = base_row[f'predicted_{target}'].values[0]
        stress_pred = row[f'predicted_{target}']

        stress_vulnerability = max(0, min(100,
            (base_pred - stress_pred) / abs(base_pred) * 100 if abs(base_pred) > 0.001 else 50))
        leverage     = min(100, row.get('debt_ratio', 0.5) * 100)
        hist_roa     = df[df['Company'] == company]['roa'].values if 'roa' in df.columns else []
        volatility   = min(100, np.std(hist_roa) * 100 * 2) if len(hist_roa) > 1 else 20
        liquidity    = max(0, min(100, (1 / (row.get('current_ratio', 1.0) + 0.01)) * 50))

        risk_score = (0.40 * stress_vulnerability + 0.30 * leverage +
                      0.20 * volatility + 0.10 * liquidity)

        risk_scores.append({
            'Company':  company, 'Sector': sector, 'Target': target.upper(),
            'Risk_Score': round(risk_score, 2),
            'Stress_Vulnerability': round(stress_vulnerability, 2),
            'Leverage_Risk': round(leverage, 2),
            'Volatility_Risk': round(volatility, 2),
            'Liquidity_Risk': round(liquidity, 2),
            f'Baseline_{target.upper()}':    round(base_pred, 4),
            f'Stagflation_{target.upper()}': round(stress_pred, 4),
        })

risk_df = pd.DataFrame(risk_scores)
primary_risk = risk_df[risk_df['Target'] == 'ROA'].drop_duplicates(subset=['Company'])

print("\nTOP 10 HIGHEST RISK FIRMS (Most Vulnerable to Stagflation):")
print("-" * 60)
for _, row in primary_risk.nlargest(10, 'Risk_Score').iterrows():
    level = "HIGH" if row['Risk_Score'] > 60 else "MED" if row['Risk_Score'] > 35 else "LOW"
    print(f"   {row['Company']:25s} | {level} | Score: {row['Risk_Score']:5.1f}")

print("\nTOP 10 LOWEST RISK FIRMS (Most Resilient):")
print("-" * 60)
for _, row in primary_risk.nsmallest(10, 'Risk_Score').iterrows():
    print(f"   {row['Company']:25s} | Score: {row['Risk_Score']:5.1f}")

sector_performance = []
for target in available_targets:
    for scenario_key in ['BASE', 'STAGFLATION', 'RECESSION', 'INFLATION_SHOCK']:
        if scenario_key in scenario_results[target]:
            sc_df = scenario_results[target][scenario_key]
            if 'Sector' in sc_df.columns:
                sector_avg = sc_df.groupby('Sector')[f'predicted_{target}'].mean().reset_index()
                sector_avg['Scenario'] = scenario_key
                sector_avg['Target']   = target.upper()
                sector_performance.append(sector_avg)

if sector_performance:
    sector_perf_df = pd.concat(sector_performance)
    stag_sector = sector_perf_df[
        (sector_perf_df['Scenario'] == 'STAGFLATION') &
        (sector_perf_df['Target'] == 'ROA')
    ].sort_values('predicted_roa', ascending=False)

    print("\nSector Resilience Under Stagflation (ROA, higher = more resilient):")
    print("-" * 50)
    for _, row in stag_sector.iterrows():
        print(f"   {row['Sector']:25s}: {row['predicted_roa']:.4f}")
else:
    sector_perf_df = pd.DataFrame()

print("\n" + "="*70)
print("MONTE CARLO SIMULATION (Uncertainty Quantification)")
print("="*70)

def monte_carlo_simulation(df, models, feature_cols, target, n_sims=500):
    print(f"\nRunning {n_sims} simulations for {target.upper()}...")
    hist_inflation = df['US Inflation Rate (%)'].dropna().values
    hist_interest  = df['US Federal Funds Rate (%)'].dropna().values
    hist_gdp_col   = 'gdp_growth' if 'gdp_growth' in df.columns else 'US GDP Growth Rate (%)'
    hist_gdp       = df[hist_gdp_col].dropna().values
    latest_data    = df.groupby('Company').last().reset_index()
    all_predictions= []

    for _ in range(n_sims):
        scenario = {
            'inflation': np.random.choice(hist_inflation) + np.random.normal(0, 0.5),
            'interest':  np.random.choice(hist_interest)  + np.random.normal(0, 0.3),
            'gdp':       np.random.choice(hist_gdp)       + np.random.normal(0, 0.5),
        }
        sim_data = update_interaction_terms(latest_data, scenario)
        for col in [c for c in feature_cols if c not in sim_data.columns]:
            sim_data[col] = 0
        X_sim = sim_data[feature_cols].fillna(0)
        if target in models:
            all_predictions.append(models[target].predict(X_sim).mean())

    all_predictions = np.array(all_predictions)
    results = {
        'mean':          np.mean(all_predictions),
        'std':           np.std(all_predictions),
        'ci_90_lower':   np.percentile(all_predictions, 5),
        'ci_90_upper':   np.percentile(all_predictions, 95),
        'ci_95_lower':   np.percentile(all_predictions, 2.5),
        'ci_95_upper':   np.percentile(all_predictions, 97.5),
        'simulations':   all_predictions,
    }
    print(f"   Mean: {results['mean']:.4f}")
    print(f"   Std Dev: {results['std']:.4f}")
    print(f"   95% CI: [{results['ci_95_lower']:.4f}, {results['ci_95_upper']:.4f}]")
    return results

mc_results = {}
for target in available_targets:
    mc_results[target] = monte_carlo_simulation(df, models, feature_cols, target, n_sims=500)

warning_signals = []
for target in available_targets:
    stag_df = scenario_results[target]['STAGFLATION']
    base_df = scenario_results[target]['BASE']
    for _, row in stag_df.iterrows():
        company  = row['Company']
        sector   = row.get('Sector', 'Unknown')
        base_row = base_df[base_df['Company'] == company]
        if len(base_row) == 0:
            continue
        base_pred   = base_row[f'predicted_{target}'].values[0]
        stress_pred = row[f'predicted_{target}']
        drop_pct    = (base_pred - stress_pred) / abs(base_pred) * 100 if abs(base_pred) > 0.001 else 0
        warnings_list = []
        if drop_pct > 30:
            warnings_list.append(f"CRITICAL: {drop_pct:.0f}% drop under stagflation")
        if row.get('debt_ratio', 0) > 0.7:
            warnings_list.append(f"HIGH LEVERAGE: {row.get('debt_ratio', 0):.1%}")
        if row.get('current_ratio', 1) < 1:
            warnings_list.append(f"LIQUIDITY CRISIS: Current ratio = {row.get('current_ratio', 0):.2f}")
        if warnings_list:
            warning_signals.append({
                'Company': company, 'Sector': sector, 'Target': target.upper(),
                'Warnings': ' | '.join(warnings_list),
                'Stress_Drop_%': round(drop_pct, 1),
                'Leverage': round(row.get('debt_ratio', 0), 3),
                'Current_Ratio': round(row.get('current_ratio', 0), 2),
            })

if warning_signals:
    warning_df_l5 = pd.DataFrame(warning_signals).sort_values('Stress_Drop_%', ascending=False)
    print("\nFIRMS REQUIRING IMMEDIATE ATTENTION:")
    for _, row in warning_df_l5.head(15).iterrows():
        print(f"\n   [!] {row['Company']} ({row['Sector']})")
        print(f"      {row['Warnings']}")
else:
    print("\n   No critical warning signals detected")
    warning_df_l5 = pd.DataFrame()

fig, ax = plt.subplots(figsize=(12, 6))
scenario_names_plot = ['INFLATION_SHOCK', 'RATE_HIKE', 'RECESSION', 'STAGFLATION']
x = np.arange(len(scenario_names_plot))
width = 0.35
for i, target in enumerate(available_targets):
    values = []
    for sc in scenario_names_plot:
        impacts = stress_impacts.get(target, [])
        val = next((imp['impact_pct'] for imp in impacts if imp['scenario'] == sc), 0)
        values.append(val)
    offset = (i - len(available_targets) / 2 + 0.5) * width
    bars = ax.bar(x + offset, values, width, label=target.upper())
    for bar, val in zip(bars, values):
        bar.set_color('green' if val > 0 else 'red')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Scenario')
ax.set_ylabel('Impact (% Change from Baseline)')
ax.set_title('Scenario Impacts by Target')
ax.set_xticks(x)
ax.set_xticklabels(scenario_names_plot, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('layer5_scenario_impacts.png', dpi=150)
print("\nSaved: layer5_scenario_impacts.png")
plt.close()

fig, axes = plt.subplots(1, len(available_targets), figsize=(6 * len(available_targets), 4))
if len(available_targets) == 1:
    axes = [axes]
for idx, target in enumerate(available_targets):
    if target in mc_results:
        ax = axes[idx]
        sims = mc_results[target]['simulations']
        # Plot histogram FIRST, then read ylim
        ax.hist(sims, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ymax = ax.get_ylim()[1]           # [FIX 5] correct — called after hist()
        ax.axvline(mc_results[target]['mean'], color='red', linewidth=2,
                   label=f"Mean: {mc_results[target]['mean']:.4f}")
        ax.axvline(mc_results[target]['ci_95_lower'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(mc_results[target]['ci_95_upper'], color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.fill_betweenx([0, ymax],
                          mc_results[target]['ci_95_lower'],
                          mc_results[target]['ci_95_upper'],
                          alpha=0.2, color='red', label='95% CI')
        ax.set_xlabel(f'Predicted {target.upper()}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Monte Carlo: {target.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('layer5_monte_carlo.png', dpi=150)
print("Saved: layer5_monte_carlo.png")
plt.close()

if len(primary_risk) > 0 and 'Sector' in primary_risk.columns:
    sector_risk_bar = primary_risk.groupby('Sector')['Risk_Score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_bar = ['darkred' if x > 60 else 'red' if x > 40 else 'orange' if x > 25 else 'green'
                  for x in sector_risk_bar.values]
    sector_risk_bar.plot(kind='bar', ax=ax, color=colors_bar)
    ax.set_ylabel('Average Risk Score (0-100)')
    ax.set_title('Risk Score by Sector (Higher = More Vulnerable to Stagflation)')
    ax.axhline(y=50, color='darkred', linestyle='--', alpha=0.5, label='High Risk (50+)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Medium Risk (30+)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('layer5_risk_by_sector.png', dpi=150)
    print("Saved: layer5_risk_by_sector.png")
    plt.close()

layer5_results = {
    'scenario_results': scenario_results,
    'mc_results':       mc_results,
    'risk_df':          risk_df,
    'primary_risk':     primary_risk,
    'warning_signals':  warning_signals,
    'sector_perf_df':   sector_perf_df,
    'stress_impacts':   stress_impacts,
}
globals().update(layer5_results)

print("\n" + "="*70)
print("LAYER 5 COMPLETE! Ready for Layer 6 (SHAP Analysis)")
print("="*70)

try:
    import shap
    print("SHAP library loaded successfully")
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '--quiet'])
    import shap
    print("SHAP installed and loaded")

print("="*70)
print("LAYER 6: SHAP ANALYSIS - Model Interpretability")
print("="*70)

models_shap      = layer4_results['models']
feature_cols_shap= layer4_results['features']

background_sample  = df[feature_cols_shap].sample(min(500, len(df)), random_state=42).fillna(0)
explanation_sample = df[feature_cols_shap].sample(min(200, len(df)), random_state=42).fillna(0)

print(f"\nBackground data size: {len(background_sample)} rows")
print(f"Explanation data size: {len(explanation_sample)} rows")

shap_importance_roa = None
shap_importance_pm  = None
top_risk_drivers    = None
explainer_roa       = None
explainer_pm        = None
shap_values_roa     = None
shap_values_pm      = None

if 'roa' in models_shap:
    model_roa = models_shap['roa']
    print("\nCreating SHAP explainer for ROA...")
    explainer_roa    = shap.TreeExplainer(model_roa)
    shap_values_roa  = explainer_roa.shap_values(explanation_sample)
    print("   SHAP values calculated")

    shap_importance_roa = pd.DataFrame({
        'feature':         feature_cols_shap,
        'shap_importance': np.abs(shap_values_roa).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print("\nTOP 15 MOST IMPORTANT FEATURES FOR ROA:")
    for i, (_, row) in enumerate(shap_importance_roa.head(15).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:35s}: {row['shap_importance']:.6f}")

    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_roa, explanation_sample,
                      feature_names=feature_cols_shap, show=False)
    plt.tight_layout()
    plt.savefig('layer6_shap_summary_roa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer6_shap_summary_roa.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_roa, explanation_sample,
                      feature_names=feature_cols_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('layer6_shap_bar_roa.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer6_shap_bar_roa.png")

    high_risk_firms = ["Lowe's", 'PepsiCo', 'Amgen']
    sample_firm = None
    for firm in high_risk_firms:
        firm_data = df[df['Company'] == firm]
        if len(firm_data) > 0:
            sample_firm = firm_data.iloc[0]
            break

    if sample_firm is not None:
        firm_features = sample_firm[feature_cols_shap].fillna(0).values.reshape(1, -1)
        firm_pred     = model_roa.predict(firm_features)[0]
        firm_shap     = explainer_roa.shap_values(firm_features)
        plt.figure(figsize=(12, 10))
        shap.waterfall_plot(shap.Explanation(
            values=firm_shap[0],
            base_values=explainer_roa.expected_value,
            data=firm_features[0],
            feature_names=feature_cols_shap), show=False)
        plt.title(f"ROA Prediction Breakdown: {sample_firm['Company']}\nPredicted ROA: {firm_pred:.4f}")
        plt.tight_layout()
        plt.savefig(f"layer6_waterfall_{sample_firm['Company'].replace(' ', '_')}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: layer6_waterfall_{sample_firm['Company'].replace(' ', '_')}.png")

    
    top_features_dep = shap_importance_roa.head(6)['feature'].tolist()
    for feature in top_features_dep:
        try:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values_roa, explanation_sample,
                                 feature_names=feature_cols_shap, show=False)
            plt.title(f'SHAP Dependence: {feature} (ROA)')
            plt.tight_layout()
            plt.savefig(f'layer6_dependence_{feature}_roa.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: layer6_dependence_{feature}_roa.png")
        except Exception as e:
            print(f"   Dependence plot skipped for {feature}: {e}")

    all_features_shap  = df[feature_cols_shap].fillna(0)
    all_shap_roa       = explainer_roa.shap_values(all_features_shap)
    negative_contrib   = pd.DataFrame(all_shap_roa, columns=feature_cols_shap)
    negative_contrib   = negative_contrib[negative_contrib < 0].abs()
    top_risk_drivers   = negative_contrib.mean().sort_values(ascending=False).head(10)

    print("\nTop Risk Drivers (Features that LOWER ROA):")
    for feat, impact in top_risk_drivers.items():
        print(f"   {feat:35s}: {impact:.6f}")

if 'profit_margin' in models_shap:
    model_pm = models_shap['profit_margin']
    print("\nCreating SHAP explainer for Profit Margin...")
    explainer_pm   = shap.TreeExplainer(model_pm)
    shap_values_pm = explainer_pm.shap_values(explanation_sample)
    print("   SHAP values calculated")

    shap_importance_pm = pd.DataFrame({
        'feature':         feature_cols_shap,
        'shap_importance': np.abs(shap_values_pm).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print("\nTop 15 most important features for PROFIT MARGIN:")
    for i, (_, row) in enumerate(shap_importance_pm.head(15).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:35s}: {row['shap_importance']:.6f}")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_pm, explanation_sample,
                      feature_names=feature_cols_shap, show=False)
    plt.tight_layout()
    plt.savefig('layer6_shap_summary_profit_margin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer6_shap_summary_profit_margin.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pm, explanation_sample,
                      feature_names=feature_cols_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('layer6_shap_bar_profit_margin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer6_shap_bar_profit_margin.png")

if shap_importance_roa is not None and shap_importance_pm is not None:
    comparison_df = shap_importance_roa.merge(
        shap_importance_pm, on='feature', suffixes=('_roa', '_pm'))
    epsilon = 1e-8
    comparison_df['ratio_roa_over_pm'] = (
        comparison_df['shap_importance_roa'] / (comparison_df['shap_importance_pm'] + epsilon)
    )

    top_10_features = shap_importance_roa.head(10)['feature'].tolist()
    roa_imp = [shap_importance_roa[shap_importance_roa['feature'] == f]['shap_importance'].values[0]
               for f in top_10_features]
    pm_imp  = [(shap_importance_pm[shap_importance_pm['feature'] == f]['shap_importance'].values[0]
                if f in shap_importance_pm['feature'].values else 0)
               for f in top_10_features]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos  = np.arange(len(top_10_features))
    height = 0.35
    ax.barh(y_pos - height / 2, roa_imp, height, label='ROA', color='steelblue')
    ax.barh(y_pos + height / 2, pm_imp,  height, label='Profit Margin', color='coral')
    ax.set_xlabel('SHAP Importance (Mean |SHAP Value|)')
    ax.set_title('Feature Importance Comparison: ROA vs Profit Margin')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10_features)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('layer6_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer6_importance_comparison.png")


if shap_importance_roa is not None:
    shap_importance_roa.to_csv('layer6_shap_importance_roa.csv', index=False)
    print("Saved: layer6_shap_importance_roa.csv")
    shap_df = pd.DataFrame(all_shap_roa, columns=feature_cols_shap)
    shap_df['Company'] = df['Company'].values
    shap_df['Year']    = df['Year'].values
    shap_df.to_csv('layer6_shap_values_all_firms.csv', index=False)
    print("Saved: layer6_shap_values_all_firms.csv")

if shap_importance_pm is not None:
    shap_importance_pm.to_csv('layer6_shap_importance_profit_margin.csv', index=False)
    print("Saved: layer6_shap_importance_profit_margin.csv")

layer6_results = {
    'shap_importance_roa': shap_importance_roa,
    'shap_importance_pm':  shap_importance_pm,
    'shap_values_roa':     shap_values_roa,
    'shap_values_pm':      shap_values_pm,
    'explainer_roa':       explainer_roa,
    'explainer_pm':        explainer_pm,
    'feature_cols':        feature_cols_shap,
    'top_risk_drivers':    top_risk_drivers,
}

print("\n" + "="*70)
print("LAYER 6 COMPLETE! Ready for Layer 7 (Risk Dashboard)")
print("="*70)


import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("="*70)
print("LAYER 7: RISK DASHBOARD - Comprehensive Risk Reporting")
print("="*70)


scenario_results_l7 = layer5_results.get('scenario_results', {})
risk_df_l7          = layer5_results.get('risk_df', pd.DataFrame())
warning_signals_l7  = layer5_results.get('warning_signals', [])
mc_results_l7       = layer5_results.get('mc_results', {})

top_risk_drivers_l7 = layer6_results.get('top_risk_drivers', None)

if len(risk_df_l7) == 0 and 'primary_risk' in layer5_results:
    risk_df_l7 = layer5_results['primary_risk']

print(f"   Risk firms loaded: {len(risk_df_l7)}")
print(f"   Warning signals: {len(warning_signals_l7)}")

def get_risk_level(score):
    return "CRITICAL" if score >= 60 else "HIGH" if score >= 40 else "MEDIUM" if score >= 25 else "LOW"

if len(risk_df_l7) > 0:
    risk_ranking = risk_df_l7.sort_values('Risk_Score', ascending=False).reset_index(drop=True)
    risk_ranking['Rank']       = risk_ranking.index + 1
    risk_ranking['Risk_Level'] = risk_ranking['Risk_Score'].apply(get_risk_level)

    print("\nTop 20 Highest Risk Firms:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Company':<25} {'Sector':<20} {'Risk Score':<12} {'Level':<10}")
    print("-" * 80)
    for _, row in risk_ranking.head(20).iterrows():
        print(f"{row['Rank']:<5} {row['Company']:<25} {row.get('Sector','N/A'):<20} "
              f"{row['Risk_Score']:<12.1f} {row['Risk_Level']:<10}")

    risk_ranking.to_csv('layer7_risk_ranking.csv', index=False)
    print("\nSaved: layer7_risk_ranking.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    top_20 = risk_ranking.head(20)
    colors_r = ['darkred' if x >= 60 else 'red' if x >= 40 else 'orange' if x >= 25 else 'green'
                for x in top_20['Risk_Score']]
    bars = ax.barh(range(len(top_20)), top_20['Risk_Score'], color=colors_r)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Company'])
    ax.set_xlabel('Risk Score (0-100)', fontsize=12)
    ax.set_title('Top 20 Highest Risk Firms', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{top_20["Risk_Score"].iloc[i]:.1f}', va='center', fontsize=9)
    ax.axvline(x=60, color='darkred', linestyle='--', alpha=0.7, label='Critical (60+)')
    ax.axvline(x=40, color='red',     linestyle='--', alpha=0.7, label='High (40+)')
    ax.axvline(x=25, color='orange',  linestyle='--', alpha=0.7, label='Medium (25+)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('layer7_risk_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer7_risk_ranking.png")

sector_risk_summary = pd.DataFrame()
if len(risk_df_l7) > 0 and 'Sector' in risk_df_l7.columns:
    sector_risk_summary = risk_df_l7.groupby('Sector').agg(
        Avg_Risk=('Risk_Score', 'mean'),
        Std_Risk=('Risk_Score', 'std'),
        Min_Risk=('Risk_Score', 'min'),
        Max_Risk=('Risk_Score', 'max'),
        Firm_Count=('Risk_Score', 'count'),
    ).round(2).sort_values('Avg_Risk', ascending=False)

    print("\nSector Risk Summary:")
    print("-" * 70)
    print(f"{'Sector':<20} {'Avg Risk':<12} {'Min':<8} {'Max':<8} {'Count':<8}")
    print("-" * 70)
    for sector, row in sector_risk_summary.iterrows():
        flag = "!" if row['Avg_Risk'] >= 40 else "?" if row['Avg_Risk'] >= 25 else "."
        print(f"{flag} {sector:<18} {row['Avg_Risk']:<12.1f} {row['Min_Risk']:<8.1f} "
              f"{row['Max_Risk']:<8.1f} {row['Firm_Count']:<8}")

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_r   = sector_risk_summary['Avg_Risk'].values
    colors_s= ['darkred' if x >= 40 else 'red' if x >= 25 else 'orange' if x >= 15 else 'green'
                for x in avg_r]
    bars = ax.bar(sector_risk_summary.index.tolist(), avg_r, color=colors_s)
    ax.set_ylabel('Average Risk Score (0-100)', fontsize=12)
    ax.set_title('Sector Risk Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=40, color='darkred', linestyle='--', alpha=0.7, label='High Risk Threshold')
    ax.axhline(y=25, color='orange',  linestyle='--', alpha=0.7, label='Medium Risk Threshold')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, avg_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('layer7_sector_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer7_sector_risk.png")
    sector_risk_summary.to_csv('layer7_sector_risk_summary.csv')
    print("Saved: layer7_sector_risk_summary.csv")

if warning_signals_l7:
    warning_df_l7 = pd.DataFrame(warning_signals_l7).sort_values('Stress_Drop_%', ascending=False)
    print("\nFirms with Critical Warnings:")
    for _, row in warning_df_l7.head(15).iterrows():
        print(f"\n   [!] {row['Company']} ({row.get('Sector','N/A')})")
        print(f"      {row['Warnings']}")

    warning_types = []
    for w in warning_signals_l7:
        if 'HIGH LEVERAGE' in w['Warnings']:    warning_types.append('High Leverage')
        if 'LIQUIDITY CRISIS' in w['Warnings']: warning_types.append('Liquidity Crisis')
        if 'CRITICAL' in w['Warnings']:         warning_types.append('Critical Drop')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if warning_types:
        wc = pd.Series(warning_types).value_counts()
        axes[0].pie(wc.values, labels=wc.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Types of Warning Signals', fontweight='bold')

    firm_wc = warning_df_l7.groupby('Company').size().sort_values(ascending=False).head(10)
    if len(firm_wc) > 0:
        colors_w = ['darkred' if x > 2 else 'red' if x > 1 else 'orange' for x in firm_wc.values]
        axes[1].barh(range(len(firm_wc)), firm_wc.values, color=colors_w)
        axes[1].set_yticks(range(len(firm_wc)))
        axes[1].set_yticklabels(firm_wc.index)
        axes[1].set_xlabel('Number of Warnings')
        axes[1].set_title('Firms with Multiple Warning Signals', fontweight='bold')
        axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig('layer7_warning_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: layer7_warning_dashboard.png")
    warning_df_l7.to_csv('layer7_warning_signals.csv', index=False)
    print("Saved: layer7_warning_signals.csv")
else:
    print("\nNo critical warning signals detected")
    warning_types = []

if scenario_results_l7:
    impact_data = []
    for target, scenarios_t in scenario_results_l7.items():
        for scenario, df_result in scenarios_t.items():
            pred_col = f'predicted_{target}'
            if pred_col in df_result.columns:
                impact_data.append({'Target': target.upper(), 'Scenario': scenario,
                                    'Predicted_Value': df_result[pred_col].mean()})
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        pivot_impacts = impact_df.pivot(index='Scenario', columns='Target', values='Predicted_Value')
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot_impacts.values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(pivot_impacts.columns)))
        ax.set_xticklabels(pivot_impacts.columns)
        ax.set_yticks(range(len(pivot_impacts.index)))
        ax.set_yticklabels(pivot_impacts.index)
        plt.colorbar(im, ax=ax, label='Predicted Value')
        ax.set_title('Scenario Impact Heatmap', fontsize=14, fontweight='bold')
        for i in range(len(pivot_impacts.index)):
            for j in range(len(pivot_impacts.columns)):
                ax.text(j, i, f'{pivot_impacts.values[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9)
        plt.tight_layout()
        plt.savefig('layer7_scenario_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: layer7_scenario_heatmap.png")

if top_risk_drivers_l7 is not None:
    print("\nFeatures that most negatively impact profitability:")
    for i, (feature, impact) in enumerate(top_risk_drivers_l7.head(10).items(), 1):
        print(f"   {i:2d}. {feature:35s}: {impact:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    feats   = list(top_risk_drivers_l7.head(10).index)
    impacts = list(top_risk_drivers_l7.head(10).values)
    colors_d= ['darkred' if i < 3 else 'red' if i < 6 else 'orange' for i in range(len(impacts))]
    bars = ax.barh(range(len(feats)), impacts, color=colors_d)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel('Average Negative Impact on ROA')
    ax.set_title('Top 10 Risk Drivers', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{impacts[i]:.5f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('layer7_risk_drivers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer7_risk_drivers.png")

if mc_results_l7:
    print("\nPrediction Confidence Intervals:")
    for target, results in mc_results_l7.items():
        print(f"\n   {target.upper()}:")
        print(f"      Mean: {results['mean']:.4f}")
        print(f"      95% CI: [{results['ci_95_lower']:.4f}, {results['ci_95_upper']:.4f}]")

    fig, axes = plt.subplots(1, len(mc_results_l7), figsize=(6 * len(mc_results_l7), 4))
    if len(mc_results_l7) == 1:
        axes = [axes]
    for idx, (target, results) in enumerate(mc_results_l7.items()):
        ax = axes[idx]
        sims = results.get('simulations', [])
        if len(sims) > 0:
            ax.hist(sims, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ymax = ax.get_ylim()[1]                    # [FIX 5] after hist()
            ax.axvline(results['mean'], color='red', linewidth=2,
                       label=f"Mean: {results['mean']:.4f}")
            ax.axvline(results['ci_95_lower'], color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(results['ci_95_upper'], color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.fill_betweenx([0, ymax],
                              results['ci_95_lower'], results['ci_95_upper'],
                              alpha=0.2, color='red', label='95% CI')
            ax.set_xlabel(f'Predicted {target.upper()}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Uncertainty Distribution: {target.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('layer7_monte_carlo_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: layer7_monte_carlo_dashboard.png")

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
if len(risk_df_l7) > 0:
    top_10 = risk_ranking.head(10)
    colors_e = ['darkred' if x >= 60 else 'red' if x >= 40 else 'orange' if x >= 25 else 'green'
                for x in top_10['Risk_Score']]
    ax1.barh(range(len(top_10)), top_10['Risk_Score'], color=colors_e)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Company'], fontsize=8)
    ax1.set_xlabel('Risk Score')
    ax1.set_title('Top 10 Highest Risk Firms', fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    ax1.axvline(x=60, color='darkred', linestyle='--', alpha=0.5)
    ax1.axvline(x=40, color='red',     linestyle='--', alpha=0.5)

ax2 = fig.add_subplot(gs[0, 1])
if len(sector_risk_summary) > 0:
    secs   = sector_risk_summary.index.tolist()[:6]
    avg_rs = sector_risk_summary['Avg_Risk'].values[:6]
    colors_e2 = ['darkred' if x >= 40 else 'red' if x >= 25 else 'orange' for x in avg_rs]
    ax2.bar(secs, avg_rs, color=colors_e2)
    ax2.set_ylabel('Avg Risk Score')
    ax2.set_title('Sector Risk Comparison', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)

ax3 = fig.add_subplot(gs[0, 2])
if warning_types:
    wc3 = pd.Series(warning_types).value_counts()
    ax3.pie(wc3.values, labels=wc3.index, autopct='%1.0f%%', startangle=90)
    ax3.set_title('Warning Signal Types', fontsize=10, fontweight='bold')

ax4 = fig.add_subplot(gs[1, :])
if top_risk_drivers_l7 is not None:
    td = list(top_risk_drivers_l7.head(8).index)
    ti = list(top_risk_drivers_l7.head(8).values)
    colors_e4 = ['darkred'] * 3 + ['red'] * 3 + ['orange'] * 2
    ax4.barh(range(len(td)), ti, color=colors_e4)
    ax4.set_yticks(range(len(td)))
    ax4.set_yticklabels(td, fontsize=9)
    ax4.set_xlabel('Average Negative Impact')
    ax4.set_title('Top 8 Risk Drivers (What Hurts Profitability Most)', fontsize=10, fontweight='bold')
    ax4.invert_yaxis()

ax5 = fig.add_subplot(gs[2, 0:2])
if scenario_results_l7 and 'roa' in scenario_results_l7:
    sc_names = ['BASE', 'INFLATION_SHOCK', 'RATE_HIKE', 'RECESSION', 'STAGFLATION', 'SOFT_LANDING']
    roa_vals = [scenario_results_l7['roa'][sc][f'predicted_roa'].mean()
                if sc in scenario_results_l7['roa'] else 0 for sc in sc_names]
    pm_vals  = [scenario_results_l7['profit_margin'][sc][f'predicted_profit_margin'].mean()
                if 'profit_margin' in scenario_results_l7 and sc in scenario_results_l7['profit_margin'] else 0
                for sc in sc_names]
    x5 = np.arange(len(sc_names))
    w5 = 0.35
    ax5.bar(x5 - w5 / 2, roa_vals, w5, label='ROA', color='steelblue')
    ax5.bar(x5 + w5 / 2, pm_vals,  w5, label='Profit Margin', color='coral')
    ax5.set_xlabel('Scenario')
    ax5.set_ylabel('Predicted Value')
    ax5.set_title('Scenario Impacts on Profitability', fontsize=10, fontweight='bold')
    ax5.set_xticks(x5)
    ax5.set_xticklabels(sc_names, rotation=45, ha='right', fontsize=8)
    ax5.legend()
    ax5.axhline(y=roa_vals[0] if roa_vals else 0, color='steelblue', linestyle='--', alpha=0.5)

ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
n_high     = len(risk_df_l7[risk_df_l7['Risk_Score'] >= 40]) if len(risk_df_l7) > 0 else 0
n_critical = len(risk_df_l7[risk_df_l7['Risk_Score'] >= 60]) if len(risk_df_l7) > 0 else 0
top_sector = sector_risk_summary.index[0] if len(sector_risk_summary) > 0 else 'N/A'
n_lev_warn = len([w for w in warning_signals_l7 if 'HIGH LEVERAGE' in w['Warnings']])
n_liq_warn = len([w for w in warning_signals_l7 if 'LIQUIDITY CRISIS' in w['Warnings']])
summary_text = (
    f"+------------------------------------+\n"
    f"|     EXECUTIVE RISK SUMMARY         |\n"
    f"+------------------------------------+\n"
    f"|  Total Firms Analyzed: {len(risk_df_l7):3d}     |\n"
    f"|  High Risk Firms (40+): {n_high:3d}     |\n"
    f"|  Critical Risk (60+): {n_critical:3d}     |\n"
    f"|                                    |\n"
    f"|  Highest Risk Sector:              |\n"
    f"|    {top_sector[:20]:<20}      |\n"
    f"|                                    |\n"
    f"|  Warning Signals:                  |\n"
    f"|    - High Leverage: {n_lev_warn} firms      |\n"
    f"|    - Liquidity:     {n_liq_warn} firms  |\n"
    f"|                                    |\n"
    f"|  Model Performance:                |\n"
    f"|    ROA Test R2: 0.88               |\n"
    f"|    Profit Margin R2: 0.78          |\n"
    f"+------------------------------------+"
)
ax6.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center', fontfamily='monospace')
ax6.set_title('Key Metrics', fontsize=10, fontweight='bold')

plt.suptitle('RISK DASHBOARD - Executive Summary', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('layer7_executive_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: layer7_executive_dashboard.png")

# ----------------------------------------------------------------------------
# HTML REPORT
# ----------------------------------------------------------------------------
top_firm_name  = risk_ranking.iloc[0]['Company']   if len(risk_df_l7) > 0 else 'N/A'
top_firm_score = risk_ranking.iloc[0]['Risk_Score'] if len(risk_df_l7) > 0 else 0.0
top_driver_name= list(top_risk_drivers_l7.keys())[0] if top_risk_drivers_l7 is not None else 'N/A'

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Risk Dashboard - Stress Testing Framework</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .risk-critical {{ color: #c0392b; font-weight: bold; }}
        .risk-high {{ color: #e67e22; font-weight: bold; }}
        .risk-medium {{ color: #f39c12; font-weight: bold; }}
        .risk-low {{ color: #27ae60; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .warning {{ background-color: #ffeaa7; padding: 10px;
                    border-left: 4px solid #e17055; margin: 10px 0; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #7f8c8d; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Financial Stress Testing Framework</h1>
    <h2>Layer 7: Risk Dashboard</h2>
    <div class="card">
        <h2>Executive Summary</h2>
        <ul>
            <li>Total Firms Analyzed: {len(risk_df_l7)}</li>
            <li>High Risk Firms (Score >= 40): {n_high}</li>
            <li>Critical Risk Firms (Score >= 60): {n_critical}</li>
            <li>Model ROA R2: 0.88 (Test Set 2022-2024)</li>
            <li>Model Profit Margin R2: 0.78 (Test Set 2022-2024)</li>
            <li>Highest Risk Firm: {top_firm_name} (Score: {top_firm_score:.1f})</li>
            <li>Highest Risk Sector: {top_sector}</li>
            <li>Primary Risk Driver: {top_driver_name}</li>
        </ul>
    </div>
    <div class="card">
        <h2>Top 10 Highest Risk Firms</h2>
        <table>
            <tr><th>Rank</th><th>Company</th><th>Sector</th><th>Risk Score</th><th>Level</th></tr>
"""

for _, row in risk_ranking.head(10).iterrows():
    rc = ("risk-critical" if row['Risk_Score'] >= 60 else
          "risk-high"     if row['Risk_Score'] >= 40 else
          "risk-medium"   if row['Risk_Score'] >= 25 else "risk-low")
    html_content += (
        f"<tr><td>{row['Rank']}</td><td>{row['Company']}</td>"
        f"<td>{row.get('Sector','N/A')}</td>"
        f"<td class='{rc}'>{row['Risk_Score']:.1f}</td>"
        f"<td>{row['Risk_Level']}</td></tr>\n"
    )

html_content += """
        </table>
    </div>
    <div class="card">
        <h2>Critical Warning Signals</h2>
"""
if warning_signals_l7:
    for w in warning_signals_l7[:10]:
        html_content += (
            f"<div class='warning'><strong>[!] {w['Company']}</strong> "
            f"({w.get('Sector','N/A')})<br>{w['Warnings']}<br>"
            f"<small>Stagflation Drop: {w.get('Stress_Drop_%',0):.1f}%</small></div>\n"
        )
else:
    html_content += "<p>No critical warning signals detected.</p>\n"

html_content += """
    </div>
    <div class="card">
        <h2>Visualizations</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div><img src="layer7_risk_ranking.png" alt="Risk Ranking"
                 style="max-width:100%; border: 1px solid #ddd;"></div>
            <div><img src="layer7_sector_risk.png" alt="Sector Risk"
                 style="max-width:100%; border: 1px solid #ddd;"></div>
            <div><img src="layer7_risk_drivers.png" alt="Risk Drivers"
                 style="max-width:100%; border: 1px solid #ddd;"></div>
            <div><img src="layer7_executive_dashboard.png" alt="Executive Dashboard"
                 style="max-width:100%; border: 1px solid #ddd;"></div>
        </div>
    </div>
    <div class="footer">
        <p>Generated by 7-Layer Stress Testing Framework | Model: XGBoost | Target: ROA & Profit Margin</p>
        <p>Data Period: 2000-2024 | Test Period: 2022-2024</p>
    </div>
</div>
</body>
</html>
"""

with open('layer7_risk_dashboard_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("Saved: layer7_risk_dashboard_report.html")

print("\n" + "="*70)
print("LAYER 7 COMPLETE! ALL DASHBOARDS GENERATED")
print("="*70)
print(f"""
LAYER 7: RISK DASHBOARD SUMMARY
--------------------------------
DASHBOARDS GENERATED:
  - layer7_risk_ranking.png
  - layer7_sector_risk.png
  - layer7_warning_dashboard.png
  - layer7_scenario_heatmap.png
  - layer7_risk_drivers.png
  - layer7_monte_carlo_dashboard.png
  - layer7_executive_dashboard.png

DATA FILES EXPORTED:
  - layer7_risk_ranking.csv
  - layer7_sector_risk_summary.csv
  - layer7_warning_signals.csv
  - layer7_risk_dashboard_report.html

KEY FINDINGS:
  - Highest Risk Firm  : {top_firm_name} (Score: {top_firm_score:.1f})
  - Highest Risk Sector: {top_sector}
  - Primary Risk Driver: {top_driver_name}
""")

print("="*70)
print("ALL 7 LAYERS COMPLETE!")
print("  Layer 1: Feature Intelligence     -> ratio & interaction features")
print("  Layer 2: Macro Transmission       -> 3-way interactions, regime dummies, stress score")
print("  Layer 3: Regime Detection         -> K-Means & GMM regime labels")
print("  Layer 4: Core Prediction          -> XGBoost for ROA & Profit Margin")
print("  Layer 5: Scenario Simulation      -> 7 stress scenarios + Monte Carlo")
print("  Layer 6: SHAP Analysis            -> feature importance & interpretability")
print("  Layer 7: Risk Dashboard           -> executive reports & HTML output")
print("="*70)
