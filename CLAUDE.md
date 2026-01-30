# CLAUDE.md - Comprehensive Pension Calculator

## Quick Reference

```bash
# Run the application
streamlit run app.py

# Run tests
python test_bond_fetcher.py

# Install dependencies
pip install -r requirements.txt
```

## Project Overview

A sophisticated UK retirement planning tool that combines bond ladder strategies with pension drawdown analysis. Built with Streamlit, it provides comprehensive tax calculations and portfolio sustainability analysis for UK retirees.

## Tech Stack

- **Framework**: Streamlit 1.28+ (web interface)
- **Language**: Python 3.11+
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly
- **Financial Data**: yfinance, Bank of England API
- **Excel Export**: openpyxl
- **Tax Rules**: UK 2025/26 tax year

## Project Structure

```
comprehensive-pension-calculator/
├── app.py                    # Main Streamlit application
├── appnew.py                 # Alternative/updated version (review for differences)
├── bond_data_fetcher.py      # Bank of England API integration
├── excel_enhanced_export.py  # Enhanced Excel export functionality
├── test_bond_fetcher.py      # Tests for bond data fetcher
├── requirements.txt          # Python dependencies
├── build.sh                  # Render deployment build script
├── render.yaml               # Render deployment config
├── .streamlit/
│   └── config.toml           # Streamlit theme and server config
└── .cache/                   # Bond data cache (gitignored)
```

## Key Files

### app.py (Main Application ~3000+ lines)

The primary application file containing:
- `EnhancedSIPPBondCalculator` class - Core calculation engine
- UK tax bands for 2025/26
- Bond database (UK Gilts and Corporate Bonds with ISIN codes)
- SIPP and ISA portfolio management
- Pension drawdown calculations
- Tax optimization logic
- Interactive Plotly visualizations

### bond_data_fetcher.py

Live market data integration:
- `BondDataFetcher` class for Bank of England API
- 24-hour intelligent caching system
- Yield curve interpolation
- Graceful fallback to default estimates

### excel_enhanced_export.py

Professional Excel report generation with:
- Multiple worksheets for different data views
- Formatted tables and charts
- Summary statistics

## Core Functionality

### 1. Bond Ladder Management
- SIPP (Self-Invested Personal Pension) bond portfolios
- ISA (Individual Savings Account) bond portfolios
- Automatic reinvestment strategies
- UK Gilts and Corporate Bond databases with ISIN codes
- Live yield data from Bank of England API

### 2. Pension Integration
- Defined benefit pensions
- State pension calculations
- Pension drawdown analysis
- 25% tax-free lump sum handling

### 3. Tax Calculations (2025/26)
```python
personal_allowance = 12570
basic_rate_threshold = 50270
higher_rate_threshold = 125140
additional_rate_threshold = 150000

# Tax rates
basic_rate = 20%
higher_rate = 40%
additional_rate = 45%

# Personal allowance tapers £1 for every £2 over £100,000
```

### 4. Analysis Features
- Year-by-year income breakdown
- Inflation-adjusted projections
- Portfolio sustainability tracking
- Tax burden analysis
- Visual analytics with interactive Plotly charts

## Bond Database Structure

Each bond entry includes:
```python
{
    'isin': 'GB00B16NNR78',           # International Securities ID
    'maturity_date': '2027-12-07',     # When the bond matures
    'coupon': 4.25,                    # Annual interest rate (%)
    'type': 'UK Gilt',                 # UK Gilt or Corporate Bond
    'rating': 'AA',                    # Credit rating
    'min_denomination': 100,           # Minimum purchase amount (£)
    'recommended_for': 'SIPP',         # SIPP or ISA
    'liquidity_tier': 1,               # 1=highest liquidity
    'min_ytm': 3.8                     # Minimum yield to maturity (%)
}
```

## Bond Data Integration

### Bank of England API

**Endpoint**: `http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes`

**Series Codes** (Nominal Spot Yields):
- `IUDMNPY` - 1 year
- `IUDMOPY` - 2 year
- `IUDMPPY` - 3 year
- `IUDMQPY` - 5 year
- `IUDMRPY` - 7 year
- `IUDMSOY` - 10 year
- `IUDMTOY` - 15 year
- `IUDMVPY` - 20 year
- `IUDMWPY` - 25 year
- `IUDBEDR` - 30 year

**Caching**: Data cached in `.cache/gilt_data_cache.pkl` for 24 hours

**Fallback**: Uses conservative default yield curve if API unavailable

## Common Development Tasks

### Adding New Bonds
Update the bond database in `EnhancedSIPPBondCalculator.__init__()` in app.py with proper ISIN codes and bond details.

### Modifying Tax Rules
Tax bands are defined in the `EnhancedSIPPBondCalculator` class initialization. Update for new tax years as needed.

### Testing Changes
- Test with various portfolio sizes
- Verify tax calculations across all bands
- Check bond ladder generation logic (ensure no duplicate bonds)
- Validate that ladder creates evenly-spaced maturity dates
- Verify each bond in the ladder has a unique ISIN
- Validate inflation adjustments
- Run `python test_bond_fetcher.py` to verify API integration

## Deployment

### Streamlit Cloud
Configuration in `.streamlit/config.toml`:
- Custom green theme
- Headless server mode
- Usage stats disabled

### Render
Configuration in `render.yaml`:
- Python 3.11 runtime
- Auto-installs dependencies
- Runs Streamlit on dynamic port

## Important Considerations

### Financial Accuracy
- All calculations must follow UK tax rules precisely
- Bond yields and prices should be realistic
- Tax-free allowances must be calculated correctly
- Personal allowance tapering applies above £100,000

### User Safety
- This tool is for educational purposes only
- Users should seek professional financial advice
- Disclaimers must remain prominent
- Market conditions and tax rules can change

### Data Validation
- Validate all user inputs
- Handle edge cases (zero income, very high values)
- Check for negative values where inappropriate
- Ensure date logic is correct

## Recent Changes

### Live Bond Data Integration (2025-12-31)
- Integrated Bank of England IADB API for real-time UK gilt yields
- 24-hour caching system for performance
- Graceful fallback to default estimates
- Data freshness indicator in UI

### Bond Ladder Duplicate Selection Fix (2025-12-31)
- Fixed issue where same bonds were selected multiple times
- Added `selected_isins` tracking set
- Ensures evenly-spaced maturity schedules with unique bonds

## Future Enhancement Areas

- Additional bond types or investment vehicles
- More sophisticated tax optimization
- Scenario comparison tools
- Monte Carlo simulation for market volatility
- Enhanced corporate bond spreads data
- Support for couples and inheritance planning
- Integration with DMO data for individual gilt inventory

## Notes for Claude

When working on this codebase:
1. Always maintain financial calculation accuracy
2. Test tax calculations thoroughly across all bands
3. Preserve the educational disclaimer
4. Keep bond data realistic and up-to-date
5. Ensure all outputs are clearly labeled
6. Consider UK-specific financial regulations
7. Document any changes to calculation logic
8. Ensure bond ladder selections are unique (no duplicate ISINs)
9. The BoE API may return 403 in sandboxed environments - fallback is expected
10. Use logging (`logger.info/warning/error`) for debugging
