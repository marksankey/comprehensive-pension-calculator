# Claude.md - Comprehensive Pension Calculator

## Project Overview

This is a sophisticated UK retirement planning tool that combines bond ladder strategies with pension drawdown analysis. Built with Streamlit, it provides comprehensive tax calculations and portfolio sustainability analysis for UK retirees.

## Tech Stack

- **Framework**: Streamlit (web interface)
- **Language**: Python 3.11+
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly
- **Financial Data**: yfinance
- **Tax Rules**: UK 2025/26 tax year

## Key Files

### app.py (Main Application)
The primary application file containing:
- `EnhancedSIPPBondCalculator` class - Core calculation engine
- UK tax bands for 2025/26
- Bond database (UK Gilts and Corporate Bonds)
- SIPP and ISA portfolio management
- Pension drawdown calculations
- Tax optimization logic

### appnew.py
Alternative or updated version of the main application (review for differences)

### excel_enhanced_export.py
Enhanced Excel export functionality for detailed financial reports

### requirements.txt
Python dependencies for the project

## Core Functionality

### 1. Bond Ladder Management
- SIPP (Self-Invested Personal Pension) bond portfolios
- ISA (Individual Savings Account) bond portfolios
- Automatic reinvestment strategies
- UK Gilts and Corporate Bond databases with ISIN codes

### 2. Pension Integration
- Defined benefit pensions
- State pension calculations
- Pension drawdown analysis
- 25% tax-free lump sum handling

### 3. Tax Calculations
- Personal allowance: £12,570
- Basic rate threshold: £50,270
- Higher rate threshold: £125,140
- Additional rate threshold: £150,000
- Personal allowance tapering
- Multiple tax bands (basic 20%, higher 40%, additional 45%)

### 4. Analysis Features
- Year-by-year income breakdown
- Inflation-adjusted projections
- Portfolio sustainability tracking
- Tax burden analysis
- Visual analytics with interactive charts

## Key Constants (2025/26 Tax Year)

```python
personal_allowance = 12570
basic_rate_threshold = 50270
higher_rate_threshold = 125140
additional_rate_threshold = 150000
```

## Bond Database Structure

Each bond entry includes:
- `isin`: International Securities Identification Number
- `maturity_date`: When the bond matures
- `coupon`: Annual interest rate
- `type`: UK Gilt or Corporate Bond
- `rating`: Credit rating (AA, AAA, etc.)
- `min_denomination`: Minimum purchase amount
- `recommended_for`: SIPP or ISA
- `liquidity_tier`: Liquidity ranking
- `min_ytm`: Minimum yield to maturity

## Common Development Tasks

### Running the Application
```bash
streamlit run app.py
```

### Adding New Bonds
Update the bond database in the `EnhancedSIPPBondCalculator.__init__()` method with proper ISIN codes and bond details.

### Modifying Tax Rules
Tax bands are defined in the calculator class initialization. Update for new tax years as needed.

### Testing Changes
- Test with various portfolio sizes
- Verify tax calculations across all bands
- Check bond ladder generation logic (ensure no duplicate bonds)
- Validate that ladder creates evenly-spaced maturity dates
- Verify each bond in the ladder has a unique ISIN
- Validate inflation adjustments

## Important Considerations

### Financial Accuracy
- All calculations should follow UK tax rules precisely
- Bond yields and prices should be realistic
- Tax-free allowances must be calculated correctly
- Personal allowance tapering applies above £100,000

### User Safety
- This tool is for educational purposes only
- Users should seek professional financial advice
- Disclaimers should remain prominent
- Market conditions and tax rules can change

### Data Validation
- Validate all user inputs
- Handle edge cases (zero income, very high values)
- Check for negative values where inappropriate
- Ensure date logic is correct

## Output Formats

The application provides:
- Interactive web interface
- Year-by-year breakdown tables
- Plotly interactive charts
- CSV export capabilities
- Enhanced Excel exports (via excel_enhanced_export.py)

## Deployment

The application is designed for Streamlit Cloud deployment. Configuration is in `.streamlit/` directory.

## Recent Changes

### Live Bond Data Integration (2025-12-31)
**Feature**: Integrated free API data sources to fetch real-time UK gilt yields and prices.

**Implementation**:
- Created `bond_data_fetcher.py` module for API integration
- Bank of England IADB API for yield curve data
- Intelligent caching system (24-hour refresh)
- Graceful fallback to default estimates if API unavailable
- Live yield interpolation for specific bond maturities
- Accurate price calculations from market yields

**Benefits**:
- Bond recommendations now use actual market data instead of hardcoded estimates
- Yields reflect current market conditions (previously could be 0.8+ percentage points off)
- Daily automatic updates via caching
- Data freshness indicator in UI shows last update time
- Zero cost - uses only free API sources

**Files Added**:
- `bond_data_fetcher.py` - API integration and caching module
- `test_bond_fetcher.py` - Test script for verification

**Files Modified**:
- `app.py` - Integrated BondDataFetcher into EnhancedSIPPBondCalculator
- `.gitignore` - Added cache directory exclusion
- `requirements.txt` - Dependencies already present (requests, pandas)

**Usage**: Data automatically refreshes on calculator initialization. Falls back to estimates if API unavailable.

### Bond Ladder Duplicate Selection Fix (2025-12-31)
**Issue**: The bond ladder was selecting the same bonds multiple times, resulting in duplicate bonds maturing in the same year instead of creating an evenly-spaced ladder structure.

**Root Cause**: The `get_bond_recommendations()` function used ±1 year flexibility when matching bonds to target years, but didn't track which bonds had already been selected. This allowed the same bond to be chosen for multiple ladder years.

**Solution**:
- Added `selected_isins` set to track already-selected bonds by ISIN
- Added skip logic to prevent duplicate bond selection
- Updated all bond selection paths to mark bonds as selected
- Made fallback bond ISINs unique per year

**Impact**: Bond ladders now properly create evenly-spaced maturity schedules with different bonds for each year, ensuring the ladder strategy works as intended.

**Files Modified**: `app.py` (lines 413, 432-434, 508, 513, 516, 530, 548)

## Future Enhancement Areas

- Additional bond types or investment vehicles
- More sophisticated tax optimization
- Scenario comparison tools
- Monte Carlo simulation for market volatility
- Enhanced corporate bond spreads data (currently uses gilt curve)
- Support for couples and inheritance planning
- Integration with DMO data for individual gilt inventory

## Bond Data Integration

### BondDataFetcher Module (`bond_data_fetcher.py`)

The application now integrates live market data through the BondDataFetcher class:

**Data Sources**:
- **Bank of England IADB API**: Provides UK gilt yield curves (nominal spot yields for 1, 2, 3, 5, 7, 10, 15, 20, 25, 30 years)
- **Fallback Estimates**: When API unavailable, uses conservative default yield curve

**Key Features**:
1. **Automatic Updates**: Fetches fresh data on calculator initialization
2. **Intelligent Caching**: Stores data for 24 hours in `.cache/` directory
3. **Yield Interpolation**: Linear interpolation for bonds with non-standard maturities
4. **Price Calculation**: Computes fair value prices from yields using present value formula
5. **Graceful Degradation**: Falls back to default estimates if API fails

**API Details**:
- **Endpoint**: `http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes`
- **Method**: HTTP GET with query parameters
- **Format**: CSV response
- **Cost**: Free (non-commercial use)
- **Rate Limit**: None specified, but cached to minimize requests

**Methods**:
- `fetch_boe_yield_curve()` - Retrieves yield curve from BoE API
- `interpolate_yield(years_to_maturity)` - Returns interpolated yield for any maturity
- `update_gilt_prices(gilt_database)` - Updates bond database with current market data
- `get_data_status()` - Returns cache status and freshness

**Testing**:
Run `python test_bond_fetcher.py` to verify functionality.

**Note**: In some network environments (e.g., sandboxed), the BoE API may return 403 Forbidden. The system will automatically use fallback estimates.

## Notes for Claude

When working on this codebase:
1. Always maintain financial calculation accuracy
2. Test tax calculations thoroughly across all bands
3. Preserve the educational disclaimer
4. Keep bond data realistic and up-to-date
5. Ensure all outputs are clearly labeled
6. Consider UK-specific financial regulations
7. Document any changes to calculation logic
