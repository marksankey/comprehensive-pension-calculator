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
- Check bond ladder generation logic
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

## Future Enhancement Areas

- Additional bond types or investment vehicles
- More sophisticated tax optimization
- Scenario comparison tools
- Monte Carlo simulation for market volatility
- Integration with live bond price feeds
- Support for couples and inheritance planning

## Notes for Claude

When working on this codebase:
1. Always maintain financial calculation accuracy
2. Test tax calculations thoroughly across all bands
3. Preserve the educational disclaimer
4. Keep bond data realistic and up-to-date
5. Ensure all outputs are clearly labeled
6. Consider UK-specific financial regulations
7. Document any changes to calculation logic
