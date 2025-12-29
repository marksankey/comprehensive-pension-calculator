# Pension Forecast Calculator - Tax Optimization Specification

**Version**: 1.0  
**Date**: 29 December 2025  
**Author**: Mark Black  
**Purpose**: Optimize tax efficiency of pension withdrawal strategy

---

## 1. Executive Summary

This specification defines changes to the pension forecast calculator to implement tax-optimized withdrawal strategies. The primary goal is to preserve tax-advantaged ISA assets by prioritizing SIPP taxable withdrawals when within the basic rate tax band.

**Key Optimization**: Switch withdrawal priority from ISA to SIPP taxable when total taxable income remains below the higher rate threshold (£50,270).

---

## 2. Current System Analysis

### 2.1 Current Withdrawal Hierarchy
The existing calculator uses the following withdrawal priority:

1. DB pension income (fixed)
2. State pension income (starts year 4 - 2030)
3. SIPP bond income (investment returns)
4. ISA bond income (investment returns)
5. SIPP tax-free withdrawal (25% tax-free lump sum)
6. ISA withdrawal (currently used before SIPP taxable)
7. SIPP taxable withdrawal (currently unused/minimal)

### 2.2 Current Results Summary
Based on the provided projection (`2025-12-29T16-08_export.csv`):

- **Projection period**: 25 years (2027-2051)
- **SIPP taxable pot**: Remains largely untouched (£424k-£459k)
- **ISA drawdown**: Gradual depletion (£92k down to £99k)
- **Effective tax rates**: 10.45% - 16.56%
- **SIPP taxable withdrawals**: £0 throughout entire projection
- **Tax-free cash usage**: Minimal in early years (£72, £2,927, £4,570)

### 2.3 Key Problem Identified
The calculator is drawing from ISA (tax-free) while leaving SIPP taxable pot untouched, despite having unused basic rate tax band capacity. This is suboptimal because:

- ISA withdrawals are 100% tax-free and should be preserved
- SIPP taxable withdrawals at basic rate (20%) are acceptable when below £50,270 total income
- ISA has superior inheritance tax treatment (IHT-free vs SIPP taxed at beneficiary's rate)

---

## 3. Tax Framework (UK Tax Year Assumptions)

### 3.1 Tax Rates and Thresholds
```
Personal Allowance: £12,570
Basic Rate Band: £12,571 - £50,270 (20% tax)
Higher Rate Band: £50,271 - £125,140 (40% tax)
Personal Allowance Taper: Reduces £1 per £2 income over £100,000
```

### 3.2 Income Sources Tax Treatment
| Source | Tax Status |
|--------|------------|
| DB Pension | Fully taxable |
| State Pension | Fully taxable |
| SIPP Bond Income | Fully taxable |
| ISA Bond Income | Tax-free (but counted in projection as income) |
| SIPP Tax-Free Withdrawal | Tax-free (25% of pot) |
| ISA Withdrawal | Tax-free |
| SIPP Taxable Withdrawal | Fully taxable |

---

## 4. Optimization Requirements

### 4.1 Primary Optimization: SIPP vs ISA Withdrawal Priority

**Requirement OPT-001**: Modify withdrawal logic to prioritize SIPP taxable over ISA

**Current Logic**:
```
IF shortfall exists:
    1. Try ISA withdrawal
    2. Then SIPP taxable withdrawal
```

**New Logic**:
```
IF shortfall exists:
    Calculate: available_basic_rate_capacity = £50,270 - current_taxable_income
    
    IF available_basic_rate_capacity > 0:
        Use SIPP taxable withdrawal (up to capacity or shortfall, whichever is lower)
    
    IF shortfall still exists:
        Use ISA withdrawal
    
    IF shortfall still exists:
        Use SIPP taxable withdrawal (at higher rate - unavoidable)
```

**Decision Tree**:
```
Is there a shortfall after all fixed income?
    └─ YES
        └─ Is current taxable income < £50,270?
            └─ YES
                └─ Available basic rate capacity = £50,270 - current_taxable_income
                └─ Draw min(shortfall, capacity, remaining_sipp_taxable) from SIPP
                └─ Update shortfall
            └─ NO
                └─ Skip to ISA
        
        └─ Is shortfall still > 0?
            └─ YES
                └─ Draw min(shortfall, remaining_isa) from ISA
                └─ Update shortfall
        
        └─ Is shortfall still > 0?
            └─ YES
                └─ Draw from SIPP taxable (higher rate taxation unavoidable)
```

### 4.2 Secondary Optimization: Front-Load Tax-Free Cash

**Requirement OPT-002**: Increase SIPP tax-free withdrawals in pre-state-pension years

**Rationale**: Maximize time for tax-free money to grow outside pension wrapper

**Implementation**:
- **Years 1-3 (2027-2029)**: Increase tax-free withdrawal target
  - Current: £72, £2,927, £4,570
  - Proposed: £15,000 - £20,000 per year (configurable parameter)
  
**Logic Change**:
```python
# Add parameter
FRONT_LOAD_TAX_FREE_YEARS = 3  # Years before state pension
FRONT_LOAD_TAX_FREE_AMOUNT = 18000  # Annual target for early years

# In withdrawal calculation
if year <= FRONT_LOAD_TAX_FREE_YEARS:
    if remaining_sipp_tax_free > 0:
        tax_free_withdrawal = min(
            FRONT_LOAD_TAX_FREE_AMOUNT,
            remaining_sipp_tax_free,
            shortfall  # Don't over-withdraw
        )
```

### 4.3 Warning System

**Requirement OPT-003**: Add warnings for tax efficiency concerns

Implement warning flags when:

1. **Higher Rate Breach Warning**
   ```
   IF total_taxable_income > £50,270:
       FLAG: "Higher rate tax (40%) applicable on income over £50,270"
   ```

2. **£100k Cliff Edge Warning**
   ```
   IF total_gross_income > £100,000:
       personal_allowance_reduction = (total_gross_income - 100000) / 2
       effective_rate_on_band = 60%
       FLAG: "Personal allowance taper active - 60% effective rate on £100k-£125k"
   ```

3. **ISA vs SIPP Suboptimality Warning**
   ```
   IF isa_withdrawal > 0 AND sipp_taxable_withdrawal == 0 AND total_taxable_income < £50,000:
       FLAG: "Tax efficiency warning: ISA used while basic rate capacity available"
   ```

---

## 5. Implementation Specifications

### 5.1 Configuration Parameters

Add the following configurable parameters to the calculator:

```python
# Tax optimization parameters
ENABLE_SIPP_PRIORITY = True  # Feature flag to enable/disable optimization
BASIC_RATE_THRESHOLD = 50270  # Upper limit of basic rate band
PERSONAL_ALLOWANCE = 12570
HIGHER_RATE_TAPER_THRESHOLD = 100000

# Front-loading parameters
ENABLE_FRONT_LOAD_TAX_FREE = True
FRONT_LOAD_TAX_FREE_YEARS = 3
FRONT_LOAD_TAX_FREE_AMOUNT = 18000

# Warning thresholds
WARN_HIGHER_RATE = True
WARN_TAPER = True
WARN_SUBOPTIMAL_ISA = True
```

### 5.2 Modified Calculation Flow

**Current annual calculation flow**:
1. Calculate fixed income (DB, State, Bond income)
2. Calculate total taxable income
3. Calculate tax
4. Calculate shortfall
5. Fill shortfall (tax-free, ISA, SIPP taxable)
6. Recalculate tax
7. Output results

**Modified annual calculation flow**:
1. Calculate fixed income (DB, State, Bond income)
2. Calculate total taxable income (pre-withdrawal)
3. **NEW: Calculate available basic rate capacity**
4. Calculate initial tax
5. Calculate shortfall
6. **NEW: Enhanced front-loading of tax-free (years 1-3)**
7. **MODIFIED: Fill shortfall with priority logic (SIPP taxable, then ISA)**
8. Recalculate tax
9. **NEW: Generate warnings**
10. Output results

### 5.3 Data Structure Changes

**Add new columns to output**:
```python
# New output columns
'available_basic_rate_capacity'  # £50,270 - total_taxable_income
'sipp_priority_withdrawal'       # Amount taken via optimization
'isa_protected_value'            # ISA value saved vs old method
'tax_efficiency_flag'            # Warning flag if suboptimal
'optimization_notes'             # Text notes about year's optimization
```

### 5.4 Testing Requirements

**Test Scenarios**:

1. **Scenario 1: Pre-State Pension Years (2027-2029)**
   - Input: Current income profile, £141k SIPP tax-free available
   - Expected: £15k-£20k annual tax-free withdrawal
   - Expected: SIPP taxable used if shortfall, ISA preserved

2. **Scenario 2: Basic Rate Capacity Available**
   - Input: Total taxable income = £40,000, shortfall = £5,000
   - Expected: £5,000 drawn from SIPP taxable (not ISA)
   - Expected: Basic rate capacity = £10,270

3. **Scenario 3: Near Higher Rate Threshold**
   - Input: Total taxable income = £48,000, shortfall = £5,000
   - Expected: £2,270 from SIPP taxable, £2,730 from ISA
   - Expected: Warning about approaching higher rate

4. **Scenario 4: No Basic Rate Capacity**
   - Input: Total taxable income = £51,000, shortfall = £5,000
   - Expected: ISA used first, SIPP taxable only if ISA depleted

5. **Scenario 5: ISA Depletion**
   - Input: ISA balance = £1,000, shortfall = £5,000
   - Expected: £1,000 from ISA, £4,000 from SIPP taxable (at higher rate if needed)

### 5.5 Comparison Output

**Requirement OPT-004**: Generate comparison report

Create a summary comparing old vs new strategy:

```python
comparison_metrics = {
    'total_tax_old': sum of tax under current strategy,
    'total_tax_new': sum of tax under optimized strategy,
    'tax_saved': difference,
    'final_isa_balance_old': final ISA under current,
    'final_isa_balance_new': final ISA under optimized,
    'isa_preserved': difference,
    'years_in_higher_rate_old': count,
    'years_in_higher_rate_new': count,
    'optimization_notes': key differences
}
```

---

## 6. Algorithm Pseudocode

### 6.1 Main Optimization Function

```python
def calculate_optimized_withdrawal(year_data, config):
    """
    Calculate tax-optimized withdrawals for a single year
    
    Args:
        year_data: dict with income, remaining pots, target
        config: dict with optimization parameters
    
    Returns:
        dict with withdrawal amounts and updated balances
    """
    
    # Step 1: Calculate fixed income
    fixed_income = (
        year_data['db_pension'] + 
        year_data['state_pension'] + 
        year_data['sipp_bond_income'] + 
        year_data['isa_bond_income']
    )
    
    # Step 2: Calculate shortfall
    shortfall = year_data['target_income'] - fixed_income
    
    if shortfall <= 0:
        return {
            'sipp_tax_free': 0,
            'sipp_taxable': 0,
            'isa_withdrawal': 0,
            'tax': calculate_tax(fixed_income, config)
        }
    
    # Step 3: Front-load tax-free cash (if enabled)
    sipp_tax_free = 0
    if config['ENABLE_FRONT_LOAD_TAX_FREE'] and year_data['year'] <= config['FRONT_LOAD_TAX_FREE_YEARS']:
        if year_data['remaining_sipp_tax_free'] > 0:
            sipp_tax_free = min(
                config['FRONT_LOAD_TAX_FREE_AMOUNT'],
                year_data['remaining_sipp_tax_free'],
                shortfall
            )
            shortfall -= sipp_tax_free
    
    # If still have shortfall, take remaining from minimal tax-free usage
    if shortfall > 0 and year_data['remaining_sipp_tax_free'] > 0:
        additional_tax_free = min(shortfall, year_data['remaining_sipp_tax_free'])
        sipp_tax_free += additional_tax_free
        shortfall -= additional_tax_free
    
    # Step 4: Calculate current taxable income
    current_taxable_income = (
        year_data['db_pension'] + 
        year_data['state_pension'] + 
        year_data['sipp_bond_income']
    )
    
    # Step 5: Calculate available basic rate capacity
    basic_rate_capacity = max(0, config['BASIC_RATE_THRESHOLD'] - current_taxable_income)
    
    # Step 6: Optimized withdrawal logic
    sipp_taxable = 0
    isa_withdrawal = 0
    
    if shortfall > 0 and config['ENABLE_SIPP_PRIORITY']:
        # Try SIPP taxable first (if capacity and funds available)
        if basic_rate_capacity > 0 and year_data['remaining_sipp_taxable'] > 0:
            sipp_taxable = min(
                shortfall,
                basic_rate_capacity,
                year_data['remaining_sipp_taxable']
            )
            shortfall -= sipp_taxable
            current_taxable_income += sipp_taxable
        
        # Then use ISA if shortfall remains
        if shortfall > 0 and year_data['remaining_isa'] > 0:
            isa_withdrawal = min(shortfall, year_data['remaining_isa'])
            shortfall -= isa_withdrawal
        
        # Finally use SIPP taxable at higher rate if still needed
        if shortfall > 0 and year_data['remaining_sipp_taxable'] > 0:
            additional_sipp = min(shortfall, year_data['remaining_sipp_taxable'])
            sipp_taxable += additional_sipp
            shortfall -= additional_sipp
            current_taxable_income += additional_sipp
    
    else:
        # Old logic: ISA first, then SIPP taxable
        if shortfall > 0 and year_data['remaining_isa'] > 0:
            isa_withdrawal = min(shortfall, year_data['remaining_isa'])
            shortfall -= isa_withdrawal
        
        if shortfall > 0 and year_data['remaining_sipp_taxable'] > 0:
            sipp_taxable = min(shortfall, year_data['remaining_sipp_taxable'])
            shortfall -= sipp_taxable
            current_taxable_income += sipp_taxable
    
    # Step 7: Calculate tax
    total_taxable_income = current_taxable_income
    tax = calculate_tax(total_taxable_income, config)
    
    # Step 8: Generate warnings
    warnings = generate_warnings(
        year_data['year'],
        total_taxable_income,
        sipp_taxable,
        isa_withdrawal,
        basic_rate_capacity,
        config
    )
    
    return {
        'sipp_tax_free': sipp_tax_free,
        'sipp_taxable': sipp_taxable,
        'isa_withdrawal': isa_withdrawal,
        'total_taxable_income': total_taxable_income,
        'tax': tax,
        'basic_rate_capacity': basic_rate_capacity,
        'warnings': warnings,
        'shortfall_remaining': shortfall
    }


def calculate_tax(taxable_income, config):
    """
    Calculate UK income tax including personal allowance taper
    """
    gross_income = taxable_income
    
    # Personal allowance taper for income over £100k
    personal_allowance = config['PERSONAL_ALLOWANCE']
    if gross_income > config['HIGHER_RATE_TAPER_THRESHOLD']:
        reduction = (gross_income - config['HIGHER_RATE_TAPER_THRESHOLD']) / 2
        personal_allowance = max(0, personal_allowance - reduction)
    
    taxable = max(0, taxable_income - personal_allowance)
    
    # Calculate tax
    tax = 0
    
    # Basic rate: 20% on income up to £50,270
    basic_rate_limit = config['BASIC_RATE_THRESHOLD']
    if taxable <= basic_rate_limit:
        tax = taxable * 0.20
    else:
        # Higher rate: 40% on income over £50,270
        tax = (basic_rate_limit * 0.20) + ((taxable - basic_rate_limit) * 0.40)
    
    return tax


def generate_warnings(year, taxable_income, sipp_withdrawal, isa_withdrawal, 
                     basic_capacity, config):
    """
    Generate warning messages for tax efficiency concerns
    """
    warnings = []
    
    # Higher rate warning
    if config['WARN_HIGHER_RATE'] and taxable_income > config['BASIC_RATE_THRESHOLD']:
        excess = taxable_income - config['BASIC_RATE_THRESHOLD']
        warnings.append(f"Higher rate tax (40%) on £{excess:,.0f} above £50,270")
    
    # Personal allowance taper warning
    if config['WARN_TAPER'] and taxable_income > config['HIGHER_RATE_TAPER_THRESHOLD']:
        warnings.append("Personal allowance taper active - 60% effective rate on £100k-£125k income")
    
    # Suboptimal ISA usage warning
    if config['WARN_SUBOPTIMAL_ISA'] and isa_withdrawal > 0 and sipp_withdrawal == 0:
        if basic_capacity > 1000:  # Only warn if significant capacity wasted
            warnings.append(f"ISA used with £{basic_capacity:,.0f} basic rate capacity available")
    
    return warnings
```

---

## 7. Output Specifications

### 7.1 Enhanced CSV Output

Add the following columns to the existing CSV export:

```
existing columns...,
basic_rate_capacity,
sipp_priority_withdrawal_amount,
tax_efficiency_warnings,
optimization_notes
```

### 7.2 Summary Report

Generate a text summary report:

```
====================================
PENSION OPTIMIZATION SUMMARY REPORT
====================================

Strategy Comparison (2027-2051)
--------------------------------
Old Strategy:
  - Total tax paid: £XXX,XXX
  - Final ISA balance: £XX,XXX
  - Years in higher rate band: XX
  - SIPP taxable used: £XXX,XXX

Optimized Strategy:
  - Total tax paid: £XXX,XXX (saved: £X,XXX)
  - Final ISA balance: £XXX,XXX (preserved: £XX,XXX)
  - Years in higher rate band: XX
  - SIPP taxable used: £XXX,XXX

Key Optimizations Applied:
--------------------------
1. SIPP taxable prioritized over ISA in XX years
2. Tax-free cash front-loaded in years 1-3
3. Basic rate capacity maximized in XX years

Tax Efficiency Warnings:
------------------------
[List any years with warnings]

Recommendations:
----------------
[Strategic notes based on results]
```

---

## 8. Implementation Checklist

- [ ] Add configuration parameters to settings/config file
- [ ] Implement `calculate_optimized_withdrawal()` function
- [ ] Modify main calculation loop to use new withdrawal logic
- [ ] Implement `calculate_tax()` with taper logic
- [ ] Implement `generate_warnings()` function
- [ ] Add new output columns to CSV export
- [ ] Create comparison report generator
- [ ] Implement test scenarios 1-5
- [ ] Update user interface to show optimization options
- [ ] Add documentation for new features
- [ ] Create example outputs showing before/after
- [ ] Validate against current projection data

---

## 9. Acceptance Criteria

The optimization is considered complete when:

1. **Functional Requirements Met**:
   - SIPP taxable withdrawals prioritized when basic rate capacity available
   - ISA preserved when basic rate withdrawal possible
   - Tax-free cash front-loaded in years 1-3 (configurable)
   - All warnings generated correctly

2. **Output Requirements Met**:
   - CSV includes new optimization columns
   - Summary report generated with comparison metrics
   - Warnings clearly flagged in output

3. **Testing Requirements Met**:
   - All 5 test scenarios pass
   - Comparison with original projection shows expected differences
   - No regressions in tax calculations

4. **Performance Requirements Met**:
   - Calculation time remains under 5 seconds for 25-year projection
   - Memory usage acceptable

5. **Documentation Requirements Met**:
   - Configuration parameters documented
   - User guide updated with optimization explanation
   - Example outputs provided

---

## 10. Future Enhancements

Potential future optimizations to consider:

1. **Roth-style SIPP Conversion Optimization**
   - Identify years to convert SIPP taxable to tax-free
   - Analyze optimal conversion amounts within basic rate band

2. **State Pension Deferral Calculator**
   - Calculate optimal deferral period (9 weeks = 1% increase)
   - Compare total lifetime income with/without deferral

3. **Inheritance Tax Planning**
   - Model IHT implications of different strategies
   - Optimize for beneficiary tax position

4. **Dynamic Income Targeting**
   - Adjust target income based on tax band optimization
   - Suggest lower income in high-tax years

5. **Investment Return Optimization**
   - Model different return assumptions for SIPP vs ISA
   - Optimize withdrawal sequence based on growth rates

---

## 11. References

- UK Tax Rates 2025/26: https://www.gov.uk/income-tax-rates
- SIPP Rules: https://www.moneyhelper.org.uk/en/pensions-and-retirement/
- ISA Regulations: https://www.gov.uk/individual-savings-accounts

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-29 | M. Black | Initial specification |

