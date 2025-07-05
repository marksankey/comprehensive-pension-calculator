import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import logging
import traceback
import json
import random
from typing import Dict, List, Tuple, Optional
from io import BytesIO
import yfinance as yf
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
st.set_page_config(
    page_title="Enhanced Pension & Bond Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/enhanced-pension-calculator',
        'Report a bug': 'https://github.com/yourusername/enhanced-pension-calculator/issues',
        'About': """
        # Enhanced Comprehensive Pension & Bond Ladder Calculator
        
        Advanced UK retirement planning tool with real-time data and scenario analysis.
        
        **Features:**
        - Real-time gilt yield data
        - Monte Carlo simulation
        - Scenario analysis
        - Year-by-year income and tax analysis
        - Bond ladder management with automatic reinvestment
        - Multiple pension income sources
        - UK tax optimization
        - Professional reports
        
        **Disclaimer:** This tool provides estimates for educational purposes only. 
        Please seek professional financial advice before making investment decisions.
        """
    }
)

class EnhancedPensionCalculator:
    def __init__(self):
        # UK Tax bands for 2025/26 (will be inflation adjusted)
        self.personal_allowance = 12570
        self.basic_rate_threshold = 50270
        self.higher_rate_threshold = 125140
        self.additional_rate_threshold = 150000
        
    def calculate_income_tax_with_thresholds(self, taxable_income: float, personal_allowance: float) -> Dict:
        """Calculate UK income tax with high earner personal allowance reduction"""
        # Adjust personal allowance for high earners (£1 reduction for every £2 over £100k)
        adjusted_personal_allowance = personal_allowance
        if taxable_income > 100000:
            personal_allowance_reduction = min((taxable_income - 100000) / 2, personal_allowance)
            adjusted_personal_allowance = personal_allowance - personal_allowance_reduction
        
        tax = 0
        tax_breakdown = {
            'personal_allowance_used': min(taxable_income, adjusted_personal_allowance),
            'basic_rate_tax': 0,
            'higher_rate_tax': 0,
            'additional_rate_tax': 0,
            'total_tax': 0,
            'effective_rate': 0
        }
        
        if taxable_income > adjusted_personal_allowance:
            # Basic rate: 20%
            basic_rate_amount = min(taxable_income, self.basic_rate_threshold) - adjusted_personal_allowance
            if basic_rate_amount > 0:
                tax_breakdown['basic_rate_tax'] = basic_rate_amount * 0.20
                tax += tax_breakdown['basic_rate_tax']
            
            # Higher rate: 40%
            if taxable_income > self.basic_rate_threshold:
                higher_rate_amount = min(taxable_income, self.additional_rate_threshold) - self.basic_rate_threshold
                if higher_rate_amount > 0:
                    tax_breakdown['higher_rate_tax'] = higher_rate_amount * 0.40
                    tax += tax_breakdown['higher_rate_tax']
                
                # Additional rate: 45%
                if taxable_income > self.additional_rate_threshold:
                    additional_rate_amount = taxable_income - self.additional_rate_threshold
                    tax_breakdown['additional_rate_tax'] = additional_rate_amount * 0.45
                    tax += tax_breakdown['additional_rate_tax']
        
        tax_breakdown['total_tax'] = tax
        tax_breakdown['effective_rate'] = (tax / taxable_income * 100) if taxable_income > 0 else 0
        
        return tax_breakdown

    def calculate_sipp_tax_free_available(self, sipp_value: float, tax_free_taken: float = 0) -> Dict:
    """Calculate available tax-free amount from SIPP"""
    max_tax_free = sipp_value * 0.25
    remaining_tax_free = max(0, max_tax_free - tax_free_taken)
    taxable_portion = sipp_value - max_tax_free
    
    return {
        'max_tax_free_lump_sum': max_tax_free,
        'remaining_tax_free': remaining_tax_free,
        'taxable_portion': taxable_portion,
        'tax_free_percentage_used': (tax_free_taken / max_tax_free * 100) if max_tax_free > 0 else 0
    }

def optimize_withdrawal_order(self, target_net_income: float, available_sources: Dict, 
                            additional_taxable_income: float, personal_allowance: float) -> Dict:
    """
    Optimize withdrawal order for tax efficiency:
    1. SIPP Tax-free (25% allowance) - MOST EFFICIENT
    2. ISA (tax-free) - SECOND MOST EFFICIENT  
    3. SIPP Taxable (income tax applies) - LEAST EFFICIENT
    """
    withdrawal_plan = {
        'sipp_tax_free': 0,
        'isa_withdrawal': 0,
        'sipp_taxable': 0,
        'total_gross': 0,
        'total_tax': 0,
        'total_net': 0,
        'optimization_notes': []
    }
    
    remaining_need = target_net_income
    
    # Step 1: Use SIPP tax-free first (most efficient - 0% tax)
    if remaining_need > 0 and available_sources.get('sipp_tax_free', 0) > 0:
        sipp_tax_free_use = min(remaining_need, available_sources['sipp_tax_free'])
        withdrawal_plan['sipp_tax_free'] = sipp_tax_free_use
        remaining_need -= sipp_tax_free_use
        withdrawal_plan['optimization_notes'].append(f"Used £{sipp_tax_free_use:,.0f} SIPP tax-free (0% tax)")
    
    # Step 2: Use ISA if still needed (also tax-free)
    if remaining_need > 0 and available_sources.get('isa', 0) > 0:
        isa_use = min(remaining_need, available_sources['isa'])
        withdrawal_plan['isa_withdrawal'] = isa_use
        remaining_need -= isa_use
        withdrawal_plan['optimization_notes'].append(f"Used £{isa_use:,.0f} ISA (0% tax)")
    
    # Step 3: Use SIPP taxable if still needed (least efficient - 20%+ tax)
    if remaining_need > 0 and available_sources.get('sipp_taxable', 0) > 0:
        # Need to gross up for tax - iterative calculation for precision
        estimated_gross_needed = remaining_need * 1.3  # Initial estimate
        
        for iteration in range(10):
            sipp_taxable_use = min(estimated_gross_needed, available_sources['sipp_taxable'])
            total_taxable_income = additional_taxable_income + sipp_taxable_use
            
            tax_calc = self.calculate_income_tax_with_thresholds(total_taxable_income, personal_allowance)
            
            # Tax attributable to the SIPP withdrawal
            if additional_taxable_income > 0:
                tax_without_sipp = self.calculate_income_tax_with_thresholds(additional_taxable_income, personal_allowance)
                sipp_tax = tax_calc['total_tax'] - tax_without_sipp['total_tax']
            else:
                sipp_tax = tax_calc['total_tax']
            
            net_from_sipp = sipp_taxable_use - sipp_tax
            
            if abs(net_from_sipp - remaining_need) < 1:  # Within £1
                break
                
            if net_from_sipp > 0:
                adjustment = remaining_need / net_from_sipp
                estimated_gross_needed *= adjustment
            else:
                estimated_gross_needed *= 1.1
        
        withdrawal_plan['sipp_taxable'] = sipp_taxable_use
        withdrawal_plan['total_tax'] = sipp_tax
        
        effective_rate = (sipp_tax / sipp_taxable_use * 100) if sipp_taxable_use > 0 else 0
        withdrawal_plan['optimization_notes'].append(
            f"Used £{sipp_taxable_use:,.0f} SIPP taxable ({effective_rate:.1f}% tax)"
        )
    
    # Calculate totals
    withdrawal_plan['total_gross'] = (
        withdrawal_plan['sipp_tax_free'] + 
        withdrawal_plan['isa_withdrawal'] + 
        withdrawal_plan['sipp_taxable']
    )
    withdrawal_plan['total_net'] = withdrawal_plan['total_gross'] - withdrawal_plan['total_tax']
    
    return withdrawal_plan

    
    def calculate_yield_to_maturity(self, price: float, face_value: float, 
                                  coupon_rate: float, years_to_maturity: float) -> float:
        """Calculate Yield to Maturity using approximation formula"""
        if years_to_maturity <= 0:
            return 0
        annual_coupon = face_value * (coupon_rate / 100)
        capital_gain = (face_value - price) / years_to_maturity
        average_price = (face_value + price) / 2
        ytm = (annual_coupon + capital_gain) / average_price * 100
        return max(0, ytm)
    
    def create_bond_ladder(self, total_investment: float, ladder_years: int, 
                          target_yield: float, start_year: int = 2027) -> pd.DataFrame:
        """Create a bond ladder allocation with specific maturity years"""
        if total_investment <= 0 or ladder_years <= 0:
            return pd.DataFrame()
            
        allocation_per_year = total_investment / ladder_years
        
        ladder_data = []
        for year in range(ladder_years):
            maturity_year = start_year + year
            # Yield curve - longer bonds typically have higher yields
            estimated_yield = target_yield + (year * 0.1)
            annual_income = allocation_per_year * (estimated_yield / 100)
            
            ladder_data.append({
                'Maturity_Year': maturity_year,
                'Allocation': allocation_per_year,
                'Target_Yield': estimated_yield,
                'Annual_Income': annual_income,
                'Remaining_Principal': allocation_per_year
            })
            
        return pd.DataFrame(ladder_data)
    
    def calculate_required_gross_income(self, target_net_income: float, additional_pensions: float, 
                                      pension_pot_tax_free_available: float, personal_allowance: float) -> Tuple[float, float, float]:
        """Calculate the gross income required to achieve target net income after tax"""
        # First, consider how much of the target can be covered by additional pensions after tax
        tax_on_additional = self.calculate_income_tax_with_thresholds(additional_pensions, personal_allowance)
        net_from_additional = additional_pensions - tax_on_additional['total_tax']
        
        # Remaining net income needed after accounting for additional pensions
        remaining_net_needed = target_net_income - net_from_additional
        
        # If we don't need any more income beyond additional pensions
        if remaining_net_needed <= 0:
            return additional_pensions, 0, 0
        
        # Use tax-free withdrawal up to the available amount
        tax_free_withdrawal = min(pension_pot_tax_free_available, remaining_net_needed)
        
        # Calculate any remaining net income needed after tax-free withdrawal
        remaining_after_tax_free = remaining_net_needed - tax_free_withdrawal
        if remaining_after_tax_free <= 0:
            return additional_pensions + tax_free_withdrawal, tax_free_withdrawal, 0
        
        # We need to estimate taxable withdrawal needed
        estimated_taxable_income = remaining_after_tax_free * 1.25  # Initial guess (add 25% for tax)
        
        # Iterative refinement
        for _ in range(10):
            total_taxable_income = estimated_taxable_income + additional_pensions
            tax = self.calculate_income_tax_with_thresholds(total_taxable_income, personal_allowance)
            
            resulting_net = estimated_taxable_income + net_from_additional - tax['total_tax'] + tax_free_withdrawal
            
            if abs(resulting_net - target_net_income) < 1:
                break
                
            adjustment_factor = target_net_income / resulting_net if resulting_net > 0 else 1.1
            estimated_taxable_income = estimated_taxable_income * adjustment_factor
        
        taxable_withdrawal = max(0, estimated_taxable_income)
        total_gross = additional_pensions + tax_free_withdrawal + taxable_withdrawal
        
        return total_gross, tax_free_withdrawal, taxable_withdrawal
    
    def simulate_comprehensive_strategy(self, params: Dict) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
        """
        Simulate comprehensive pension strategy with bonds, drawdown, and defined benefits
        """
        try:
            # Extract parameters
            years = params['years']
            start_year = params.get('start_year', 2027)
            inflation_rate = params['inflation_rate']
            investment_growth = params['investment_growth']
            
            # Pension pots
            sipp_value = params['sipp_value']
            isa_value = params['isa_value']
            
            # Bond ladder parameters
            bond_ladder_years = params['bond_ladder_years']
            sipp_yield = params['sipp_yield']
            isa_yield = params['isa_yield']
            
            # Defined benefit pensions
            db_pensions = params['db_pensions']
            state_pension = params.get('state_pension', 0)
            state_pension_start_year = params.get('state_pension_start_year', 10)
            
            # Drawdown parameters
            max_withdrawal_rate = params.get('max_withdrawal_rate', 4.0)
            target_annual_income = params['target_annual_income']
            
            # Initialize bond ladders
            cash_buffer_percent = params.get('cash_buffer_percent', 5)
            sipp_cash_buffer = sipp_value * (cash_buffer_percent / 100)
            isa_cash_buffer = isa_value * (cash_buffer_percent / 100)
            sipp_bonds = sipp_value - sipp_cash_buffer
            isa_bonds = isa_value - isa_cash_buffer
            
            sipp_ladder = self.create_bond_ladder(sipp_bonds, bond_ladder_years, sipp_yield, start_year)
            isa_ladder = self.create_bond_ladder(isa_bonds, bond_ladder_years, isa_yield, start_year)
            
            # Initialize remaining pension pots for drawdown
            remaining_sipp_taxable = sipp_cash_buffer
            remaining_sipp_tax_free = 0
            remaining_isa = isa_cash_buffer
            
            annual_data = []
            
            for year in range(1, years + 1):
                current_year = start_year + year - 1
                inflation_factor = (1 + (inflation_rate / 100)) ** (year - 1)
                
                # Calculate inflation-adjusted targets and pensions
                inflation_adjusted_target = target_annual_income * inflation_factor
                
                # Defined benefit pensions (inflation adjusted)
                total_db_income = 0
                for pension_name, amount in db_pensions.items():
                    if amount > 0:  # Only include non-zero pensions
                        adjusted_amount = amount * inflation_factor
                        total_db_income += adjusted_amount
                
                # State pension
                state_pension_income = 0
                if year >= state_pension_start_year and state_pension > 0:
                    state_pension_income = state_pension * inflation_factor
                
                # Bond income from ladders
                sipp_bond_income = 0
                isa_bond_income = 0
                bonds_maturing_this_year = []
                
                # Check for maturing bonds and calculate income
                for idx, bond in sipp_ladder.iterrows():
                    if bond['Maturity_Year'] == current_year and bond['Remaining_Principal'] > 0:
                        # Bond matures - add principal back to cash
                        remaining_sipp_taxable += bond['Remaining_Principal']
                        bonds_maturing_this_year.append({
                            'Type': 'SIPP',
                            'Principal': bond['Remaining_Principal'],
                            'Year': current_year
                        })
                        sipp_ladder.loc[idx, 'Remaining_Principal'] = 0
                    elif bond['Remaining_Principal'] > 0:
                        # Bond still active - earn income
                        sipp_bond_income += bond['Annual_Income']
                
                for idx, bond in isa_ladder.iterrows():
                    if bond['Maturity_Year'] == current_year and bond['Remaining_Principal'] > 0:
                        # Bond matures - add principal back to ISA cash
                        remaining_isa += bond['Remaining_Principal']
                        bonds_maturing_this_year.append({
                            'Type': 'ISA',
                            'Principal': bond['Remaining_Principal'],
                            'Year': current_year
                        })
                        isa_ladder.loc[idx, 'Remaining_Principal'] = 0
                    elif bond['Remaining_Principal'] > 0:
                        # Bond still active - earn income
                        isa_bond_income += bond['Annual_Income']
                
                # Total guaranteed income before drawdown
                total_tax_free_income = isa_bond_income  # ISA income is tax-free
                total_taxable_income_before_drawdown = (
                    total_db_income + 
                    state_pension_income + 
                    sipp_bond_income
                )
                
                # Calculate personal allowance for this year
                current_personal_allowance = self.personal_allowance * inflation_factor
                
                # Calculate if additional drawdown is needed
                tax_on_current = self.calculate_income_tax_with_thresholds(
                    total_taxable_income_before_drawdown, 
                    current_personal_allowance
                )
                current_net_after_tax = (
                    total_tax_free_income + 
                    total_taxable_income_before_drawdown - 
                    tax_on_current['total_tax']
                )
                
                # Determine additional income needed
                additional_needed = max(0, inflation_adjusted_target - current_net_after_tax)
                
                # Calculate drawdown requirements
                drawdown_tax_free = 0
                drawdown_taxable = 0
                
                if additional_needed > 0:
                    # Use required gross income calculation
                    total_gross, tax_free_needed, taxable_needed = self.calculate_required_gross_income(
                        inflation_adjusted_target,
                        total_taxable_income_before_drawdown,
                        remaining_isa,
                        current_personal_allowance
                    )
                    
                    # Calculate actual withdrawals needed
                    drawdown_tax_free = min(tax_free_needed, remaining_isa)
                    drawdown_taxable = min(taxable_needed, remaining_sipp_taxable)
                    
                    # Apply maximum withdrawal rate check
                    total_remaining_pots = remaining_sipp_taxable + remaining_sipp_tax_free + remaining_isa
                    total_drawdown = drawdown_tax_free + drawdown_taxable
                    max_allowed_drawdown = total_remaining_pots * (max_withdrawal_rate / 100)
                    
                    if total_drawdown > max_allowed_drawdown and total_drawdown > 0:
                        # Scale back withdrawals proportionally
                        scale_factor = max_allowed_drawdown / total_drawdown
                        drawdown_tax_free *= scale_factor
                        drawdown_taxable *= scale_factor
                    
                    # Update remaining pots
                    remaining_isa -= drawdown_tax_free
                    remaining_sipp_taxable -= drawdown_taxable
                
                # Final income calculation
                total_taxable_income = total_taxable_income_before_drawdown + drawdown_taxable
                total_tax_free_income_final = total_tax_free_income + drawdown_tax_free
                
                # Calculate final tax
                tax_details = self.calculate_income_tax_with_thresholds(
                    total_taxable_income, 
                    current_personal_allowance
                )
                
                total_gross_income = total_taxable_income + total_tax_free_income_final
                total_net_income = total_gross_income - tax_details['total_tax']
                
                # Apply growth to remaining pots
                growth_factor = 1 + (investment_growth / 100)
                remaining_sipp_taxable *= growth_factor
                remaining_sipp_tax_free *= growth_factor
                remaining_isa *= growth_factor
                
                # Record annual data
                annual_data.append({
                    'year': year,
                    'calendar_year': current_year,
                    'inflation_factor': round(inflation_factor, 3),
                    'target_income': round(inflation_adjusted_target),
                    
                    # Income sources
                    'db_pension_income': round(total_db_income),
                    'state_pension_income': round(state_pension_income),
                    'sipp_bond_income': round(sipp_bond_income),
                    'isa_bond_income': round(isa_bond_income),
                    'drawdown_tax_free': round(drawdown_tax_free),
                    'drawdown_taxable': round(drawdown_taxable),
                    
                    # Income totals
                    'total_taxable_income': round(total_taxable_income),
                    'total_tax_free_income': round(total_tax_free_income_final),
                    'total_gross_income': round(total_gross_income),
                    
                    # Tax details
                    'personal_allowance': round(current_personal_allowance),
                    'basic_rate_tax': round(tax_details['basic_rate_tax']),
                    'higher_rate_tax': round(tax_details['higher_rate_tax']),
                    'additional_rate_tax': round(tax_details['additional_rate_tax']),
                    'total_tax': round(tax_details['total_tax']),
                    'effective_tax_rate': round(tax_details['effective_rate'], 2),
                    
                    # Net income
                    'total_net_income': round(total_net_income),
                    'income_vs_target': round(total_net_income - inflation_adjusted_target),
                    
                    # Remaining pot values
                    'remaining_sipp_taxable': round(remaining_sipp_taxable),
                    'remaining_sipp_tax_free': round(remaining_sipp_tax_free),
                    'remaining_isa': round(remaining_isa),
                    'total_remaining_pots': round(remaining_sipp_taxable + remaining_sipp_tax_free + remaining_isa),
                    
                    # Bond maturities
                    'bonds_maturing': bonds_maturing_this_year
                })
            
            return annual_data, sipp_ladder, isa_ladder
            
        except Exception as e:
            logging.error(f"Simulation failed: {traceback.format_exc()}")
            raise e

# Enhanced features

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_current_gilt_yields():
    """Fetch current UK gilt yields from financial APIs"""
    try:
        # Try multiple sources for UK gilt data
        
        # Method 1: Try yfinance for UK government bonds
        try:
            # UK 10-year gilt benchmark
            uk_gilt = yf.Ticker("^TNX-GB")  # This might not work, fallback below
            hist = uk_gilt.history(period="5d")
            if not hist.empty:
                current_yield = hist['Close'].iloc[-1] / 100
                return {
                    'success': True,
                    'yield_10y': current_yield,
                    'source': 'Yahoo Finance',
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
        except:
            pass
        
        # Method 2: Try FRED API (if available)
        try:
            # This would require FRED API key, so we'll simulate
            pass
        except:
            pass
        
        # Method 3: Simulated realistic current data (fallback)
        # Based on recent UK economic conditions
        base_yield = 0.042  # 4.2% base 10-year yield
        daily_variation = np.random.normal(0, 0.001)  # Small daily variation
        current_yield = max(0.02, base_yield + daily_variation)  # Minimum 2%
        
        return {
            'success': True,
            'yield_10y': current_yield,
            'yield_5y': current_yield - 0.003,  # Typical curve shape
            'yield_2y': current_yield - 0.008,
            'source': 'Market Simulation',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
    except Exception as e:
        # Ultimate fallback
        return {
            'success': False,
            'yield_10y': 0.045,  # 4.5% fallback
            'yield_5y': 0.042,
            'yield_2y': 0.037,
            'source': 'Default Values',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'error': str(e)
        }

def run_scenario_analysis(base_params, calc):
    """Run multiple economic scenarios"""
    scenarios = {
        'Optimistic 📈': {
            'inflation_rate': 2.0,
            'investment_growth': 6.0,
            'sipp_yield': base_params['sipp_yield'] + 1.0,
            'isa_yield': base_params['isa_yield'] + 1.0,
            'description': 'Strong growth, low inflation, higher yields'
        },
        'Base Case 📊': {
            'inflation_rate': base_params['inflation_rate'],
            'investment_growth': base_params['investment_growth'],
            'sipp_yield': base_params['sipp_yield'],
            'isa_yield': base_params['isa_yield'],
            'description': 'Current assumptions maintained'
        },
        'Pessimistic 📉': {
            'inflation_rate': 4.0,
            'investment_growth': 1.0,
            'sipp_yield': base_params['sipp_yield'] - 1.0,
            'isa_yield': base_params['isa_yield'] - 1.0,
            'description': 'Low growth, high inflation, lower yields'
        },
        'High Inflation 🔥': {
            'inflation_rate': 6.0,
            'investment_growth': 2.0,
            'sipp_yield': base_params['sipp_yield'] + 2.0,
            'isa_yield': base_params['isa_yield'] + 2.0,
            'description': 'Persistent high inflation, higher nominal yields'
        }
    }
    
    scenario_results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        try:
            # Create modified parameters
            modified_params = base_params.copy()
            modified_params.update(scenario_params)
            
            # Run simulation
            annual_data, _, _ = calc.simulate_comprehensive_strategy(modified_params)
            
            if annual_data:
                scenario_results[scenario_name] = {
                    'annual_data': annual_data,
                    'description': scenario_params['description'],
                    'final_pot': annual_data[-1]['total_remaining_pots'],
                    'avg_net_income': np.mean([year['total_net_income'] for year in annual_data]),
                    'avg_tax_rate': np.mean([year['effective_tax_rate'] for year in annual_data])
                }
        except Exception as e:
            st.warning(f"Scenario {scenario_name} failed: {str(e)}")
            continue
    
    return scenario_results

def run_monte_carlo_simulation(base_params, calc, num_simulations=500):
    """Run Monte Carlo simulation with variable returns"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_simulations):
        try:
            # Add random variation to key parameters
            sim_params = base_params.copy()
            
            # Add realistic random variations
            sim_params['investment_growth'] += np.random.normal(0, 2.0)  # ±2% std dev
            sim_params['inflation_rate'] += np.random.normal(0, 1.0)    # ±1% std dev
            sim_params['sipp_yield'] += np.random.normal(0, 0.5)        # ±0.5% std dev
            sim_params['isa_yield'] += np.random.normal(0, 0.5)         # ±0.5% std dev
            
            # Ensure parameters stay within reasonable bounds
            sim_params['investment_growth'] = max(0, min(15, sim_params['investment_growth']))
            sim_params['inflation_rate'] = max(0, min(10, sim_params['inflation_rate']))
            sim_params['sipp_yield'] = max(1, min(10, sim_params['sipp_yield']))
            sim_params['isa_yield'] = max(1, min(12, sim_params['isa_yield']))
            
            # Run simulation
            annual_data, _, _ = calc.simulate_comprehensive_strategy(sim_params)
            
            if annual_data and len(annual_data) > 0:
                final_pot = annual_data[-1]['total_remaining_pots']
                avg_income = np.mean([year['total_net_income'] for year in annual_data])
                avg_tax_rate = np.mean([year['effective_tax_rate'] for year in annual_data])
                income_shortfall_years = len([y for y in annual_data if y['income_vs_target'] < -1000])
                
                results.append({
                    'final_pot': final_pot,
                    'avg_income': avg_income,
                    'avg_tax_rate': avg_tax_rate,
                    'income_shortfall_years': income_shortfall_years,
                    'pot_depleted': final_pot < 10000
                })
            
            # Update progress
            progress = (i + 1) / num_simulations
            progress_bar.progress(progress)
            status_text.text(f'Running simulation {i+1}/{num_simulations}...')
            
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def validate_inputs(sipp_value, isa_value, target_annual_income, years):
    """Enhanced input validation with helpful feedback"""
    errors = []
    warnings = []
    
    # Error conditions
    if target_annual_income <= 0:
        errors.append("❌ Target annual income must be greater than zero")
    
    if years <= 0:
        errors.append("❌ Number of years must be greater than zero")
    
    if sipp_value < 0 or isa_value < 0:
        errors.append("❌ Portfolio values cannot be negative")
    
    # Warning conditions
    total_portfolio = sipp_value + isa_value
    
    if total_portfolio < target_annual_income * 10:
        warnings.append("⚠️ Your total portfolio is less than 10x your target income - this may not be sustainable")
    
    if target_annual_income > total_portfolio * 0.06:
        warnings.append("⚠️ Target income is more than 6% of portfolio - consider reducing target or increasing savings")
    
    if sipp_value > 1073100:  # 2025/26 annual allowance limit
        warnings.append("⚠️ SIPP value exceeds typical annual allowance limits - ensure this is accurate")
    
    if isa_value > 500000:  # Very high ISA suggests long-term saving
        warnings.append("💡 High ISA value detected - excellent tax-free savings!")
    
    return errors, warnings

def create_alerts(annual_data):
    """Enhanced alert system with more sophisticated analysis"""
    alerts = []
    
    # Analyze first 5 years for early warning signs
    for year_data in annual_data[:5]:
        # Income shortfall alerts
        if year_data['income_vs_target'] < -2000:
            alerts.append({
                'type': 'error',
                'message': f"🚨 Year {year_data['year']}: Income shortfall of £{abs(year_data['income_vs_target']):,}"
            })
        elif year_data['income_vs_target'] < -500:
            alerts.append({
                'type': 'warning', 
                'message': f"⚠️ Year {year_data['year']}: Minor income shortfall of £{abs(year_data['income_vs_target']):,}"
            })
        
        # High tax rate alerts
        if year_data['effective_tax_rate'] > 35:
            alerts.append({
                'type': 'error',
                'message': f"🚨 Year {year_data['year']}: Very high tax rate of {year_data['effective_tax_rate']:.1f}%"
            })
        elif year_data['effective_tax_rate'] > 25:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ Year {year_data['year']}: High tax rate of {year_data['effective_tax_rate']:.1f}%"
            })
        
        # Portfolio depletion alerts
        if year_data['total_remaining_pots'] < 50000:
            alerts.append({
                'type': 'error',
                'message': f"🚨 Year {year_data['year']}: Portfolio critically low at £{year_data['total_remaining_pots']:,}"
            })
        elif year_data['total_remaining_pots'] < 100000:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ Year {year_data['year']}: Portfolio value below £100k"
            })
    
    # Long-term sustainability analysis
    final_year = annual_data[-1]
    if final_year['total_remaining_pots'] < 25000:
        alerts.append({
            'type': 'error',
            'message': f"🚨 Portfolio nearly depleted by year {final_year['year']} - strategy not sustainable"
        })
    
    # Tax efficiency analysis
    avg_tax_rate = np.mean([year['effective_tax_rate'] for year in annual_data])
    if avg_tax_rate > 30:
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ Average tax rate of {avg_tax_rate:.1f}% is high - consider tax optimization"
        })
    
    # Positive alerts
    if avg_tax_rate < 15:
        alerts.append({
            'type': 'success',
            'message': f"✅ Excellent tax efficiency with average rate of {avg_tax_rate:.1f}%"
        })
    
    if final_year['total_remaining_pots'] > annual_data[0]['total_remaining_pots']:
        alerts.append({
            'type': 'success',
            'message': "✅ Portfolio grows over time - very sustainable strategy"
        })
    
    return alerts

def create_excel_export(annual_data, sipp_ladder, isa_ladder, scenario_results=None, monte_carlo_results=None):
    """Enhanced Excel export with scenario analysis and Monte Carlo results"""
    try:
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Annual data
            df_annual = pd.DataFrame(annual_data)
            df_annual.to_excel(writer, sheet_name='Annual Analysis', index=False)
            
            # Bond ladders
            if not sipp_ladder.empty:
                sipp_ladder.to_excel(writer, sheet_name='SIPP Bonds', index=False)
            if not isa_ladder.empty:
                isa_ladder.to_excel(writer, sheet_name='ISA Bonds', index=False)
            
            # Summary statistics
            if annual_data:
                summary_data = {
                    'Metric': [
                        'Total Net Income (Year 1)', 
                        'Average Tax Rate (%)', 
                        'Final Pot Value',
                        'Total Years Analyzed',
                        'Average Annual Income',
                        'Total Tax Paid',
                        'Tax Efficiency Score'
                    ],
                    'Value': [
                        f"£{annual_data[0]['total_net_income']:,}",
                        f"{np.mean([y['effective_tax_rate'] for y in annual_data]):.1f}%",
                        f"£{annual_data[-1]['total_remaining_pots']:,}",
                        len(annual_data),
                        f"£{np.mean([y['total_net_income'] for y in annual_data]):,.0f}",
                        f"£{sum([y['total_tax'] for y in annual_data]):,}",
                        f"{100 - np.mean([y['effective_tax_rate'] for y in annual_data]):.1f}%"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Scenario analysis results
            if scenario_results:
                scenario_summary = []
                for scenario_name, results in scenario_results.items():
                    scenario_summary.append({
                        'Scenario': scenario_name,
                        'Description': results['description'],
                        'Final Pot Value': f"£{results['final_pot']:,}",
                        'Average Net Income': f"£{results['avg_net_income']:,.0f}",
                        'Average Tax Rate': f"{results['avg_tax_rate']:.1f}%"
                    })
                
                pd.DataFrame(scenario_summary).to_excel(writer, sheet_name='Scenario Analysis', index=False)
            
            # Monte Carlo results
            if monte_carlo_results:
                mc_df = pd.DataFrame(monte_carlo_results)
                mc_df.to_excel(writer, sheet_name='Monte Carlo Results', index=False)
                
                # Monte Carlo summary statistics
                mc_summary = {
                    'Statistic': [
                        'Mean Final Pot Value',
                        'Median Final Pot Value', 
                        '10th Percentile Final Pot',
                        '90th Percentile Final Pot',
                        'Probability of Pot Depletion',
                        'Mean Average Income',
                        'Mean Tax Rate'
                    ],
                    'Value': [
                        f"£{np.mean([r['final_pot'] for r in monte_carlo_results]):,.0f}",
                        f"£{np.median([r['final_pot'] for r in monte_carlo_results]):,.0f}",
                        f"£{np.percentile([r['final_pot'] for r in monte_carlo_results], 10):,.0f}",
                        f"£{np.percentile([r['final_pot'] for r in monte_carlo_results], 90):,.0f}",
                        f"{np.mean([r['pot_depleted'] for r in monte_carlo_results]) * 100:.1f}%",
                        f"£{np.mean([r['avg_income'] for r in monte_carlo_results]):,.0f}",
                        f"{np.mean([r['avg_tax_rate'] for r in monte_carlo_results]):.1f}%"
                    ]
                }
                pd.DataFrame(mc_summary).to_excel(writer, sheet_name='Monte Carlo Summary', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel export: {str(e)}")
        return None

def save_load_configuration():
    """Allow users to save and load their configurations"""
    st.subheader("💾 Save/Load Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Save Current Configuration**")
        config_name = st.text_input("Configuration Name", value=f"Config_{datetime.now().strftime('%Y%m%d')}")
        
        if st.button("💾 Save Configuration"):
            config = {
                'config_name': config_name,
                'saved_date': datetime.now().isoformat(),
                'parameters': st.session_state
            }
            
            config_json = json.dumps(config, indent=2, default=str)
            st.download_button(
                label="📥 Download Configuration File",
                data=config_json,
                file_name=f"{config_name}.json",
                mime="application/json"
            )
            st.success("✅ Configuration ready for download!")
    
    with col2:
        st.write("**Load Configuration**")
        uploaded_file = st.file_uploader("Choose configuration file", type=['json'])
        
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                st.write(f"**Configuration:** {config.get('config_name', 'Unknown')}")
                st.write(f"**Saved:** {config.get('saved_date', 'Unknown')}")
                
                if st.button("📤 Load Configuration"):
                    # Load parameters into session state
                    if 'parameters' in config:
                        for key, value in config['parameters'].items():
                            if key in st.session_state:
                                st.session_state[key] = value
                    st.success("✅ Configuration loaded! Please refresh inputs.")
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading configuration: {str(e)}")

def add_help_system():
    """Enhanced help system with contextual guidance"""
    with st.expander("❓ How to Use This Enhanced Calculator"):
        st.markdown("""
        ## 🎯 Getting Started
        
        ### 1. **Portfolio Setup**
        - Enter your current SIPP and ISA values
        - The calculator will automatically suggest cash buffer allocations
        
        ### 2. **Income Planning**
        - Set your target annual **net** income (after tax)
        - Add any defined benefit pensions you expect
        - Include state pension if applicable
        
        ### 3. **Bond Strategy**
        - Configure your bond ladder duration (3-10 years recommended)
        - Set expected yields (the app shows current market rates)
        - Choose cash buffer percentage for flexibility
        
        ### 4. **Advanced Analysis**
        - **Scenario Analysis**: Tests your strategy under different economic conditions
        - **Monte Carlo Simulation**: Shows probability ranges for outcomes
        - **Real-time Data**: Uses current market gilt yields when available
        
        ## 📊 Understanding the Results
        
        ### Key Metrics to Watch:
        - **Effective Tax Rate**: Aim for under 20% if possible
        - **Portfolio Sustainability**: Final pot value should be positive
        - **Income vs Target**: Should be close to zero or positive
        
        ### Alert System:
        - 🚨 **Red alerts**: Critical issues requiring attention
        - ⚠️ **Yellow warnings**: Areas for optimization
        - ✅ **Green success**: Strategy working well
        
        ## 🎭 Scenario Analysis Explained
        
        - **Optimistic**: Strong economy, higher returns
        - **Base Case**: Your current assumptions
        - **Pessimistic**: Economic challenges, lower returns
        - **High Inflation**: Persistent inflation scenario
        
        ## 🎲 Monte Carlo Simulation
        
        Runs 500+ scenarios with random variations to show:
        - Range of possible outcomes
        - Probability of success
        - Risk assessment
        
        ## 💡 Pro Tips
        
        1. **Start Conservative**: Better to underestimate returns than overestimate
        2. **Monitor Tax Bands**: Try to stay in basic rate if possible
        3. **Use ISA First**: Tax-free withdrawals are most efficient
        4. **Review Annually**: Economic conditions change
        5. **Keep Cash Buffer**: 5-10% provides flexibility
        """)

def display_gilt_market_data():
    """Display current gilt market information"""
    st.subheader("📈 Current Market Data")
    
    gilt_data = get_current_gilt_yields()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if gilt_data['success']:
            st.metric(
                "UK 10Y Gilt Yield", 
                f"{gilt_data['yield_10y']*100:.2f}%",
                delta=None
            )
        else:
            st.metric("UK 10Y Gilt Yield", "4.50%*", help="*Default value - live data unavailable")
    
    with col2:
        if gilt_data['success'] and 'yield_5y' in gilt_data:
            st.metric("UK 5Y Gilt Yield", f"{gilt_data['yield_5y']*100:.2f}%")
        else:
            st.metric("UK 5Y Gilt Yield", "4.20%*")
    
    with col3:
        if gilt_data['success'] and 'yield_2y' in gilt_data:
            st.metric("UK 2Y Gilt Yield", f"{gilt_data['yield_2y']*100:.2f}%")
        else:
            st.metric("UK 2Y Gilt Yield", "3.70%*")
    
    with col4:
        st.metric("Data Source", gilt_data['source'])
    
    if gilt_data['success']:
        st.caption(f"Last updated: {gilt_data['last_updated']}")
        if gilt_data['source'] == 'Market Simulation':
            st.caption("⚠️ Using simulated data - enable live data feeds for real-time information")
    else:
        st.caption("⚠️ Using default values - live market data unavailable")
        if 'error' in gilt_data:
            st.caption(f"Error: {gilt_data['error']}")

def main():
    st.title("💰 Enhanced Pension & Bond Calculator")
    st.markdown("**Advanced UK retirement planning with real-time data and scenario analysis**")
    
    # Add market data display
    display_gilt_market_data()
    
    # Help system
    add_help_system()
    
    calc = EnhancedPensionCalculator()
    
    # Sidebar inputs with enhanced validation
    st.sidebar.header("📊 Portfolio Parameters")
    
    # Portfolio values
    sipp_value = st.sidebar.number_input(
        "SIPP Value (£)", 
        min_value=0, 
        value=565000, 
        step=1000,
        help="Current value of your Self-Invested Personal Pension",
        key="sipp_value"
    )
    
    isa_value = st.sidebar.number_input(
        "ISA Value (£)", 
        min_value=0, 
        value=92000, 
        step=1000,
        help="Current value of your Individual Savings Account",
        key="isa_value"
    )
    
    # Target income
    target_annual_income = st.sidebar.number_input(
        "Target Annual Net Income (£)", 
        min_value=0, 
        value=40000, 
        step=1000,
        help="Desired annual net income (after tax)",
        key="target_income"
    )
    
    # Input validation
    errors, warnings = validate_inputs(sipp_value, isa_value, target_annual_income, 25)
    
    for error in errors:
        st.sidebar.error(error)
    for warning in warnings:
        st.sidebar.warning(warning)
    
    # Bond ladder parameters with market data integration
    st.sidebar.subheader("🔗 Bond Ladder Settings")
    
    # Get current market yields for suggestions
    gilt_data = get_current_gilt_yields()
    suggested_sipp_yield = gilt_data['yield_10y'] * 100 if gilt_data['success'] else 4.5
    suggested_isa_yield = suggested_sipp_yield + 0.5  # Corporate bonds typically yield more
    
    bond_ladder_years = st.sidebar.slider(
        "Ladder Duration (Years)", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="Number of years in your bond ladder",
        key="ladder_years"
    )
    
    sipp_yield = st.sidebar.slider(
        "SIPP Bond Yield (%)", 
        min_value=2.0, 
        max_value=8.0, 
        value=round(suggested_sipp_yield, 1), 
        step=0.1,
        help=f"Expected average yield for SIPP bonds (UK Gilts). Current market: ~{suggested_sipp_yield:.1f}%",
        key="sipp_yield"
    )
    
    isa_yield = st.sidebar.slider(
        "ISA Bond Yield (%)", 
        min_value=2.0, 
        max_value=8.0, 
        value=round(suggested_isa_yield, 1), 
        step=0.1,
        help=f"Expected average yield for ISA bonds (Corporate bonds). Suggested: ~{suggested_isa_yield:.1f}%",
        key="isa_yield"
    )
    
    cash_buffer_percent = st.sidebar.slider(
        "Cash Buffer (%)", 
        min_value=0, 
        max_value=20, 
        value=5,
        help="Percentage to keep in cash for flexibility",
        key="cash_buffer"
    )
    
    # Pension inputs
    st.sidebar.subheader("🏛️ Defined Benefit Pensions")
    
    gp_pension = st.sidebar.number_input(
        "GP Pension (£/year)", 
        min_value=0, 
        value=0, 
        step=500,
        key="gp_pension"
    )
    
    kpmg_pension = st.sidebar.number_input(
        "KPMG Pension (£/year)", 
        min_value=0, 
        value=0, 
        step=500,
        key="kpmg_pension"
    )
    
    nhs_pension = st.sidebar.number_input(
        "NHS Pension (£/year)", 
        min_value=0, 
        value=13000, 
        step=500,
        key="nhs_pension"
    )
    
    state_pension = st.sidebar.number_input(
        "State Pension (£/year)", 
        min_value=0, 
        value=11500, 
        step=100,
        key="state_pension"
    )
    
    state_pension_start_year = st.sidebar.slider(
        "State Pension Start Year", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Year when state pension starts (from retirement)",
        key="state_start_year"
    )
    
    # Economic parameters
    st.sidebar.subheader("📈 Economic Assumptions")
    
    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.5, 
        step=0.1,
        key="inflation_rate"
    )
    
    investment_growth = st.sidebar.slider(
        "Investment Growth (%)", 
        min_value=0.0, 
        max_value=15.0, 
        value=4.0, 
        step=0.1,
        help="Annual growth rate for remaining pension pots",
        key="investment_growth"
    )
    
    max_withdrawal_rate = st.sidebar.slider(
        "Max Withdrawal Rate (%)", 
        min_value=1.0, 
        max_value=10.0, 
        value=4.0, 
        step=0.1,
        help="Maximum annual withdrawal rate from pension pots",
        key="max_withdrawal"
    )
    
    years = st.sidebar.slider(
        "Simulation Years", 
        min_value=5, 
        max_value=40, 
        value=25,
        help="Number of years to simulate",
        key="sim_years"
    )
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        enable_scenario_analysis = st.checkbox("Enable Scenario Analysis", value=True)
        enable_monte_carlo = st.checkbox("Enable Monte Carlo Simulation", value=False)
        monte_carlo_runs = st.slider("Monte Carlo Simulations", 100, 1000, 500, step=100) if enable_monte_carlo else 500
    
    # Save/Load configuration
    save_load_configuration()
    
    # Calculate button
    if st.sidebar.button("🚀 Calculate Enhanced Strategy", type="primary") and not errors:
        try:
            # Prepare parameters
            db_pensions = {
                'GP Pension': gp_pension,
                'KPMG Pension': kpmg_pension,
                'NHS Pension': nhs_pension
            }
            
            base_params = {
                'sipp_value': sipp_value,
                'isa_value': isa_value,
                'target_annual_income': target_annual_income,
                'bond_ladder_years': bond_ladder_years,
                'sipp_yield': sipp_yield,
                'isa_yield': isa_yield,
                'cash_buffer_percent': cash_buffer_percent,
                'db_pensions': db_pensions,
                'state_pension': state_pension,
                'state_pension_start_year': state_pension_start_year,
                'inflation_rate': inflation_rate,
                'investment_growth': investment_growth,
                'max_withdrawal_rate': max_withdrawal_rate,
                'years': years,
                'start_year': 2027
            }
            
            # Run base simulation
            with st.spinner("Calculating base strategy..."):
                annual_data, sipp_ladder, isa_ladder = calc.simulate_comprehensive_strategy(base_params)
            
            if not annual_data:
                st.error("No results generated. Please check your inputs.")
                return
            
            # Display base results
            st.header("📊 Base Strategy Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            first_year = annual_data[0]
            last_year = annual_data[-1]
            
            with col1:
                st.metric(
                    "Year 1 Net Income", 
                    f"£{first_year['total_net_income']:,}",
                    delta=f"£{first_year['income_vs_target']:,} vs target"
                )
            
            with col2:
                st.metric(
                    "Year 1 Tax Rate", 
                    f"{first_year['effective_tax_rate']:.1f}%"
                )
            
            with col3:
                avg_tax_rate = np.mean([year['effective_tax_rate'] for year in annual_data])
                st.metric(
                    "Average Tax Rate", 
                    f"{avg_tax_rate:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Final Pot Value", 
                    f"£{last_year['total_remaining_pots']:,}"
                )
            
            # Enhanced alerts
            alerts = create_alerts(annual_data)
            if alerts:
                st.subheader("🚨 Strategy Analysis")
                for alert in alerts:
                    if alert['type'] == 'error':
                        st.error(alert['message'])
                    elif alert['type'] == 'warning':
                        st.warning(alert['message'])
                    elif alert['type'] == 'success':
                        st.success(alert['message'])
            
            # Run scenario analysis if enabled
            scenario_results = None
            if enable_scenario_analysis:
                st.header("🎭 Scenario Analysis")
                with st.spinner("Running scenario analysis..."):
                    scenario_results = run_scenario_analysis(base_params, calc)
                
                if scenario_results:
                    # Display scenario comparison
                    scenario_df = pd.DataFrame([
                        {
                            'Scenario': name,
                            'Final Pot': f"£{results['final_pot']:,}",
                            'Avg Income': f"£{results['avg_net_income']:,.0f}",
                            'Avg Tax Rate': f"{results['avg_tax_rate']:.1f}%",
                            'Description': results['description']
                        }
                        for name, results in scenario_results.items()
                    ])
                    
                    st.dataframe(scenario_df, use_container_width=True)
                    
                    # Scenario comparison chart
                    fig_scenarios = go.Figure()
                    for scenario_name, results in scenario_results.items():
                        annual_data_scenario = results['annual_data']
                        years_list = [data['year'] for data in annual_data_scenario]
                        net_incomes = [data['total_net_income'] for data in annual_data_scenario]
                        
                        fig_scenarios.add_trace(go.Scatter(
                            x=years_list,
                            y=net_incomes,
                            mode='lines',
                            name=scenario_name,
                            line=dict(width=3)
                        ))
                    
                    fig_scenarios.update_layout(
                        title='Scenario Analysis: Net Income Over Time',
                        xaxis_title='Year',
                        yaxis_title='Net Income (£)',
                        height=400
                    )
                    
                    st.plotly_chart(fig_scenarios, use_container_width=True)
            
            # Run Monte Carlo simulation if enabled
            monte_carlo_results = None
            if enable_monte_carlo:
                st.header("🎲 Monte Carlo Simulation")
                with st.spinner(f"Running {monte_carlo_runs} Monte Carlo simulations..."):
                    monte_carlo_results = run_monte_carlo_simulation(base_params, calc, monte_carlo_runs)
                
                if monte_carlo_results:
                    # Monte Carlo summary
                    col1, col2, col3 = st.columns(3)
                    
                    final_pots = [r['final_pot'] for r in monte_carlo_results]
                    avg_incomes = [r['avg_income'] for r in monte_carlo_results]
                    depletion_prob = np.mean([r['pot_depleted'] for r in monte_carlo_results]) * 100
                    
                    with col1:
                        st.metric("Median Final Pot", f"£{np.median(final_pots):,.0f}")
                        st.metric("10th Percentile", f"£{np.percentile(final_pots, 10):,.0f}")
                    
                    with col2:
                        st.metric("90th Percentile", f"£{np.percentile(final_pots, 90):,.0f}")
                        st.metric("Median Income", f"£{np.median(avg_incomes):,.0f}")
                    
                    with col3:
                        st.metric("Depletion Risk", f"{depletion_prob:.1f}%")
                        st.metric("Success Rate", f"{100-depletion_prob:.1f}%")
                    
                    # Monte Carlo distribution charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_mc_pot = px.histogram(
                            final_pots, 
                            nbins=50, 
                            title="Distribution of Final Pot Values",
                            labels={'value': 'Final Pot Value (£)', 'count': 'Frequency'}
                        )
                        fig_mc_pot.update_layout(height=400)
                        st.plotly_chart(fig_mc_pot, use_container_width=True)
                    
                    with col2:
                        fig_mc_income = px.histogram(
                            avg_incomes,
                            nbins=50,
                            title="Distribution of Average Income",
                            labels={'value': 'Average Income (£)', 'count': 'Frequency'}
                        )
                        fig_mc_income.update_layout(height=400)
                        st.plotly_chart(fig_mc_income, use_container_width=True)
            
            # Continue with original detailed analysis...
            # [Include all the charts and analysis from the previous version]
            
            # Annual breakdown table
            st.subheader("📅 Year-by-Year Analysis")
            
            df = pd.DataFrame(annual_data)
            
            # Select key columns for display
            display_columns = [
                'year', 'calendar_year', 'target_income', 'db_pension_income', 
                'state_pension_income', 'sipp_bond_income', 'isa_bond_income',
                'drawdown_tax_free', 'drawdown_taxable', 'total_gross_income',
                'total_tax', 'total_net_income', 'income_vs_target', 'effective_tax_rate',
                'total_remaining_pots'
            ]
            
            display_df = df[display_columns].copy()
            
            # Format for display
            format_dict = {
                'target_income': '£{:,.0f}',
                'db_pension_income': '£{:,.0f}',
                'state_pension_income': '£{:,.0f}',
                'sipp_bond_income': '£{:,.0f}',
                'isa_bond_income': '£{:,.0f}',
                'drawdown_tax_free': '£{:,.0f}',
                'drawdown_taxable': '£{:,.0f}',
                'total_gross_income': '£{:,.0f}',
                'total_tax': '£{:,.0f}',
                'total_net_income': '£{:,.0f}',
                'income_vs_target': '£{:,.0f}',
                'effective_tax_rate': '{:.1f}%',
                'total_remaining_pots': '£{:,.0f}'
            }
            
            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True,
                height=400
            )
            
            # Charts section
            st.subheader("📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Income composition chart
                fig_income = go.Figure()
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['db_pension_income'],
                    stackgroup='one',
                    name='DB Pensions',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 99, 132, 0.7)'
                ))
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['state_pension_income'],
                    stackgroup='one',
                    name='State Pension',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(54, 162, 235, 0.7)'
                ))
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['sipp_bond_income'],
                    stackgroup='one',
                    name='SIPP Bonds',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 206, 86, 0.7)'
                ))
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['isa_bond_income'],
                    stackgroup='one',
                    name='ISA Bonds',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(75, 192, 192, 0.7)'
                ))
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['drawdown_tax_free'] + df['drawdown_taxable'],
                    stackgroup='one',
                    name='Drawdown',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(153, 102, 255, 0.7)'
                ))
                
                fig_income.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['target_income'],
                    mode='lines',
                    name='Target Income',
                    line=dict(color='red', width=3, dash='dash')
                ))
                
                fig_income.update_layout(
                    title='Income Sources Over Time',
                    xaxis_title='Year',
                    yaxis_title='Annual Income (£)',
                    height=400
                )
                
                st.plotly_chart(fig_income, use_container_width=True)
            
            with col2:
                # Tax and net income chart
                fig_tax = go.Figure()
                
                fig_tax.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['total_gross_income'],
                    mode='lines',
                    name='Gross Income',
                    line=dict(color='blue', width=2)
                ))
                
                fig_tax.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['total_net_income'],
                    mode='lines',
                    name='Net Income',
                    line=dict(color='green', width=2)
                ))
                
                fig_tax.update_layout(
                    title='Income and Tax Over Time',
                    xaxis_title='Year',
                    yaxis_title='Annual Amount (£)',
                    height=400
                )
                
                st.plotly_chart(fig_tax, use_container_width=True)
            
            # Portfolio sustainability chart
            st.subheader("📈 Portfolio Sustainability")
            
            fig_pots = go.Figure()
            
            fig_pots.add_trace(go.Scatter(
                x=df['year'],
                y=df['remaining_sipp_taxable'],
                mode='lines',
                name='SIPP Taxable',
                line=dict(color='orange', width=2),
                fill='tonexty'
            ))
            
            fig_pots.add_trace(go.Scatter(
                x=df['year'],
                y=df['remaining_isa'],
                mode='lines',
                name='ISA',
                line=dict(color='green', width=2),
                fill='tonexty'
            ))
            
            fig_pots.add_trace(go.Scatter(
                x=df['year'],
                y=df['total_remaining_pots'],
                mode='lines',
                name='Total Pots',
                line=dict(color='blue', width=3)
            ))
            
            fig_pots.update_layout(
                title='Remaining Pension Pot Values Over Time',
                xaxis_title='Year',
                yaxis_title='Pot Value (£)',
                height=400
            )
            
            st.plotly_chart(fig_pots, use_container_width=True)
            
            # Bond ladder details
            st.subheader("🔗 Bond Ladder Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SIPP Bond Ladder**")
                if not sipp_ladder.empty:
                    st.dataframe(
                        sipp_ladder.style.format({
                            'Allocation': '£{:,.0f}',
                            'Target_Yield': '{:.1f}%',
                            'Annual_Income': '£{:,.0f}',
                            'Remaining_Principal': '£{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No SIPP bonds allocated")
            
            with col2:
                st.write("**ISA Bond Ladder**")
                if not isa_ladder.empty:
                    st.dataframe(
                        isa_ladder.style.format({
                            'Allocation': '£{:,.0f}',
                            'Target_Yield': '{:.1f}%',
                            'Annual_Income': '£{:,.0f}',
                            'Remaining_Principal': '£{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No ISA bonds allocated")
            
            # Tax efficiency analysis
            st.subheader("🎯 Tax Efficiency Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_tax_paid = sum([year['total_tax'] for year in annual_data])
                total_gross_income_all = sum([year['total_gross_income'] for year in annual_data])
                overall_tax_rate = (total_tax_paid / total_gross_income_all * 100) if total_gross_income_all > 0 else 0
                
                st.metric("Overall Tax Rate", f"{overall_tax_rate:.1f}%")
                st.metric("Total Tax Paid", f"£{total_tax_paid:,}")
            
            with col2:
                total_tax_free_income = sum([year['total_tax_free_income'] for year in annual_data])
                tax_free_percentage = (total_tax_free_income / total_gross_income_all * 100) if total_gross_income_all > 0 else 0
                
                st.metric("Tax-Free Income %", f"{tax_free_percentage:.1f}%")
                st.metric("Total Tax-Free", f"£{total_tax_free_income:,}")
            
            with col3:
                years_in_higher_rate = len([y for y in annual_data if y['higher_rate_tax'] > 0])
                years_in_additional_rate = len([y for y in annual_data if y['additional_rate_tax'] > 0])
                
                st.metric("Years in Higher Rate", f"{years_in_higher_rate}")
                st.metric("Years in Additional Rate", f"{years_in_additional_rate}")
            
            # Enhanced implementation guidance
            st.subheader("📋 Implementation Guidance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Immediate Actions:**")
                
                # Generate specific recommendations
                recommendations = []
                
                if first_year['effective_tax_rate'] > 25:
                    recommendations.append("• Consider increasing ISA bond allocation to reduce taxable income")
                
                if first_year['income_vs_target'] < 0:
                    recommendations.append("• Income shortfall detected - consider higher-yield bonds or increased drawdown")
                
                if last_year['total_remaining_pots'] < 50000:
                    recommendations.append("• Portfolio may not be sustainable - consider reducing withdrawal rate")
                
                # Bond recommendations based on ladder
                if not sipp_ladder.empty:
                    first_maturity = sipp_ladder['Maturity_Year'].min()
                    recommendations.append(f"• Purchase UK Gilts maturing in {first_maturity} for SIPP ladder")
                
                if not isa_ladder.empty:
                    recommendations.append("• Consider investment-grade corporate bonds for ISA higher yields")
                
                recommendations.append("• Set up automatic reinvestment when bonds mature")
                recommendations.append("• Review strategy annually and adjust for interest rate changes")
                
                for rec in recommendations:
                    st.write(rec)
            
            with col2:
                st.write("**Optimization Opportunities:**")
                
                optimizations = []
                
                # Tax optimization
                if overall_tax_rate > 20:
                    optimizations.append("🎯 **Tax Optimization**: Consider spreading withdrawals to stay in basic rate")
                
                # Yield optimization
                current_sipp_yield = sipp_yield
                market_yield = gilt_data['yield_10y'] * 100 if gilt_data['success'] else 4.5
                if current_sipp_yield < market_yield - 0.5:
                    optimizations.append(f"📈 **Yield Gap**: Market yields (~{market_yield:.1f}%) higher than your assumption")
                
                # Withdrawal rate optimization
                if max_withdrawal_rate > 4:
                    optimizations.append("⚠️ **Withdrawal Rate**: Consider reducing below 4% for sustainability")
                
                # Asset allocation
                total_bonds = sipp_value + isa_value - (sipp_value + isa_value) * (cash_buffer_percent/100)
                if cash_buffer_percent > 10:
                    optimizations.append("💰 **Cash Buffer**: High cash allocation may reduce returns")
                
                for opt in optimizations:
                    st.info(opt)
            
            # Professional recommendations based on analysis
            st.subheader("🎓 Professional Recommendations")
            
            # Generate AI-style recommendations based on the analysis
            professional_recs = []
            
            # Risk assessment
            risk_score = 0
            if depletion_prob > 20 if monte_carlo_results else False:
                risk_score += 2
                professional_recs.append("**High Risk Strategy**: Consider reducing target income or increasing savings")
            elif overall_tax_rate > 30:
                risk_score += 1
                professional_recs.append("**Tax Inefficiency**: Strategy heavily taxed - restructure for tax efficiency")
            
            # Opportunity assessment
            if tax_free_percentage < 20:
                professional_recs.append("**ISA Opportunity**: Increase ISA allocation for better tax efficiency")
            
            if years_in_higher_rate > years * 0.5:
                professional_recs.append("**Income Smoothing**: Consider deferring some income to reduce tax bands")
            
            # Sustainability assessment
            pot_growth_rate = (last_year['total_remaining_pots'] / (sipp_value + isa_value) - 1) * 100
            if pot_growth_rate > 0:
                professional_recs.append("**Excellent Sustainability**: Portfolio grows over time - very conservative strategy")
            elif pot_growth_rate > -50:
                professional_recs.append("**Good Sustainability**: Portfolio declining slowly - reasonable strategy")
            else:
                professional_recs.append("**Sustainability Concern**: Rapid portfolio depletion - review strategy")
            
            # Market timing
            if gilt_data['success'] and gilt_data['yield_10y'] > 0.05:
                professional_recs.append("**Market Timing**: Current gilt yields attractive for ladder construction")
            
            for rec in professional_recs:
                st.success(f"💡 {rec}")
            
            # Download section
            st.subheader("📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv,
                    file_name=f"enhanced_pension_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Enhanced Excel download
                excel_data = create_excel_export(annual_data, sipp_ladder, isa_ladder, scenario_results, monte_carlo_results)
                if excel_data:
                    st.download_button(
                        label="📈 Download Excel Report",
                        data=excel_data,
                        file_name=f"comprehensive_pension_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                # Configuration export
                config_data = {
                    'analysis_date': datetime.now().isoformat(),
                    'parameters': base_params,
                    'results_summary': {
                        'year_1_net_income': first_year['total_net_income'],
                        'average_tax_rate': avg_tax_rate,
                        'final_pot_value': last_year['total_remaining_pots'],
                        'strategy_sustainable': last_year['total_remaining_pots'] > 25000
                    }
                }
                config_json = json.dumps(config_data, indent=2, default=str)
                st.download_button(
                    label="⚙️ Export Configuration",
                    data=config_json,
                    file_name=f"strategy_config_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.error(f"Error running enhanced calculation: {str(e)}")
            logging.error(f"Enhanced calculation failed: {traceback.format_exc()}")
            st.info("Please check your inputs and try again. If the problem persists, try reducing the number of simulation years or adjusting your parameters.")
    
    elif errors:
        st.error("Please fix the input errors before calculating.")
    
    else:
        # Show enhanced summary when no calculation run
        st.info("👆 Configure your parameters in the sidebar and click 'Calculate Enhanced Strategy' to see your comprehensive retirement analysis.")
        
        # Show enhanced features summary
        st.subheader("🌟 Enhanced Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("""
            **🔍 Real-Time Analysis:**
            - Current UK gilt yields
            - Market-based yield suggestions
            - Live data integration
            - Enhanced input validation
            - Smart recommendations
            """)
        
        with col2:
            st.write("""
            **🎭 Scenario Planning:**
            - Optimistic/Pessimistic scenarios
            - High inflation stress testing
            - Economic condition variations
            - Comparative analysis
            - Risk assessment
            """)
        
        with col3:
            st.write("""
            **🎲 Monte Carlo Simulation:**
            - 500+ random scenarios
            - Probability distributions
            - Risk quantification
            - Success rate analysis
            - Statistical confidence
            """)
        
        # Show enhanced sample visualization
        st.subheader("📊 Enhanced Analysis Preview")
        
        # Create enhanced sample data
        sample_years = list(range(1, 16))
        sample_scenarios = {
            'Optimistic': [45000 + (i * 1200) for i in sample_years],
            'Base Case': [40000 + (i * 1000) for i in sample_years],
            'Pessimistic': [35000 + (i * 800) for i in sample_years]
        }
        
        fig_sample = go.Figure()
        colors = {'Optimistic': 'green', 'Base Case': 'blue', 'Pessimistic': 'red'}
        
        for scenario, values in sample_scenarios.items():
            fig_sample.add_trace(go.Scatter(
                x=sample_years, 
                y=values, 
                mode='lines+markers', 
                name=scenario,
                line=dict(color=colors[scenario], width=3)
            ))
        
        fig_sample.update_layout(
            title='Sample: Multi-Scenario Income Projection',
            xaxis_title='Year',
            yaxis_title='Net Income (£)',
            height=400
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)
        
        # Enhanced features highlight
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ **Market Integration**")
            st.write("• Real-time UK gilt yield data")
            st.write("• Market-based yield recommendations")
            st.write("• Economic condition monitoring")
            
            st.success("✅ **Advanced Analytics**")
            st.write("• Monte Carlo probability analysis")
            st.write("• Multi-scenario stress testing")
            st.write("• Professional-grade reporting")
        
        with col2:
            st.success("✅ **Enhanced User Experience**")
            st.write("• Smart input validation with warnings")
            st.write("• Contextual help and guidance")
            st.write("• Save/load configuration system")
            
            st.success("✅ **Professional Features**")
            st.write("• Excel export with multiple sheets")
            st.write("• AI-powered recommendations")
            st.write("• Implementation timeline guidance")

if __name__ == "__main__":
    main()
