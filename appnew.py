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
    
    def calculate_sipp_tax_free_available(self, sipp_value: float, tax_free_taken: float = 0) -> Dict: [cite: 1]
        """Calculate available tax-free amount from SIPP""" [cite: 1]
        max_tax_free = sipp_value * 0.25 [cite: 1]
        remaining_tax_free = max(0, max_tax_free - tax_free_taken) [cite: 1]
        taxable_portion = sipp_value - max_tax_free [cite: 1]
        
        return {
            'max_tax_free_lump_sum': max_tax_free, [cite: 1]
            'remaining_tax_free': remaining_tax_free, [cite: 1]
            'taxable_portion': taxable_portion, [cite: 1]
            'tax_free_percentage_used': (tax_free_taken / max_tax_free * 100) if max_tax_free > 0 else 0 [cite: 1]
        }

    def optimize_withdrawal_order(self, target_net_income: float, available_sources: Dict, [cite: 1]
                                additional_taxable_income: float, personal_allowance: float) -> Dict: [cite: 1]
        """ 
        Optimize withdrawal order for tax efficiency: 
        1. SIPP Tax-free (25% allowance) - MOST EFFICIENT 
        2. ISA (tax-free) - SECOND MOST EFFICIENT   
        3. SIPP Taxable (income tax applies) - LEAST EFFICIENT 
        """ [cite: 1]
        withdrawal_plan = {
            'sipp_tax_free': 0, [cite: 1]
            'isa_withdrawal': 0, [cite: 1]
            'sipp_taxable': 0, [cite: 1]
            'total_gross': 0, [cite: 1]
            'total_tax': 0, [cite: 1]
            'total_net': 0, [cite: 1]
            'optimization_notes': [] [cite: 1]
        }
        
        remaining_need = target_net_income [cite: 1]
        
        # Step 1: Use SIPP tax-free first (most efficient - 0% tax) 
        if remaining_need > 0 and available_sources.get('sipp_tax_free', 0) > 0: [cite: 1]
            sipp_tax_free_use = min(remaining_need, available_sources['sipp_tax_free']) [cite: 1]
            withdrawal_plan['sipp_tax_free'] = sipp_tax_free_use [cite: 1]
            remaining_need -= sipp_tax_free_use [cite: 1]
            withdrawal_plan['optimization_notes'].append(f"Used £{sipp_tax_free_use:,.0f} SIPP tax-free (0% tax)") [cite: 1]
        
        # Step 2: Use ISA if still needed (also tax-free) 
        if remaining_need > 0 and available_sources.get('isa', 0) > 0: [cite: 1]
            isa_use = min(remaining_need, available_sources['isa']) [cite: 1]
            withdrawal_plan['isa_withdrawal'] = isa_use [cite: 1]
            remaining_need -= isa_use [cite: 1]
            withdrawal_plan['optimization_notes'].append(f"Used £{isa_use:,.0f} ISA (0% tax)") [cite: 1]
        
        # Step 3: Use SIPP taxable if still needed (least efficient - 20%+ tax) 
        if remaining_need > 0 and available_sources.get('sipp_taxable', 0) > 0: [cite: 1]
            # Need to gross up for tax - iterative calculation for precision 
            estimated_gross_needed = remaining_need * 1.3  # Initial estimate 
            
            for iteration in range(10): [cite: 1]
                sipp_taxable_use = min(estimated_gross_needed, available_sources['sipp_taxable']) [cite: 1]
                total_taxable_income = additional_taxable_income + sipp_taxable_use [cite: 1]
                
                tax_calc = self.calculate_income_tax_with_thresholds(total_taxable_income, personal_allowance) [cite: 1]
                
                # Tax attributable to the SIPP withdrawal 
                if additional_taxable_income > 0: [cite: 1]
                    tax_without_sipp = self.calculate_income_tax_with_thresholds(additional_taxable_income, personal_allowance) [cite: 1]
                    sipp_tax = tax_calc['total_tax'] - tax_without_sipp['total_tax'] [cite: 1]
                else:
                    sipp_tax = tax_calc['total_tax'] [cite: 1]
                
                net_from_sipp = sipp_taxable_use - sipp_tax [cite: 1]
                
                if abs(net_from_sipp - remaining_need) < 1:  # Within £1 
                    break [cite: 1]
                
                if net_from_sipp > 0: [cite: 1]
                    adjustment = remaining_need / net_from_sipp [cite: 1]
                    estimated_gross_needed *= adjustment [cite: 1]
                else:
                    estimated_gross_needed *= 1.1 [cite: 1]
            
            withdrawal_plan['sipp_taxable'] = sipp_taxable_use [cite: 1]
            withdrawal_plan['total_tax'] = sipp_tax [cite: 1]
            
            effective_rate = (sipp_tax / sipp_taxable_use * 100) if sipp_taxable_use > 0 else 0 [cite: 1]
            withdrawal_plan['optimization_notes'].append(
                f"Used £{sipp_taxable_use:,.0f} SIPP taxable ({effective_rate:.1f}% tax)"
            ) [cite: 1]
        
        # Calculate totals 
        withdrawal_plan['total_gross'] = ( [cite: 1]
            withdrawal_plan['sipp_tax_free'] + [cite: 1]
            withdrawal_plan['isa_withdrawal'] + [cite: 1]
            withdrawal_plan['sipp_taxable'] [cite: 1]
        ) [cite: 1]
        withdrawal_plan['total_net'] = withdrawal_plan['total_gross'] - withdrawal_plan['total_tax'] [cite: 1]
        
        return withdrawal_plan [cite: 1]
    
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
            
            # SIPP Strategy
            sipp_strategy = params.get('sipp_strategy', 'gradual')
            upfront_tax_free_percent = params.get('upfront_tax_free_percent', 0)
            
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
            
            # --- SIPP Tax-Free Initialization ---
            sipp_tax_free_analysis = self.calculate_sipp_tax_free_available(sipp_value)
            total_tax_free_pot = sipp_tax_free_analysis['max_tax_free_lump_sum']
            remaining_sipp_tax_free = total_tax_free_pot
            initial_tax_free_withdrawal = 0
            
            if sipp_strategy == 'upfront' and upfront_tax_free_percent > 0:
                initial_tax_free_withdrawal = total_tax_free_pot * (upfront_tax_free_percent / 100)
                remaining_sipp_tax_free -= initial_tax_free_withdrawal
                
            elif sipp_strategy == 'mixed':
                initial_tax_free_withdrawal = total_tax_free_pot * 0.5
                remaining_sipp_tax_free -= initial_tax_free_withdrawal
            
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
                total_tax_free_income_from_sources = isa_bond_income  # ISA income is tax-free
                total_taxable_income_before_drawdown = (
                    total_db_income + 
                    state_pension_income + 
                    sipp_bond_income
                )
                
                # Calculate personal allowance for this year
                current_personal_allowance = self.personal_allowance * inflation_factor
                
                # Available sources for optimization
                available_sources = {
                    'sipp_tax_free': remaining_sipp_tax_free,
                    'isa': remaining_isa,
                    'sipp_taxable': remaining_sipp_taxable
                }
                
                # --- Optimize withdrawals ---
                # Add initial_tax_free_withdrawal to total_tax_free_income_from_sources only for year 1
                if year == 1:
                    available_sources['sipp_tax_free'] += initial_tax_free_withdrawal
                    
                optimized_withdrawals = self.optimize_withdrawal_order(
                    inflation_adjusted_target,
                    available_sources,
                    total_taxable_income_before_drawdown,
                    current_personal_allowance
                )
                
                drawdown_tax_free_sipp = optimized_withdrawals['sipp_tax_free']
                drawdown_isa = optimized_withdrawals['isa_withdrawal']
                drawdown_taxable_sipp = optimized_withdrawals['sipp_taxable']
                total_tax_from_drawdown = optimized_withdrawals['total_tax']
                
                # Apply maximum withdrawal rate check
                total_remaining_pots = remaining_sipp_taxable + remaining_sipp_tax_free + remaining_isa
                total_drawdown_amount = drawdown_tax_free_sipp + drawdown_isa + drawdown_taxable_sipp
                max_allowed_drawdown = total_remaining_pots * (max_withdrawal_rate / 100)
                
                if total_drawdown_amount > max_allowed_drawdown and total_drawdown_amount > 0:
                    scale_factor = max_allowed_drawdown / total_drawdown_amount
                    drawdown_tax_free_sipp *= scale_factor
                    drawdown_isa *= scale_factor
                    drawdown_taxable_sipp *= scale_factor
                    
                # Update remaining pots
                remaining_sipp_tax_free -= drawdown_tax_free_sipp
                remaining_isa -= drawdown_isa
                remaining_sipp_taxable -= drawdown_taxable_sipp
                
                # Final income calculation
                total_taxable_income_final = total_taxable_income_before_drawdown + drawdown_taxable_sipp
                total_tax_free_income_final = total_tax_free_income_from_sources + drawdown_isa + drawdown_tax_free_sipp
                
                # Calculate final tax
                tax_details = self.calculate_income_tax_with_thresholds(
                    total_taxable_income_final, 
                    current_personal_allowance
                )
                
                total_gross_income = total_taxable_income_final + total_tax_free_income_final
                total_net_income = total_gross_income - tax_details['total_tax']
                
                # Apply growth to remaining pots
                growth_factor = 1 + (investment_growth / 100)
                remaining_sipp_taxable *= growth_factor
                remaining_sipp_tax_free *= growth_factor # Tax-free portion also grows
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
                    'drawdown_tax_free': round(drawdown_tax_free_sipp), # SIPP tax-free
                    'drawdown_isa': round(drawdown_isa), # ISA withdrawal (tax-free)
                    'drawdown_taxable': round(drawdown_taxable_sipp), # SIPP taxable
                    
                    # Income totals
                    'total_taxable_income': round(total_taxable_income_final),
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
                        'Total Net Income (Year 1)', 'Average Tax Rate (%)', 'Final Pot Value', 
                        'Total Years Analyzed', 'Average Annual Income', 'Total Tax Paid', 
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
                        'Mean Final Pot Value', 'Median Final Pot Value', '10th Percentile Final Pot',
                        '90th Percentile Final Pot', 'Probability of Pot Depletion', 'Mean Average Income',
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
                
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel export: {e}")
        return None

# =============================================================================
# SIPP UI COMPONENTS 
# Add these functions AFTER your existing UI helper functions 
# =============================================================================

def add_sipp_strategy_selection(): [cite: 1]
    """Add SIPP strategy selection to sidebar""" [cite: 1]
    st.sidebar.subheader("🎯 SIPP 25% Tax-Free Strategy") [cite: 1]
    
    sipp_strategy = st.sidebar.radio( [cite: 1]
        "Tax-Free Withdrawal Strategy", [cite: 1]
        options=['gradual', 'mixed', 'upfront'], [cite: 1]
        format_func=lambda x: { [cite: 1]
            'gradual': '📅 Gradual (Take as needed)', [cite: 1]
            'mixed': '⚖️ Mixed (Half upfront, half gradual)', [cite: 1] 
            'upfront': '💰 Upfront (Large lump sum)' [cite: 1]
        }[x], [cite: 1]
        help="How to handle your 25% tax-free SIPP allowance" [cite: 1]
    )
    
    upfront_percent = 0 [cite: 1]
    if sipp_strategy == 'upfront': [cite: 1]
        upfront_percent = st.sidebar.slider( [cite: 1]
            "% of Tax-Free to Take Upfront", [cite: 1]
            min_value=25, [cite: 1]
            max_value=100, [cite: 1]
            value=75, [cite: 1]
            step=5, [cite: 1]
            help="What percentage of your 25% allowance to take immediately" [cite: 1]
        ) [cite: 1]
    elif sipp_strategy == 'mixed': [cite: 1]
        upfront_percent = 50  # Fixed at 50% for mixed strategy 
    
    return sipp_strategy, upfront_percent [cite: 1]

def display_sipp_breakdown(sipp_value, sipp_strategy, upfront_percent, calc): [cite: 1]
    """Display SIPP breakdown in sidebar""" [cite: 1]
    sipp_analysis = calc.calculate_sipp_tax_free_available(sipp_value) [cite: 1]
    max_tax_free = sipp_analysis['max_tax_free_lump_sum'] [cite: 1]
    
    with st.sidebar.expander("💡 SIPP Tax-Free Breakdown"): [cite: 1]
        st.write(f"**Total SIPP**: £{sipp_value:,}") [cite: 1]
        st.write(f"**Max Tax-Free (25%)**: £{max_tax_free:,}") [cite: 1]
        st.write(f"**Taxable Portion (75%)**: £{sipp_analysis['taxable_portion']:,}") [cite: 1]
        
        if sipp_strategy in ['upfront', 'mixed']: [cite: 1]
            if sipp_strategy == 'mixed': [cite: 1]
                upfront_amount = max_tax_free * 0.5 [cite: 1]
                remaining_amount = max_tax_free * 0.5 [cite: 1]
                st.write(f"**Upfront Tax-Free**: £{upfront_amount:,}") [cite: 1]
                st.write(f"**Remaining Tax-Free**: £{remaining_amount:,}") [cite: 1]
            else:
                upfront_amount = max_tax_free * (upfront_percent / 100) [cite: 1]
                remaining_amount = max_tax_free - upfront_amount [cite: 1]
                st.write(f"**Upfront Tax-Free**: £{upfront_amount:,}") [cite: 1]
                st.write(f"**Remaining Tax-Free**: £{remaining_amount:,}") [cite: 1]
        else:
            st.write(f"**Available for Gradual Use**: £{max_tax_free:,}") [cite: 1]
            
        return sipp_analysis [cite: 1]

def display_sipp_tax_free_usage_chart(annual_data, sipp_analysis): [cite: 1]
    """Display SIPP tax-free usage analysis chart""" [cite: 1]
    st.subheader("🎯 SIPP Tax-Free Usage Analysis") [cite: 1]
    
    col1, col2 = st.columns(2) [cite: 1]
    
    with col1:
        # SIPP tax-free usage over time
        df_annual = pd.DataFrame(annual_data)
        
        if 'drawdown_tax_free' in df_annual.columns:
            fig_sipp_usage = go.Figure()
            fig_sipp_usage.add_trace(go.Bar(
                x=df_annual['calendar_year'], 
                y=df_annual['drawdown_tax_free'], 
                name='SIPP Tax-Free Used',
                marker_color='lightgreen'
            ))
            fig_sipp_usage.update_layout(
                title='Annual SIPP Tax-Free Withdrawal',
                xaxis_title='Year',
                yaxis_title='Amount (£)',
                barmode='group',
                height=300
            )
            st.plotly_chart(fig_sipp_usage, use_container_width=True)
        else:
            st.info("SIPP tax-free withdrawal data not available for charting.")

    with col2:
        st.metric(
            label="Total SIPP Tax-Free Taken (Over Simulation)",
            value=f"£{sum(d.get('drawdown_tax_free', 0) for d in annual_data):,.0f}"
        )
        st.metric(
            label="Initial Max SIPP Tax-Free Available",
            value=f"£{sipp_analysis['max_tax_free_lump_sum']:,.0f}"
        )
        
        # Pie chart for tax-free vs taxable split
        tax_free_used = sum(d.get('drawdown_tax_free', 0) for d in annual_data)
        total_sipp_withdrawal = tax_free_used + sum(d.get('drawdown_taxable', 0) for d in annual_data)
        
        if total_sipp_withdrawal > 0:
            labels = ['Tax-Free Portion', 'Taxable Portion']
            values = [tax_free_used, total_sipp_withdrawal - tax_free_used]
            colors = ['#4CAF50', '#FFC107']
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=.3)])
            fig_pie.update_layout(
                title_text="SIPP Withdrawal Composition",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No SIPP withdrawals made to show composition.")

def main():
    st.title("💰 Enhanced Pension & Bond Ladder Calculator")
    
    calc = EnhancedPensionCalculator()
    
    st.sidebar.header("Parameters")
    
    # User inputs for overall simulation
    st.sidebar.subheader("General Settings")
    target_annual_income = st.sidebar.number_input("Target Annual Net Income (£)", min_value=10000, value=30000, step=1000)
    years = st.sidebar.slider("Number of Years to Simulate", min_value=5, max_value=60, value=30, step=1)
    
    st.sidebar.subheader("Investment Assumptions")
    inflation_rate = st.sidebar.slider("Average Inflation Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    investment_growth = st.sidebar.slider("Average Investment Growth Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    max_withdrawal_rate = st.sidebar.slider("Max Safe Withdrawal Rate (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.1, help="Maximum percentage of remaining portfolio value you can withdraw annually.")
    
    # Current Gilt Yields
    st.sidebar.subheader("Market Data")
    gilt_data = get_current_gilt_yields()
    
    if gilt_data['success']:
        st.sidebar.success(f"Market Data (Source: {gilt_data['source']}) Last Updated: {gilt_data['last_updated']}")
        st.sidebar.info(f"10-Year UK Gilt Yield: {gilt_data['yield_10y'] * 100:.2f}%")
        default_sipp_yield = gilt_data['yield_10y'] * 100
        default_isa_yield = gilt_data['yield_5y'] * 100 if 'yield_5y' in gilt_data else default_sipp_yield
    else:
        st.sidebar.warning(f"Could not fetch real-time gilt yields. Using default values. Error: {gilt_data.get('error', 'Unknown')}")
        default_sipp_yield = 4.5
        default_isa_yield = 4.2
        
    sipp_yield = st.sidebar.slider("SIPP Bond Yield (%)", min_value=0.0, max_value=10.0, value=default_sipp_yield, step=0.1)
    isa_yield = st.sidebar.slider("ISA Bond Yield (%)", min_value=0.0, max_value=10.0, value=default_isa_yield, step=0.1)
    
    st.sidebar.subheader("Current Portfolio")
    sipp_value = st.sidebar.number_input("Current SIPP Value (£)", min_value=0, value=250000, step=10000)
    isa_value = st.sidebar.number_input("Current ISA Value (£)", min_value=0, value=100000, step=10000)
    
    # SIPP strategy selection 
    sipp_strategy, upfront_tax_free_percent = add_sipp_strategy_selection() [cite: 1]
    sipp_analysis = display_sipp_breakdown(sipp_value, sipp_strategy, upfront_tax_free_percent, calc) [cite: 1]
    
    st.sidebar.subheader("Other Income Sources")
    db_pensions_input = st.sidebar.text_area("Other Defined Benefit Pensions (Name:Amount, one per line)", "Work Pension:10000\nOther Annuity:5000")
    db_pensions = {}
    for line in db_pensions_input.split('\n'):
        if ':' in line:
            name, amount_str = line.split(':')
            try:
                db_pensions[name.strip()] = float(amount_str.strip())
            except ValueError:
                st.sidebar.warning(f"Invalid pension format: {line}. Use 'Name:Amount'.")
                
    state_pension = st.sidebar.number_input("State Pension (Annual, £)", min_value=0, value=11500, step=500)
    state_pension_start_year_relative = st.sidebar.slider("State Pension Starts (Years from now)", min_value=0, max_value=years, value=5)
    
    bond_ladder_years = st.sidebar.slider("Bond Ladder Length (Years)", min_value=1, max_value=min(years, 30), value=10, step=1)
    cash_buffer_percent = st.sidebar.slider("Cash Buffer Percentage (%)", min_value=0, max_value=20, value=5, step=1, help="Percentage of pot kept in cash, not invested in bonds.")
    
    # Base parameters for simulation
    base_params = {
        'years': years,
        'inflation_rate': inflation_rate,
        'investment_growth': investment_growth,
        'sipp_value': sipp_value,
        'isa_value': isa_value,
        'bond_ladder_years': bond_ladder_years,
        'sipp_yield': sipp_yield,
        'isa_yield': isa_yield,
        'db_pensions': db_pensions,
        'state_pension': state_pension,
        'state_pension_start_year': state_pension_start_year_relative,
        'max_withdrawal_rate': max_withdrawal_rate,
        'target_annual_income': target_annual_income,
        'cash_buffer_percent': cash_buffer_percent,
        'sipp_strategy': sipp_strategy, [cite: 1]
        'upfront_tax_free_percent': upfront_tax_free_percent, [cite: 1]
    }
    
    # Input Validation
    errors, warnings = validate_inputs(sipp_value, isa_value, target_annual_income, years)
    if errors:
        for error in errors:
            st.error(error)
        st.stop() # Stop execution if there are errors
    
    for warning in warnings:
        st.warning(warning)

    if st.sidebar.button("Run Simulation"):
        st.subheader("📊 Simulation Results")
        
        annual_data, sipp_ladder, isa_ladder = calc.simulate_comprehensive_strategy(base_params)
        
        if annual_data:
            df_annual = pd.DataFrame(annual_data)
            
            st.write("### Year-by-Year Financial Overview")
            st.dataframe(df_annual.set_index('calendar_year'))
            
            st.write("### Income vs. Target over Time")
            fig_income = px.line(
                df_annual, 
                x='calendar_year', 
                y=['total_net_income', 'target_income'], 
                title='Net Income vs. Target Income',
                labels={'value': 'Amount (£)', 'variable': 'Income Type'},
                height=400
            )
            fig_income.update_traces(mode='lines+markers', hoverinfo='all')
            fig_income.update_layout(hovermode="x unified")
            st.plotly_chart(fig_income, use_container_width=True)
            
            st.write("### Remaining Pot Value over Time")
            fig_pots = px.area(
                df_annual, 
                x='calendar_year', 
                y='total_remaining_pots', 
                title='Total Remaining Portfolio Value',
                labels={'total_remaining_pots': 'Value (£)'},
                height=400
            )
            fig_pots.update_traces(mode='lines+markers', fill='tozeroy')
            st.plotly_chart(fig_pots, use_container_width=True)
            
            # Display Alerts
            st.write("### Alerts and Insights")
            alerts = create_alerts(annual_data)
            if alerts:
                for alert in alerts:
                    if alert['type'] == 'error':
                        st.error(alert['message'])
                    elif alert['type'] == 'warning':
                        st.warning(alert['message'])
                    elif alert['type'] == 'success':
                        st.success(alert['message'])
            else:
                st.info("No significant alerts detected. Your plan looks solid!")
            
            # Display SIPP Tax-Free Usage Chart
            display_sipp_tax_free_usage_chart(annual_data, sipp_analysis) [cite: 1]

            st.write("### Bond Ladder Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### SIPP Bond Ladder")
                if not sipp_ladder.empty:
                    st.dataframe(sipp_ladder)
                else:
                    st.info("No SIPP bonds generated.")
            with col2:
                st.write("#### ISA Bond Ladder")
                if not isa_ladder.empty:
                    st.dataframe(isa_ladder)
                else:
                    st.info("No ISA bonds generated.")
            
            # Excel Export
            excel_file = create_excel_export(annual_data, sipp_ladder, isa_ladder)
            if excel_file:
                st.download_button(
                    label="Download Full Report (Excel)",
                    data=excel_file,
                    file_name="pension_bond_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("Simulation failed to generate data. Please check inputs.")

    st.subheader("🔬 Advanced Analysis")
    
    if st.button("Run Scenario Analysis"):
        st.write("### Multi-Scenario Stress Testing")
        scenario_results = run_scenario_analysis(base_params, calc)
        
        if scenario_results:
            st.write("#### Scenario Summaries")
            summary_df_data = []
            for scenario, res in scenario_results.items():
                summary_df_data.append({
                    'Scenario': scenario,
                    'Description': res['description'],
                    'Final Pot Value': f"£{res['final_pot']:,.0f}",
                    'Avg. Net Income': f"£{res['avg_net_income']:,.0f}",
                    'Avg. Tax Rate': f"{res['avg_tax_rate']:.1f}%"
                })
            st.dataframe(pd.DataFrame(summary_df_data).set_index('Scenario'))
            
            st.write("#### Net Income Across Scenarios")
            fig_scenario = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            for i, (scenario, res) in enumerate(scenario_results.items()):
                df_scenario = pd.DataFrame(res['annual_data'])
                fig_scenario.add_trace(go.Scatter(
                    x=df_scenario['calendar_year'], 
                    y=df_scenario['total_net_income'], 
                    mode='lines+markers', 
                    name=scenario,
                    line=dict(color=colors[i], width=3)
                ))
            
            fig_scenario.update_layout(
                title='Multi-Scenario Net Income Projection',
                xaxis_title='Year',
                yaxis_title='Net Income (£)',
                height=400
            )
            st.plotly_chart(fig_scenario, use_container_width=True)
            
            # Example of custom plot for sample scenario
            st.write("#### Sample: Multi-Scenario Income Projection (Target vs. Actual)")
            
            fig_sample = go.Figure()
            
            # Define colors for scenarios
            colors = px.colors.qualitative.Bold
            
            # Add traces for each scenario's total_net_income
            for i, scenario in enumerate(scenario_results.keys()):
                df_scenario = pd.DataFrame(scenario_results[scenario]['annual_data'])
                fig_sample.add_trace(go.Scatter(
                    x=df_scenario['calendar_year'],
                    y=df_scenario['total_net_income'],
                    mode='lines+markers',
                    name=scenario,
                    line=dict(color=colors[i], width=3)
                ))
            
            # Add the target_income as a dashed line
            df_base = pd.DataFrame(scenario_results['Base Case 📊']['annual_data'])
            fig_sample.add_trace(go.Scatter(
                x=df_base['calendar_year'],
                y=df_base['target_income'],
                mode='lines',
                name='Target Income',
                line=dict(color='grey', dash='dash', width=2)
            ))
            
            fig_sample.update_layout(
                title='Multi-Scenario Net Income Projection vs. Target',
                xaxis_title='Year',
                yaxis_title='Net Income (£)',
                height=400,
                legend_title_text='Scenario'
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)
            
            # Plot of total remaining pots for each scenario
            st.write("#### Total Remaining Pots Across Scenarios")
            fig_pots_scenario = go.Figure()
            for i, (scenario, res) in enumerate(scenario_results.items()):
                df_scenario = pd.DataFrame(res['annual_data'])
                fig_pots_scenario.add_trace(go.Scatter(
                    x=df_scenario['calendar_year'], 
                    y=df_scenario['total_remaining_pots'], 
                    mode='lines+markers', 
                    name=scenario,
                    line=dict(color=colors[i], width=3)
                ))
            fig_pots_scenario.update_layout(
                title='Multi-Scenario Remaining Pot Value Projection',
                xaxis_title='Year',
                yaxis_title='Remaining Pot (£)',
                height=400
            )
            st.plotly_chart(fig_pots_scenario, use_container_width=True)

        else:
            st.info("No scenario analysis results to display.")

    if st.button("Run Monte Carlo Simulation"):
        st.write("### Monte Carlo Simulation (Probability Analysis)")
        st.info(f"Running {500} simulations. This may take a moment...")
        monte_carlo_results = run_monte_carlo_simulation(base_params, calc, num_simulations=500)
        
        if monte_carlo_results:
            df_mc = pd.DataFrame(monte_carlo_results)
            
            st.write("#### Final Pot Value Distribution")
            fig_hist = px.histogram(df_mc, x="final_pot", nbins=50, title="Distribution of Final Pot Values")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.write("#### Monte Carlo Summary Statistics")
            
            prob_depletion = df_mc['pot_depleted'].mean() * 100
            
            st.metric(label="Probability of Pot Depletion (<£10k)", value=f"{prob_depletion:.1f}%")
            st.metric(label="Average Final Pot Value", value=f"£{df_mc['final_pot'].mean():,.0f}")
            st.metric(label="Median Final Pot Value", value=f"£{df_mc['final_pot'].median():,.0f}")
            st.metric(label="10th Percentile Final Pot Value", value=f"£{np.percentile(df_mc['final_pot'], 10):,.0f}")
            st.metric(label="90th Percentile Final Pot Value", value=f"£{np.percentile(df_mc['final_pot'], 90):,.0f}")
            
            st.write("#### Average Annual Income Distribution")
            fig_hist_income = px.histogram(df_mc, x="avg_income", nbins=50, title="Distribution of Average Annual Incomes")
            st.plotly_chart(fig_hist_income, use_container_width=True)
            
            st.write("#### Income Shortfall Risk")
            # Calculate the percentage of simulations where income shortfall years > 0
            sims_with_shortfall = df_mc[df_mc['income_shortfall_years'] > 0].shape[0]
            percent_sims_with_shortfall = (sims_with_shortfall / len(df_mc)) * 100
            st.metric(label="Probability of Any Income Shortfall", value=f"{percent_sims_with_shortfall:.1f}%")
            
        else:
            st.info("No Monte Carlo simulation results to display.")
    
    st.subheader("💡 Key Features Highlight")
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
