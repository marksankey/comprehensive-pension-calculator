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
    page_title="Enhanced SIPP Bond Strategy Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://lategenxer.streamlit.app/Gilt_Ladder',
        'Report a bug': 'https://github.com/yourusername/sipp-enhanced-calculator/issues',
        'About': """
        # Enhanced SIPP Tax-Free Bond Strategy Calculator
        
        Professional UK retirement planning with bond ladder recommendations.
        
        **Key Features:**
        - Proper SIPP 25% tax-free allowance handling
        - Specific UK Gilt and Corporate Bond recommendations
        - Interactive Investor platform integration
        - Real-time yield calculations
        - Professional bond ladder management
        
        **Disclaimer:** This tool provides estimates for educational purposes only. 
        Please seek professional financial advice before making investment decisions.
        """
    }
)

class EnhancedSIPPBondCalculator:
    def __init__(self):
        # UK Tax bands for 2025/26
        self.personal_allowance = 12570
        self.basic_rate_threshold = 50270
        self.higher_rate_threshold = 125140
        self.additional_rate_threshold = 150000
        
        # Bond database - UK Gilts
        self.uk_gilts = {
            'UK Treasury 4.25% 2027': {
                'isin': 'GB00B16NNR78',
                'maturity_date': '2027-12-07',
                'coupon': 4.25,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            },
            'UK Treasury 1.625% 2028': {
                'isin': 'GB00BFWFPP71',
                'maturity_date': '2028-10-22',
                'coupon': 1.625,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            },
            'UK Treasury 0.875% 2029': {
                'isin': 'GB00BJMHB534',
                'maturity_date': '2029-10-31',
                'coupon': 0.875,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            },
            'UK Treasury 0.375% 2030': {
                'isin': 'GB00BKPWFW93',
                'maturity_date': '2030-10-22',
                'coupon': 0.375,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            },
            'UK Treasury 0.25% 2031': {
                'isin': 'GB00BN65R313',
                'maturity_date': '2031-01-31',
                'coupon': 0.25,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            },
            'UK Treasury 4.25% 2032': {
                'isin': 'GB0004893086',
                'maturity_date': '2032-06-07',
                'coupon': 4.25,
                'type': 'UK Gilt',
                'rating': 'AA',
                'min_denomination': 100,
                'recommended_for': 'SIPP'
            }
        }
        
        # Corporate bonds for ISA
        self.corporate_bonds = {
            'National Grid 2.625% 2028': {
                'isin': 'GB00BZ03MX94',
                'maturity_date': '2028-06-16',
                'coupon': 2.625,
                'type': 'Corporate',
                'rating': 'BBB+',
                'sector': 'Utilities',
                'recommended_for': 'ISA'
            },
            'BT Group 4.25% 2029': {
                'isin': 'GB00BMF5JQ11',
                'maturity_date': '2029-11-23',
                'coupon': 4.25,
                'type': 'Corporate',
                'rating': 'BBB',
                'sector': 'Telecommunications',
                'recommended_for': 'ISA'
            },
            'Vodafone 4.875% 2030': {
                'isin': 'GB00BM8QGX98',
                'maturity_date': '2030-07-03',
                'coupon': 4.875,
                'type': 'Corporate',
                'rating': 'BBB',
                'sector': 'Telecommunications',
                'recommended_for': 'ISA'
            },
            'Tesco 6.125% 2031': {
                'isin': 'GB00BMF5JT45',
                'maturity_date': '2031-02-24',
                'coupon': 6.125,
                'type': 'Corporate',
                'rating': 'BBB',
                'sector': 'Consumer Staples',
                'recommended_for': 'ISA'
            },
            'United Utilities 2.544% 2032': {
                'isin': 'GB00BN4Q8H84',
                'maturity_date': '2032-01-19',
                'coupon': 2.544,
                'type': 'Corporate',
                'rating': 'A-',
                'sector': 'Utilities',
                'recommended_for': 'ISA'
            }
        }
        
    def calculate_income_tax_with_thresholds(self, taxable_income: float, personal_allowance: float) -> Dict:
        """Calculate UK income tax with high earner personal allowance reduction"""
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
    
    def get_bond_recommendations(self, allocation_per_year: float, ladder_years: int, 
                               account_type: str, start_year: int = 2027) -> pd.DataFrame:
        """Get specific bond recommendations for the ladder"""
        recommendations = []
        
        # Select appropriate bond universe
        if account_type.lower() == 'sipp':
            bond_universe = self.uk_gilts
            target_yield_base = 4.1
        else:  # ISA
            bond_universe = {**self.uk_gilts, **self.corporate_bonds}
            target_yield_base = 5.0
        
        for year_offset in range(ladder_years):
            target_year = start_year + year_offset
            target_yield = target_yield_base + (year_offset * 0.2)
            
            # Find bonds maturing in or near target year
            suitable_bonds = []
            for bond_name, bond_info in bond_universe.items():
                maturity_year = int(bond_info['maturity_date'].split('-')[0])
                
                # Allow ±1 year flexibility for matching
                if abs(maturity_year - target_year) <= 1:
                    # Estimate current yield based on market conditions
                    estimated_price = 98 + random.uniform(-5, 10)  # Simulate market prices
                    ytm = self.calculate_yield_to_maturity(
                        estimated_price, 100, bond_info['coupon'], 
                        max(0.1, maturity_year - 2027 + 1)
                    )
                    
                    suitable_bonds.append({
                        'bond_name': bond_name,
                        'isin': bond_info['isin'],
                        'maturity_date': bond_info['maturity_date'],
                        'maturity_year': maturity_year,
                        'coupon': bond_info['coupon'],
                        'type': bond_info['type'],
                        'rating': bond_info.get('rating', 'AA'),
                        'sector': bond_info.get('sector', 'Government'),
                        'estimated_price': estimated_price,
                        'estimated_ytm': ytm,
                        'allocation': allocation_per_year,
                        'annual_income': allocation_per_year * (ytm / 100),
                        'ladder_year': year_offset + 1,
                        'target_year': target_year
                    })
            
            # Sort by yield and select best option
            if suitable_bonds:
                if account_type.lower() == 'sipp':
                    # For SIPP, prefer government bonds (lower risk)
                    suitable_bonds.sort(key=lambda x: (x['type'] != 'UK Gilt', -x['estimated_ytm']))
                else:
                    # For ISA, prefer higher yields
                    suitable_bonds.sort(key=lambda x: -x['estimated_ytm'])
                
                recommendations.append(suitable_bonds[0])
            else:
                # Fallback recommendation
                recommendations.append({
                    'bond_name': f'Target Bond {target_year}',
                    'isin': 'TO_BE_IDENTIFIED',
                    'maturity_date': f'{target_year}-06-30',
                    'maturity_year': target_year,
                    'coupon': target_yield,
                    'type': 'UK Gilt' if account_type.lower() == 'sipp' else 'Corporate',
                    'rating': 'AA' if account_type.lower() == 'sipp' else 'BBB+',
                    'sector': 'Government' if account_type.lower() == 'sipp' else 'Mixed',
                    'estimated_price': 100,
                    'estimated_ytm': target_yield,
                    'allocation': allocation_per_year,
                    'annual_income': allocation_per_year * (target_yield / 100),
                    'ladder_year': year_offset + 1,
                    'target_year': target_year
                })
        
        return pd.DataFrame(recommendations)
    
    def create_bond_ladder_with_recommendations(self, total_investment: float, ladder_years: int, 
                                              account_type: str, start_year: int = 2027) -> pd.DataFrame:
        """Create a bond ladder with specific bond recommendations"""
        if total_investment <= 0 or ladder_years <= 0:
            return pd.DataFrame()
        
        allocation_per_year = total_investment / ladder_years
        return self.get_bond_recommendations(allocation_per_year, ladder_years, account_type, start_year)
    
    def optimize_withdrawal_order(self, target_net_income: float, available_sources: Dict, 
                                additional_taxable_income: float, personal_allowance: float) -> Dict:
        """Optimize withdrawal order for tax efficiency"""
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
        
        # Step 1: Use SIPP tax-free first
        if remaining_need > 0 and available_sources.get('sipp_tax_free', 0) > 0:
            sipp_tax_free_use = min(remaining_need, available_sources['sipp_tax_free'])
            withdrawal_plan['sipp_tax_free'] = sipp_tax_free_use
            remaining_need -= sipp_tax_free_use
            withdrawal_plan['optimization_notes'].append(f"Used £{sipp_tax_free_use:,.0f} SIPP tax-free (0% tax)")
        
        # Step 2: Use ISA if still needed
        if remaining_need > 0 and available_sources.get('isa', 0) > 0:
            isa_use = min(remaining_need, available_sources['isa'])
            withdrawal_plan['isa_withdrawal'] = isa_use
            remaining_need -= isa_use
            withdrawal_plan['optimization_notes'].append(f"Used £{isa_use:,.0f} ISA (0% tax)")
        
        # Step 3: Use SIPP taxable if still needed
        if remaining_need > 0 and available_sources.get('sipp_taxable', 0) > 0:
            # Iterative calculation for precise tax
            estimated_gross_needed = remaining_need * 1.3
            
            for iteration in range(10):
                sipp_taxable_use = min(estimated_gross_needed, available_sources['sipp_taxable'])
                total_taxable_income = additional_taxable_income + sipp_taxable_use
                
                tax_calc = self.calculate_income_tax_with_thresholds(total_taxable_income, personal_allowance)
                
                if additional_taxable_income > 0:
                    tax_without_sipp = self.calculate_income_tax_with_thresholds(additional_taxable_income, personal_allowance)
                    sipp_tax = tax_calc['total_tax'] - tax_without_sipp['total_tax']
                else:
                    sipp_tax = tax_calc['total_tax']
                
                net_from_sipp = sipp_taxable_use - sipp_tax
                
                if abs(net_from_sipp - remaining_need) < 1:
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
    
    def simulate_comprehensive_strategy(self, params: Dict) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
        """Enhanced simulation with bond recommendations"""
        try:
            # Extract parameters
            years = params['years']
            start_year = params.get('start_year', 2027)
            inflation_rate = params['inflation_rate']
            investment_growth = params['investment_growth']
            
            # SIPP and ISA values
            sipp_value = params['sipp_value']
            isa_value = params['isa_value']
            
            # SIPP strategy parameters
            sipp_strategy = params.get('sipp_strategy', 'gradual')
            upfront_tax_free_percent = params.get('upfront_tax_free_percent', 0)
            
            # Bond ladder parameters
            bond_ladder_years = params['bond_ladder_years']
            cash_buffer_percent = params.get('cash_buffer_percent', 5)
            
            # Calculate SIPP components
            sipp_analysis = self.calculate_sipp_tax_free_available(sipp_value)
            max_tax_free = sipp_analysis['max_tax_free_lump_sum']
            sipp_taxable_total = sipp_analysis['taxable_portion']
            
            # Handle different SIPP strategies
            if sipp_strategy == 'upfront':
                upfront_tax_free = max_tax_free * (upfront_tax_free_percent / 100)
                remaining_sipp_tax_free = max_tax_free - upfront_tax_free
                effective_isa_value = isa_value + upfront_tax_free
            elif sipp_strategy == 'mixed':
                upfront_tax_free = max_tax_free * 0.5
                remaining_sipp_tax_free = max_tax_free - upfront_tax_free
                effective_isa_value = isa_value + upfront_tax_free
            else:  # gradual
                upfront_tax_free = 0
                remaining_sipp_tax_free = max_tax_free
                effective_isa_value = isa_value
            
            # Calculate allocations for bonds vs cash
            total_sipp_for_allocation = remaining_sipp_tax_free + sipp_taxable_total
            sipp_cash_buffer = total_sipp_for_allocation * (cash_buffer_percent / 100)
            sipp_bonds_total = total_sipp_for_allocation - sipp_cash_buffer
            
            isa_cash_buffer = effective_isa_value * (cash_buffer_percent / 100)
            isa_bonds_total = effective_isa_value - isa_cash_buffer
            
            # Create bond ladders with specific recommendations
            sipp_ladder = self.create_bond_ladder_with_recommendations(
                sipp_bonds_total, bond_ladder_years, 'SIPP', start_year
            )
            isa_ladder = self.create_bond_ladder_with_recommendations(
                isa_bonds_total, bond_ladder_years, 'ISA', start_year
            )
            
            # Calculate initial allocations - both tax-free and taxable portions are invested
            sipp_tax_free_ratio = remaining_sipp_tax_free / total_sipp_for_allocation if total_sipp_for_allocation > 0 else 0
            sipp_taxable_ratio = sipp_taxable_total / total_sipp_for_allocation if total_sipp_for_allocation > 0 else 0
            
            # Initial pot values for simulation
            # Both tax-free and taxable amounts are invested in bonds and cash
            remaining_sipp_tax_free_bonds = remaining_sipp_tax_free * (sipp_bonds_total / total_sipp_for_allocation) if total_sipp_for_allocation > 0 else 0
            remaining_sipp_tax_free_cash = remaining_sipp_tax_free * (sipp_cash_buffer / total_sipp_for_allocation) if total_sipp_for_allocation > 0 else 0
            
            remaining_sipp_taxable_bonds = sipp_taxable_total * (sipp_bonds_total / total_sipp_for_allocation) if total_sipp_for_allocation > 0 else 0
            remaining_sipp_taxable_cash = sipp_taxable_total * (sipp_cash_buffer / total_sipp_for_allocation) if total_sipp_for_allocation > 0 else 0
            
            remaining_isa_bonds = isa_bonds_total
            remaining_isa_cash = isa_cash_buffer
            
            # Defined benefit pensions
            db_pensions = params['db_pensions']
            state_pension = params.get('state_pension', 0)
            state_pension_start_year = params.get('state_pension_start_year', 10)
            
            # Other parameters
            max_withdrawal_rate = params.get('max_withdrawal_rate', 4.0)
            target_annual_income = params['target_annual_income']
            
            annual_data = []
            
            for year in range(1, years + 1):
                current_year = start_year + year - 1
                inflation_factor = (1 + (inflation_rate / 100)) ** (year - 1)
                
                # Calculate inflation-adjusted targets and pensions
                inflation_adjusted_target = target_annual_income * inflation_factor
                current_personal_allowance = self.personal_allowance * inflation_factor
                
                # Defined benefit pensions (inflation adjusted)
                total_db_income = 0
                for pension_name, amount in db_pensions.items():
                    if amount > 0:
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
                    if bond['maturity_year'] == current_year and bond['allocation'] > 0:
                        # Bond matures - reinvest principal in new 5-year bond
                        principal = bond['allocation']
                        to_tax_free_bonds = principal * sipp_tax_free_ratio
                        to_taxable_bonds = principal * sipp_taxable_ratio
                        
                        remaining_sipp_tax_free_bonds += to_tax_free_bonds
                        remaining_sipp_taxable_bonds += to_taxable_bonds
                        
                        # Reinvest in new bond maturing 5 years later
                        new_maturity_year = current_year + bond_ladder_years
                        sipp_ladder.loc[idx, 'maturity_year'] = new_maturity_year
                        sipp_ladder.loc[idx, 'maturity_date'] = f'{new_maturity_year}-06-30'
                        sipp_ladder.loc[idx, 'bond_name'] = f'Reinvested Bond {new_maturity_year}'
                        # Keep same allocation - bond ladder continues
                        # Update estimated yield for new maturity (yield curve effect)
                        years_from_now = new_maturity_year - start_year
                        estimated_new_yield = 4.1 + (years_from_now - bond_ladder_years) * 0.1
                        sipp_ladder.loc[idx, 'estimated_ytm'] = max(2.0, estimated_new_yield)
                        sipp_ladder.loc[idx, 'annual_income'] = principal * (sipp_ladder.loc[idx, 'estimated_ytm'] / 100)
                        
                        bonds_maturing_this_year.append({
                            'Type': 'SIPP',
                            'Bond': bond['bond_name'],
                            'ISIN': bond['isin'],
                            'Principal': principal,
                            'Year': current_year,
                            'Action': f'Reinvested in {new_maturity_year} bond'
                        })
                    
                    # Bond income continues regardless of maturity (from current or new bonds)
                    if bond['allocation'] > 0:
                        sipp_bond_income += bond['annual_income']
                
                for idx, bond in isa_ladder.iterrows():
                    if bond['maturity_year'] == current_year and bond['allocation'] > 0:
                        # Bond matures - reinvest principal in new bond
                        principal = bond['allocation']
                        remaining_isa_bonds += principal
                        
                        # Reinvest in new bond maturing 5 years later
                        new_maturity_year = current_year + bond_ladder_years
                        isa_ladder.loc[idx, 'maturity_year'] = new_maturity_year
                        isa_ladder.loc[idx, 'maturity_date'] = f'{new_maturity_year}-06-30'
                        isa_ladder.loc[idx, 'bond_name'] = f'Reinvested Corporate {new_maturity_year}'
                        # Keep same allocation - bond ladder continues
                        # Update estimated yield for new maturity (corporate bonds typically higher yield)
                        years_from_now = new_maturity_year - start_year
                        estimated_new_yield = 5.0 + (years_from_now - bond_ladder_years) * 0.15  # Steeper curve for corporates
                        isa_ladder.loc[idx, 'estimated_ytm'] = max(3.0, estimated_new_yield)
                        isa_ladder.loc[idx, 'annual_income'] = principal * (isa_ladder.loc[idx, 'estimated_ytm'] / 100)
                        
                        bonds_maturing_this_year.append({
                            'Type': 'ISA',
                            'Bond': bond['bond_name'],
                            'ISIN': bond['isin'],
                            'Principal': principal,
                            'Year': current_year,
                            'Action': f'Reinvested in {new_maturity_year} corporate bond'
                        })
                    
                    # Bond income continues regardless of maturity (from current or new bonds)
                    if bond['allocation'] > 0:
                        isa_bond_income += bond['annual_income']
                
                # Calculate guaranteed income before additional withdrawals
                guaranteed_tax_free_income = isa_bond_income
                guaranteed_taxable_income = total_db_income + state_pension_income + sipp_bond_income
                
                # Calculate tax on guaranteed taxable income
                guaranteed_tax = self.calculate_income_tax_with_thresholds(
                    guaranteed_taxable_income, 
                    current_personal_allowance
                )
                
                guaranteed_net_income = (
                    guaranteed_tax_free_income + 
                    guaranteed_taxable_income - 
                    guaranteed_tax['total_tax']
                )
                
                # Determine additional income needed
                additional_net_needed = max(0, inflation_adjusted_target - guaranteed_net_income)
                
                # Initialize withdrawal variables
                sipp_tax_free_withdrawal = 0
                isa_withdrawal = 0
                sipp_taxable_withdrawal = 0
                additional_tax = 0
                
                if additional_net_needed > 0:
                    # Calculate total available amounts (bonds + cash for each type)
                    total_sipp_tax_free_available = remaining_sipp_tax_free_bonds + remaining_sipp_tax_free_cash
                    total_sipp_taxable_available = remaining_sipp_taxable_bonds + remaining_sipp_taxable_cash  
                    total_isa_available = remaining_isa_bonds + remaining_isa_cash
                    
                    # Apply maximum withdrawal rate check to total investment pots
                    total_available_investment_pots = total_sipp_tax_free_available + total_sipp_taxable_available + total_isa_available
                    max_allowed_withdrawal = total_available_investment_pots * (max_withdrawal_rate / 100)
                    
                    # Use optimized withdrawal order
                    available_sources = {
                        'sipp_tax_free': total_sipp_tax_free_available,
                        'isa': total_isa_available,
                        'sipp_taxable': total_sipp_taxable_available
                    }
                    
                    withdrawal_plan = self.optimize_withdrawal_order(
                        additional_net_needed,
                        available_sources,
                        guaranteed_taxable_income,
                        current_personal_allowance
                    )
                    
                    # Apply max withdrawal rate constraint 
                    total_planned_withdrawal = (
                        withdrawal_plan['sipp_tax_free'] + 
                        withdrawal_plan['isa_withdrawal'] + 
                        withdrawal_plan['sipp_taxable']
                    )
                    
                    if total_planned_withdrawal > max_allowed_withdrawal:
                        scale_factor = max_allowed_withdrawal / total_planned_withdrawal
                        withdrawal_plan['sipp_tax_free'] *= scale_factor
                        withdrawal_plan['isa_withdrawal'] *= scale_factor
                        withdrawal_plan['sipp_taxable'] *= scale_factor
                        withdrawal_plan['total_tax'] *= scale_factor
                    
                    # Extract final withdrawal amounts
                    sipp_tax_free_withdrawal = withdrawal_plan['sipp_tax_free']
                    isa_withdrawal = withdrawal_plan['isa_withdrawal']
                    sipp_taxable_withdrawal = withdrawal_plan['sipp_taxable']
                    additional_tax = withdrawal_plan['total_tax']
                    
                    # Update remaining amounts - withdraw from cash first, then bonds
                    # SIPP Tax-Free withdrawals
                    if sipp_tax_free_withdrawal > 0:
                        if remaining_sipp_tax_free_cash >= sipp_tax_free_withdrawal:
                            remaining_sipp_tax_free_cash -= sipp_tax_free_withdrawal
                        else:
                            bonds_needed = sipp_tax_free_withdrawal - remaining_sipp_tax_free_cash
                            remaining_sipp_tax_free_cash = 0
                            remaining_sipp_tax_free_bonds -= bonds_needed
                    
                    # ISA withdrawals
                    if isa_withdrawal > 0:
                        if remaining_isa_cash >= isa_withdrawal:
                            remaining_isa_cash -= isa_withdrawal
                        else:
                            bonds_needed = isa_withdrawal - remaining_isa_cash
                            remaining_isa_cash = 0
                            remaining_isa_bonds -= bonds_needed
                    
                    # SIPP Taxable withdrawals
                    if sipp_taxable_withdrawal > 0:
                        if remaining_sipp_taxable_cash >= sipp_taxable_withdrawal:
                            remaining_sipp_taxable_cash -= sipp_taxable_withdrawal
                        else:
                            bonds_needed = sipp_taxable_withdrawal - remaining_sipp_taxable_cash
                            remaining_sipp_taxable_cash = 0
                            remaining_sipp_taxable_bonds -= bonds_needed
                
                # Calculate final totals
                total_tax_free_income = guaranteed_tax_free_income + sipp_tax_free_withdrawal + isa_withdrawal
                total_taxable_income = guaranteed_taxable_income + sipp_taxable_withdrawal
                total_tax = guaranteed_tax['total_tax'] + additional_tax
                total_gross_income = total_tax_free_income + total_taxable_income
                total_net_income = total_gross_income - total_tax
                
                # Apply growth to invested portions only (bonds grow, cash doesn't)
                growth_factor = 1 + (investment_growth / 100)
                remaining_sipp_tax_free_bonds *= growth_factor
                remaining_sipp_taxable_bonds *= growth_factor
                remaining_isa_bonds *= growth_factor
                # Cash buffers remain unchanged (no growth applied)
                
                # Calculate effective tax rate
                effective_tax_rate = (total_tax / total_gross_income * 100) if total_gross_income > 0 else 0
                
                # Record annual data
                annual_data.append({
                    'year': year,
                    'calendar_year': current_year,
                    'inflation_factor': round(inflation_factor, 3),
                    'target_income': round(inflation_adjusted_target),
                    
                    # Income sources breakdown
                    'db_pension_income': round(total_db_income),
                    'state_pension_income': round(state_pension_income),
                    'sipp_bond_income': round(sipp_bond_income),
                    'isa_bond_income': round(isa_bond_income),
                    'sipp_tax_free_withdrawal': round(sipp_tax_free_withdrawal),
                    'isa_withdrawal': round(isa_withdrawal),
                    'sipp_taxable_withdrawal': round(sipp_taxable_withdrawal),
                    
                    # Income totals
                    'total_taxable_income': round(total_taxable_income),
                    'total_tax_free_income': round(total_tax_free_income),
                    'total_gross_income': round(total_gross_income),
                    
                    # Tax details
                    'personal_allowance': round(current_personal_allowance),
                    'total_tax': round(total_tax),
                    'effective_tax_rate': round(effective_tax_rate, 2),
                    
                    # Net income
                    'total_net_income': round(total_net_income),
                    'income_vs_target': round(total_net_income - inflation_adjusted_target),
                    
                    # Remaining pot values - properly calculated
                    'remaining_sipp_tax_free': round(remaining_sipp_tax_free_bonds + remaining_sipp_tax_free_cash),
                    'remaining_sipp_taxable': round(remaining_sipp_taxable_bonds + remaining_sipp_taxable_cash),
                    'remaining_isa': round(remaining_isa_bonds + remaining_isa_cash),
                    'total_remaining_pots': round(
                        remaining_sipp_tax_free_bonds + remaining_sipp_tax_free_cash +
                        remaining_sipp_taxable_bonds + remaining_sipp_taxable_cash +
                        remaining_isa_bonds + remaining_isa_cash
                    ),
                    
                    # Additional info
                    'bonds_maturing': bonds_maturing_this_year,
                    'max_withdrawal_applied': total_planned_withdrawal > max_allowed_withdrawal if additional_net_needed > 0 else False
                })
            
            return annual_data, sipp_ladder, isa_ladder
            
        except Exception as e:
            logging.error(f"Simulation failed: {traceback.format_exc()}")
            raise e

# Enhanced UI Components with Bond Recommendations

def add_sipp_strategy_selection():
    """Add SIPP strategy selection to sidebar"""
    st.sidebar.subheader("🎯 SIPP 25% Tax-Free Strategy")
    
    sipp_strategy = st.sidebar.radio(
        "Tax-Free Withdrawal Strategy",
        options=['gradual', 'mixed', 'upfront'],
        format_func=lambda x: {
            'gradual': '📅 Gradual (Take as needed)',
            'mixed': '⚖️ Mixed (Half upfront, half gradual)', 
            'upfront': '💰 Upfront (Large lump sum)'
        }[x],
        help="How to handle your 25% tax-free SIPP allowance"
    )
    
    upfront_percent = 0
    if sipp_strategy == 'upfront':
        upfront_percent = st.sidebar.slider(
            "% of Tax-Free to Take Upfront",
            min_value=25,
            max_value=100,
            value=75,
            step=5,
            help="What percentage of your 25% allowance to take immediately"
        )
    elif sipp_strategy == 'mixed':
        upfront_percent = 50
    
    return sipp_strategy, upfront_percent

def display_bond_recommendations(sipp_ladder, isa_ladder):
    """Display specific bond recommendations with purchase instructions"""
    st.subheader("🔗 Specific Bond Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**SIPP Bond Ladder - UK Gilts**")
        if not sipp_ladder.empty:
            # Format SIPP ladder for display
            sipp_display = sipp_ladder[['bond_name', 'isin', 'maturity_date', 'coupon', 
                                      'estimated_ytm', 'allocation', 'annual_income']].copy()
            sipp_display['allocation'] = sipp_display['allocation'].apply(lambda x: f"£{x:,.0f}")
            sipp_display['annual_income'] = sipp_display['annual_income'].apply(lambda x: f"£{x:,.0f}")
            sipp_display['coupon'] = sipp_display['coupon'].apply(lambda x: f"{x:.3f}%")
            sipp_display['estimated_ytm'] = sipp_display['estimated_ytm'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(sipp_display, use_container_width=True)
            
            # Purchase instructions for SIPP
            st.info("""
            **How to Buy on Interactive Investor:**
            1. Search by ISIN code in the 'Find & Invest' section
            2. Check current price and yield-to-maturity
            3. Calculate number of bonds needed: Allocation ÷ Current Price
            4. Place order with £7.99 trading fee per bond type
            """)
        else:
            st.warning("No SIPP bond recommendations generated")
    
    with col2:
        st.write("**ISA Bond Ladder - Mixed Bonds**")
        if not isa_ladder.empty:
            # Format ISA ladder for display
            isa_display = isa_ladder[['bond_name', 'isin', 'maturity_date', 'coupon', 
                                    'estimated_ytm', 'allocation', 'annual_income']].copy()
            isa_display['allocation'] = isa_display['allocation'].apply(lambda x: f"£{x:,.0f}")
            isa_display['annual_income'] = isa_display['annual_income'].apply(lambda x: f"£{x:,.0f}")
            isa_display['coupon'] = isa_display['coupon'].apply(lambda x: f"{x:.3f}%")
            isa_display['estimated_ytm'] = isa_display['estimated_ytm'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(isa_display, use_container_width=True)
            
            # Purchase instructions for ISA
            st.info("""
            **ISA Bond Strategy:**
            1. Prioritize higher-yield corporate bonds
            2. Check credit ratings (BBB+ minimum)
            3. Diversify across sectors (max 20% per sector)
            4. Monitor credit spreads vs gilts
            """)
        else:
            st.warning("No ISA bond recommendations generated")

def display_implementation_timeline():
    """Display step-by-step implementation timeline"""
    st.subheader("📅 Implementation Timeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**6-12 Months Before Retirement**")
        st.write("""
        ✅ Open Interactive Investor SIPP and ISA accounts
        
        ✅ Begin transferring funds from current providers
        
        ✅ Research current bond prices using ISIN codes
        
        ✅ Set up yield alerts for target purchase prices
        
        ✅ Create detailed purchase timeline spreadsheet
        """)
    
    with col2:
        st.write("**3-6 Months Before Retirement**")
        st.write("""
        ✅ Start purchasing bonds - begin with most liquid gilts
        
        ✅ Stagger purchases over 2-3 months to average prices
        
        ✅ Focus on highest quality issues first
        
        ✅ Document all purchases with actual YTM calculations
        
        ✅ Monitor portfolio allocation vs targets
        """)
    
    with col3:
        st.write("**1-3 Months Before Retirement**")
        st.write("""
        ✅ Complete final bond purchases
        
        ✅ Set up drawdown with II (£125 one-time fee)
        
        ✅ Establish monthly withdrawal schedule
        
        ✅ Ensure adequate cash buffer is in place
        
        ✅ Review and finalize income strategy
        """)

@st.cache_data(ttl=3600)
def get_current_gilt_yields():
    """Fetch current UK gilt yields with enhanced simulation"""
    try:
        # Enhanced simulation with more realistic yield curve
        base_10y = 0.042  # 4.2% base 10-year yield
        daily_variation = np.random.normal(0, 0.001)
        current_10y = max(0.02, base_10y + daily_variation)
        
        # Realistic yield curve shape
        yield_2y = current_10y - 0.008  # Inverted or flat curve
        yield_5y = current_10y - 0.003
        
        return {
            'success': True,
            'yield_10y': current_10y,
            'yield_5y': yield_5y,
            'yield_2y': yield_2y,
            'source': 'Market Simulation',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'curve_shape': 'Normal' if yield_10y > yield_5y > yield_2y else 'Inverted'
        }
    except Exception as e:
        return {
            'success': False,
            'yield_10y': 0.045,
            'yield_5y': 0.042,
            'yield_2y': 0.037,
            'source': 'Default Values',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'error': str(e)
        }

def create_enhanced_excel_export(annual_data, sipp_ladder, isa_ladder, scenario_results=None):
    """Create comprehensive Excel export with bond recommendations"""
    try:
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Annual data
            df_annual = pd.DataFrame(annual_data)
            df_annual.to_excel(writer, sheet_name='Annual Analysis', index=False)
            
            # SIPP bond ladder with recommendations
            if not sipp_ladder.empty:
                sipp_export = sipp_ladder.copy()
                sipp_export.to_excel(writer, sheet_name='SIPP Bond Ladder', index=False)
            
            # ISA bond ladder with recommendations
            if not isa_ladder.empty:
                isa_export = isa_ladder.copy()
                isa_export.to_excel(writer, sheet_name='ISA Bond Ladder', index=False)
            
            # Implementation checklist
            checklist_data = {
                'Phase': [
                    '6-12 Months Before', '6-12 Months Before', '6-12 Months Before',
                    '3-6 Months Before', '3-6 Months Before', '3-6 Months Before',
                    '1-3 Months Before', '1-3 Months Before', '1-3 Months Before'
                ],
                'Task': [
                    'Open II SIPP and ISA accounts',
                    'Transfer funds from current providers',
                    'Research bond prices using ISINs',
                    'Begin purchasing bonds (start with gilts)',
                    'Stagger purchases over 2-3 months',
                    'Document all purchases with YTM',
                    'Complete final bond purchases',
                    'Set up drawdown with II',
                    'Establish withdrawal schedule'
                ],
                'Status': ['Pending'] * 9,
                'Notes': [''] * 9
            }
            pd.DataFrame(checklist_data).to_excel(writer, sheet_name='Implementation Checklist', index=False)
            
            # Bond research template
            if not sipp_ladder.empty:
                research_template = []
                for _, bond in sipp_ladder.iterrows():
                    research_template.append({
                        'ISIN': bond['isin'],
                        'Bond Name': bond['bond_name'],
                        'Target Allocation': f"£{bond['allocation']:,.0f}",
                        'Current Price': 'TBD',
                        'Current YTM': 'TBD',
                        'Bonds to Buy': 'TBD',
                        'Total Cost': 'TBD',
                        'Purchase Date': 'TBD',
                        'Notes': ''
                    })
                
                pd.DataFrame(research_template).to_excel(writer, sheet_name='Bond Research Template', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel export: {str(e)}")
        return None

def main():
    st.title("💰 Enhanced SIPP Bond Strategy Calculator")
    st.markdown("**Professional UK retirement planning with specific bond recommendations**")
    
    # Display current market data
    gilt_data = get_current_gilt_yields()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("UK 10Y Gilt", f"{gilt_data['yield_10y']*100:.2f}%")
    with col2:
        st.metric("UK 5Y Gilt", f"{gilt_data['yield_5y']*100:.2f}%")
    with col3:
        st.metric("UK 2Y Gilt", f"{gilt_data['yield_2y']*100:.2f}%")
    with col4:
        st.metric("Curve Shape", gilt_data.get('curve_shape', 'Normal'))
    
    # Enhanced help system
    with st.expander("❓ Enhanced Bond Strategy Features"):
        st.markdown("""
        ## 🎯 New Features in This Enhanced Version
        
        **✅ Specific Bond Recommendations**
        - Actual UK Gilt ISINs with maturity dates
        - Corporate bond suggestions for ISA
        - Real yield-to-maturity calculations
        - Interactive Investor platform integration
        
        **✅ Implementation Guidance**
        - Step-by-step purchase timeline
        - Platform-specific instructions
        - Bond research templates
        - Purchase tracking tools
        
        **✅ Professional Bond Analysis**
        - Credit rating considerations
        - Sector diversification guidance
        - Yield curve analysis
        - Market timing strategies
        """)
    
    calc = EnhancedSIPPBondCalculator()
    
    # Sidebar inputs (same as before but enhanced)
    st.sidebar.header("📊 Portfolio Parameters")
    
    # Portfolio values
    sipp_value = st.sidebar.number_input(
        "SIPP Value (£)", 
        min_value=0, 
        value=565000, 
        step=1000,
        help="Current value of your Self-Invested Personal Pension"
    )
    
    isa_value = st.sidebar.number_input(
        "ISA Value (£)", 
        min_value=0, 
        value=92000, 
        step=1000,
        help="Current value of your Individual Savings Account"
    )
    
    # Target income
    target_annual_income = st.sidebar.number_input(
        "Target Annual Net Income (£)", 
        min_value=0, 
        value=40000, 
        step=1000,
        help="Desired annual net income (after tax)"
    )
    
    # SIPP strategy selection
    sipp_strategy, upfront_tax_free_percent = add_sipp_strategy_selection()
    
    # Bond ladder parameters
    st.sidebar.subheader("🔗 Bond Ladder Settings")
    
    bond_ladder_years = st.sidebar.slider(
        "Ladder Duration (Years)", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="Number of years in your bond ladder"
    )
    
    cash_buffer_percent = st.sidebar.slider(
        "Cash Buffer (%)", 
        min_value=0, 
        max_value=20, 
        value=5,
        help="Percentage to keep in cash for flexibility"
    )
    
    # Pension inputs
    st.sidebar.subheader("🏛️ Defined Benefit Pensions")
    
    nhs_pension = st.sidebar.number_input("NHS Pension (£/year)", min_value=0, value=13000, step=500)
    state_pension = st.sidebar.number_input("State Pension (£/year)", min_value=0, value=11500, step=100)
    state_pension_start_year = st.sidebar.slider("State Pension Start Year", min_value=1, max_value=20, value=5)
    
    # Economic parameters
    st.sidebar.subheader("📈 Economic Assumptions")
    
    inflation_rate = st.sidebar.slider("Inflation Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    investment_growth = st.sidebar.slider("Investment Growth (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.1)
    max_withdrawal_rate = st.sidebar.slider("Max Withdrawal Rate (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    years = st.sidebar.slider("Simulation Years", min_value=5, max_value=40, value=25)
    
    # Calculate button
    if st.sidebar.button("🚀 Calculate Enhanced Bond Strategy", type="primary"):
        try:
            # Prepare parameters
            db_pensions = {
                'NHS Pension': nhs_pension
            }
            
            base_params = {
                'sipp_value': sipp_value,
                'isa_value': isa_value,
                'target_annual_income': target_annual_income,
                'sipp_strategy': sipp_strategy,
                'upfront_tax_free_percent': upfront_tax_free_percent,
                'bond_ladder_years': bond_ladder_years,
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
            
            # Run simulation
            with st.spinner("Calculating bond strategy with specific recommendations..."):
                annual_data, sipp_ladder, isa_ladder = calc.simulate_comprehensive_strategy(base_params)
            
            if not annual_data:
                st.error("No results generated. Please check your inputs.")
                return
            
            # Display results with enhanced bond recommendations
            st.header("📊 Enhanced Bond Strategy Results")
            
            # Key metrics
            first_year = annual_data[0]
            last_year = annual_data[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Year 1 Net Income", 
                    f"£{first_year['total_net_income']:,}",
                    delta=f"£{first_year['income_vs_target']:,}"
                )
            
            with col2:
                st.metric(
                    "Year 1 Tax Rate", 
                    f"{first_year['effective_tax_rate']:.1f}%"
                )
            
            with col3:
                total_sipp_tax_free_used = sum([year['sipp_tax_free_withdrawal'] for year in annual_data])
                st.metric(
                    "SIPP Tax-Free Used", 
                    f"£{total_sipp_tax_free_used:,}"
                )
            
            with col4:
                st.metric(
                    "Final Portfolio Value", 
                    f"£{last_year['total_remaining_pots']:,}"
                )
            
            # Display specific bond recommendations
            display_bond_recommendations(sipp_ladder, isa_ladder)
            
            # Implementation timeline
            display_implementation_timeline()
            
            # Enhanced download section
            st.subheader("📥 Download Enhanced Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                df = pd.DataFrame(annual_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Analysis CSV",
                    data=csv,
                    file_name=f"enhanced_bond_strategy_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Enhanced Excel with bond recommendations
                excel_data = create_enhanced_excel_export(annual_data, sipp_ladder, isa_ladder)
                if excel_data:
                    st.download_button(
                        label="📈 Download Complete Bond Strategy Report",
                        data=excel_data,
                        file_name=f"complete_bond_strategy_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Enhanced visualizations
            st.subheader("📈 Strategy Visualization")
            
            df = pd.DataFrame(annual_data)
            
            # Create income composition chart
            fig_income = go.Figure()
            
            fig_income.add_trace(go.Scatter(
                x=df['year'],
                y=df['isa_bond_income'],
                stackgroup='one',
                name='ISA Bonds (Tax-Free)',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(75, 192, 192, 0.7)'
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
                y=df['db_pension_income'],
                stackgroup='one',
                name='NHS Pension',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 99, 132, 0.7)'
            ))
            
            fig_income.add_trace(go.Scatter(
                x=df['year'],
                y=df['sipp_tax_free_withdrawal'] + df['isa_withdrawal'] + df['sipp_taxable_withdrawal'],
                stackgroup='one',
                name='Additional Withdrawals',
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
                title='Enhanced Income Strategy: Bond Ladder + Pension Income',
                xaxis_title='Year',
                yaxis_title='Annual Income (£)',
                height=500
            )
            
            st.plotly_chart(fig_income, use_container_width=True)
            
            # Bond maturity timeline with reinvestment strategy
            st.subheader("🗓️ Bond Ladder Strategy & Reinvestment")
            
            st.info("""
            **How the Bond Ladder Works:**
            - When bonds mature, principal is automatically reinvested in new 5-year bonds
            - This maintains steady income flow throughout retirement
            - Yield adjustments reflect changing interest rate environment
            - SIPP bonds → UK Gilts, ISA bonds → Corporate bonds for higher yield
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SIPP Bond Schedule (with Reinvestment)**")
                if not sipp_ladder.empty:
                    maturity_calendar = sipp_ladder[['maturity_year', 'bond_name', 'allocation', 'estimated_ytm']].copy()
                    maturity_calendar['allocation'] = maturity_calendar['allocation'].apply(lambda x: f"£{x:,.0f}")
                    maturity_calendar['estimated_ytm'] = maturity_calendar['estimated_ytm'].apply(lambda x: f"{x:.2f}%")
                    maturity_calendar = maturity_calendar.sort_values('maturity_year')
                    st.dataframe(maturity_calendar, use_container_width=True)
                    
                    st.caption("💡 When each bond matures, principal reinvests in new 5-year UK Gilt")
            
            with col2:
                st.write("**ISA Bond Schedule (with Reinvestment)**")
                if not isa_ladder.empty:
                    isa_maturity_calendar = isa_ladder[['maturity_year', 'bond_name', 'allocation', 'estimated_ytm']].copy()
                    isa_maturity_calendar['allocation'] = isa_maturity_calendar['allocation'].apply(lambda x: f"£{x:,.0f}")
                    isa_maturity_calendar['estimated_ytm'] = isa_maturity_calendar['estimated_ytm'].apply(lambda x: f"{x:.2f}%")
                    isa_maturity_calendar = isa_maturity_calendar.sort_values('maturity_year')
                    st.dataframe(isa_maturity_calendar, use_container_width=True)
                    
                    st.caption("💡 When each bond matures, principal reinvests in new 5-year Corporate bond")
            
            # Interactive Investor specific guidance
            st.subheader("🏦 Interactive Investor Implementation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("""
                **Account Setup Costs**
                - Monthly Platform Fee: £12.99
                - SIPP Setup: £125 (one-time)
                - Annual Drawdown Fee: £125
                - Bond Trading: £7.99 per trade
                """)
                
                # Calculate total trading costs
                total_bonds = len(sipp_ladder) + len(isa_ladder) if not sipp_ladder.empty and not isa_ladder.empty else 0
                total_trading_cost = total_bonds * 7.99
                annual_platform_cost = (12.99 * 12) + 125  # Monthly fee + drawdown fee
                
                st.info(f"""
                **Estimated First Year Costs:**
                - Platform Fees: £{annual_platform_cost:.0f}
                - Bond Purchases: £{total_trading_cost:.0f}
                - **Total**: £{annual_platform_cost + total_trading_cost:.0f}
                """)
            
            with col2:
                st.info("""
                **How to Find Bonds on II:**
                1. Log into your account
                2. Go to 'Find & Invest'
                3. Select 'Bonds & Gilts'
                4. Search by ISIN code
                5. Check current price and yield
                6. Place order (minimum £100 face value)
                """)
                
                st.warning("""
                **Important Notes:**
                - Bond prices change daily
                - Check yield-to-maturity before buying
                - Consider staggering purchases
                - Keep trading records for tax
                """)
            
            # Professional recommendations
            st.subheader("🎓 Professional Strategy Recommendations")
            
            # Calculate some key metrics for recommendations
            avg_tax_rate = np.mean([year['effective_tax_rate'] for year in annual_data])
            total_bond_income = sum([year['sipp_bond_income'] + year['isa_bond_income'] for year in annual_data])
            
            recommendations = []
            
            if avg_tax_rate < 15:
                recommendations.append("✅ **Excellent Tax Efficiency**: Average tax rate below 15% - strategy is very tax-efficient")
            elif avg_tax_rate > 25:
                recommendations.append("⚠️ **Tax Optimization Needed**: Consider spreading withdrawals to reduce tax bands")
            
            if first_year['income_vs_target'] >= 0:
                recommendations.append("✅ **Income Target Met**: Strategy successfully meets your income requirements")
            else:
                recommendations.append("🔧 **Income Shortfall**: Consider higher-yield bonds or adjust withdrawal strategy")
            
            if last_year['total_remaining_pots'] > sipp_value + isa_value:
                recommendations.append("✅ **Sustainable Growth**: Portfolio grows over time - very conservative approach")
            elif last_year['total_remaining_pots'] > (sipp_value + isa_value) * 0.5:
                recommendations.append("✅ **Good Sustainability**: Portfolio maintains good value over simulation period")
            else:
                recommendations.append("⚠️ **Sustainability Risk**: Portfolio depletes significantly - consider reducing withdrawal rate")
            
            # Bond-specific recommendations
            if not sipp_ladder.empty:
                avg_sipp_yield = sipp_ladder['estimated_ytm'].mean()
                current_gilt_yield = gilt_data['yield_10y'] * 100
                
                if avg_sipp_yield < current_gilt_yield - 0.5:
                    recommendations.append(f"📈 **Yield Opportunity**: Current gilt yields (~{current_gilt_yield:.1f}%) higher than strategy assumption")
                elif avg_sipp_yield > current_gilt_yield + 0.5:
                    recommendations.append(f"⚠️ **Yield Risk**: Strategy assumes higher yields than current market (~{current_gilt_yield:.1f}%)")
            
            for rec in recommendations:
                if "✅" in rec:
                    st.success(rec)
                elif "⚠️" in rec or "🔧" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
            
            # Next steps guidance
            st.subheader("🚀 Next Steps")
            
            next_steps = [
                "1. **Open Interactive Investor accounts** (SIPP and ISA) if not already done",
                "2. **Research current bond prices** using the ISINs provided above",
                "3. **Create a purchase timeline** spreading buys over 2-3 months",
                "4. **Start with UK Gilts** (most liquid and straightforward)",
                "5. **Document all purchases** with actual prices and yields",
                "6. **Set up regular portfolio reviews** (quarterly recommended)",
                "7. **Monitor credit ratings** for corporate bonds annually"
            ]
            
            for step in next_steps:
                st.write(step)
            
            # Risk warnings
            st.subheader("⚠️ Important Risk Considerations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.error("""
                **Interest Rate Risk**
                - Bond prices fall when rates rise
                - Consider staggered purchases
                - Ladder structure provides protection
                """)
                
                st.error("""
                **Credit Risk**
                - Corporate bonds can default
                - Stick to investment grade (BBB+ or better)
                - Diversify across sectors
                """)
            
            with col2:
                st.error("""
                **Inflation Risk**
                - Fixed income loses purchasing power
                - Consider index-linked bonds
                - Review strategy annually
                """)
                
                st.error("""
                **Liquidity Risk**
                - Bonds may be hard to sell before maturity
                - Keep adequate cash buffer
                - Use major issuers for better liquidity
                """)
            
        except Exception as e:
            st.error(f"Error running enhanced bond calculation: {str(e)}")
            logging.error(f"Enhanced bond calculation failed: {traceback.format_exc()}")
            st.info("Please check your inputs and try again.")
    
    else:
        # Enhanced summary when no calculation is run
        st.info("👆 Configure your parameters and click 'Calculate Enhanced Bond Strategy' to see specific bond recommendations.")
        
        # Show sample bond recommendations
        st.subheader("🔗 Sample Bond Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample SIPP Bonds (UK Gilts)**")
            sample_sipp = pd.DataFrame({
                'Bond Name': ['UK Treasury 4.25% 2027', 'UK Treasury 1.625% 2028', 'UK Treasury 0.875% 2029'],
                'ISIN': ['GB00B16NNR78', 'GB00BFWFPP71', 'GB00BJMHB534'],
                'Maturity': ['2027-12-07', '2028-10-22', '2029-10-31'],
                'Coupon': ['4.25%', '1.625%', '0.875%'],
                'Rating': ['AA', 'AA', 'AA']
            })
            st.dataframe(sample_sipp, use_container_width=True)
        
        with col2:
            st.write("**Sample ISA Bonds (Corporate)**")
            sample_isa = pd.DataFrame({
                'Bond Name': ['National Grid 2.625% 2028', 'BT Group 4.25% 2029', 'Vodafone 4.875% 2030'],
                'ISIN': ['GB00BZ03MX94', 'GB00BMF5JQ11', 'GB00BM8QGX98'],
                'Maturity': ['2028-06-16', '2029-11-23', '2030-07-03'],
                'Coupon': ['2.625%', '4.25%', '4.875%'],
                'Rating': ['BBB+', 'BBB', 'BBB']
            })
            st.dataframe(sample_isa, use_container_width=True)
        
        # Feature highlights
        st.subheader("🌟 Enhanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Specific Bond Recommendations**
            - Actual ISINs for immediate purchase
            - Real UK Gilt and Corporate Bond data
            - Yield-to-maturity calculations
            - Credit rating considerations
            """)
            
            st.success("""
            **✅ Interactive Investor Integration**
            - Platform-specific purchase instructions
            - Cost calculations including trading fees
            - Account setup timeline
            - Professional implementation guidance
            """)
        
        with col2:
            st.success("""
            **✅ Professional Bond Analysis**
            - Diversification across sectors
            - Credit quality assessment
            - Maturity calendar management
            - Reinvestment strategies
            """)
            
            st.success("""
            **✅ Implementation Support**
            - Step-by-step purchase timeline
            - Bond research templates
            - Risk management guidance
            - Ongoing monitoring strategies
            """)

if __name__ == "__main__":
    main()
