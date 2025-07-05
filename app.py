import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import logging
import traceback
from typing import Dict, List, Tuple, Optional
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
st.set_page_config(
    page_title="Comprehensive Pension & Bond Ladder Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/comprehensive-pension-calculator',
        'Report a bug': 'https://github.com/yourusername/comprehensive-pension-calculator/issues',
        'About': """
        # Comprehensive Pension & Bond Ladder Calculator
        
        This application combines bond ladder strategies with pension drawdown analysis 
        for comprehensive UK retirement planning.
        
        **Features:**
        - Year-by-year income and tax analysis
        - Bond ladder management with reinvestment
        - Multiple pension income sources
        - UK tax optimization
        - Inflation adjustments
        
        **Disclaimer:** This tool provides estimates for educational purposes only. 
        Please seek professional financial advice before making investment decisions.
        """
    }
)

class ComprehensivePensionCalculator:
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

def create_alerts(annual_data):
    """Create alerts for potential issues"""
    alerts = []
    
    for year_data in annual_data[:5]:  # Check first 5 years
        # Income shortfall alert
        if year_data['income_vs_target'] < -2000:
            alerts.append(f"⚠️ Year {year_data['year']}: Income shortfall of £{abs(year_data['income_vs_target']):,}")
        
        # High tax rate alert
        if year_data['effective_tax_rate'] > 30:
            alerts.append(f"🚨 Year {year_data['year']}: High effective tax rate of {year_data['effective_tax_rate']:.1f}%")
        
        # Low pot value alert
        if year_data['total_remaining_pots'] < 100000:
            alerts.append(f"⚠️ Year {year_data['year']}: Pot value below £100k")
    
    if alerts:
        st.subheader("🚨 Strategy Alerts")
        for alert in alerts:
            st.warning(alert)

def create_excel_export(annual_data, sipp_ladder, isa_ladder):
    """Create comprehensive Excel export"""
    try:
        output = BytesIO()
        
        # Create Excel writer
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
                        'Total Years Analyzed'
                    ],
                    'Value': [
                        f"£{annual_data[0]['total_net_income']:,}",
                        f"{np.mean([y['effective_tax_rate'] for y in annual_data]):.1f}%",
                        f"£{annual_data[-1]['total_remaining_pots']:,}",
                        len(annual_data)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating Excel export: {str(e)}")
        return None

def main():
    st.title("💰 Comprehensive Pension & Bond Ladder Calculator")
    st.markdown("**Analyze your retirement strategy with bonds, drawdown, and defined benefit pensions**")
    
    calc = ComprehensivePensionCalculator()
    
    # Sidebar inputs
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
    
    # Bond ladder parameters
    st.sidebar.subheader("🔗 Bond Ladder Settings")
    
    bond_ladder_years = st.sidebar.slider(
        "Ladder Duration (Years)", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="Number of years in your bond ladder"
    )
    
    sipp_yield = st.sidebar.slider(
        "SIPP Bond Yield (%)", 
        min_value=2.0, 
        max_value=8.0, 
        value=4.5, 
        step=0.1,
        help="Expected average yield for SIPP bonds (UK Gilts)"
    )
    
    isa_yield = st.sidebar.slider(
        "ISA Bond Yield (%)", 
        min_value=2.0, 
        max_value=8.0, 
        value=5.0, 
        step=0.1,
        help="Expected average yield for ISA bonds (Corporate bonds)"
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
    
    gp_pension = st.sidebar.number_input(
        "GP Pension (£/year)", 
        min_value=0, 
        value=0, 
        step=500
    )
    
    kpmg_pension = st.sidebar.number_input(
        "KPMG Pension (£/year)", 
        min_value=0, 
        value=0, 
        step=500
    )
    
    nhs_pension = st.sidebar.number_input(
        "NHS Pension (£/year)", 
        min_value=0, 
        value=13000, 
        step=500
    )
    
    state_pension = st.sidebar.number_input(
        "State Pension (£/year)", 
        min_value=0, 
        value=11500, 
        step=100
    )
    
    state_pension_start_year = st.sidebar.slider(
        "State Pension Start Year", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Year when state pension starts (from retirement)"
    )
    
    # Economic parameters
    st.sidebar.subheader("📈 Economic Assumptions")
    
    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.5, 
        step=0.1
    )
    
    investment_growth = st.sidebar.slider(
        "Investment Growth (%)", 
        min_value=0.0, 
        max_value=15.0, 
        value=4.0, 
        step=0.1,
        help="Annual growth rate for remaining pension pots"
    )
    
    max_withdrawal_rate = st.sidebar.slider(
        "Max Withdrawal Rate (%)", 
        min_value=1.0, 
        max_value=10.0, 
        value=4.0, 
        step=0.1,
        help="Maximum annual withdrawal rate from pension pots"
    )
    
    years = st.sidebar.slider(
        "Simulation Years", 
        min_value=5, 
        max_value=40, 
        value=25,
        help="Number of years to simulate"
    )
    
    # Calculate button
    if st.sidebar.button("🚀 Calculate Strategy", type="primary"):
        try:
            # Prepare parameters
            db_pensions = {
                'GP Pension': gp_pension,
                'KPMG Pension': kpmg_pension,
                'NHS Pension': nhs_pension
            }
            
            params = {
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
            
            # Run simulation
            with st.spinner("Calculating comprehensive retirement strategy..."):
                annual_data, sipp_ladder, isa_ladder = calc.simulate_comprehensive_strategy(params)
            
            if not annual_data:
                st.error("No results generated. Please check your inputs.")
                return
            
            # Display results
            st.header("📊 Strategy Results")
            
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
            
            # Alerts
            create_alerts(annual_data)
            
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
            
            # Charts
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
                    y=df['total_tax'],
                    mode='lines',
                    name='Tax Paid',
                    line=dict(color='red', width=2)
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
            
            # Implementation guidance
            st.subheader("📋 Implementation Guidance")
            
            st.write("**Recommended Actions:**")
            
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
            
            # Download data
            st.subheader("📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv,
                    file_name=f"pension_strategy_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download
                excel_data = create_excel_export(annual_data, sipp_ladder, isa_ladder)
                if excel_data:
                    st.download_button(
                        label="📈 Download Excel Report",
                        data=excel_data,
                        file_name=f"comprehensive_pension_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
        except Exception as e:
            st.error(f"Error running calculation: {str(e)}")
            logging.error(f"Calculation failed: {traceback.format_exc()}")
            st.info("Please check your inputs and try again. If the problem persists, try reducing the number of simulation years or adjusting your target income.")
    
    else:
        # Show summary when no calculation run
        st.info("👆 Configure your parameters in the sidebar and click 'Calculate Strategy' to see your personalized retirement analysis.")
        
        # Show example summary
        st.subheader("📋 What This Calculator Shows")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("""
            **Income Sources:**
            - Bond ladder income (SIPP & ISA)
            - Defined benefit pensions
            - State pension
            - Pension pot drawdown
            - Year-by-year breakdown
            """)
        
        with col2:
            st.write("""
            **Tax Analysis:**
            - Complete UK tax calculations
            - Personal allowance tapering
            - Income tax breakdown by rate
            - Effective tax rates
            - Tax optimization suggestions
            """)
        
        with col3:
            st.write("""
            **Strategy Features:**
            - Bond maturity planning
            - Automatic reinvestment
            - Withdrawal rate management
            - Inflation adjustments
            - Portfolio sustainability
            """)
        
        # Show sample data visualization
        st.subheader("📊 Sample Analysis")
        
        # Create sample data for demonstration
        sample_years = list(range(1, 11))
        sample_income = [40000 + (i * 1000) for i in sample_years]
        sample_tax = [income * 0.15 for income in sample_income]
        
        fig_sample = go.Figure()
        fig_sample.add_trace(go.Scatter(
            x=sample_years, 
            y=sample_income, 
            mode='lines+markers', 
            name='Target Income',
            line=dict(color='blue', width=3)
        ))
        fig_sample.add_trace(go.Scatter(
            x=sample_years, 
            y=sample_tax, 
            mode='lines+markers', 
            name='Tax Paid',
            line=dict(color='red', width=2)
        ))
        
        fig_sample.update_layout(
            title='Sample: Income and Tax Projection',
            xaxis_title='Year',
            yaxis_title='Amount (£)',
            height=300
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)
        
        # Key features highlight
        st.subheader("🌟 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ **Comprehensive Analysis**")
            st.write("• Combines bonds, pensions, and drawdown")
            st.write("• Year-by-year detailed projections")
            st.write("• UK tax rules with personal allowance tapering")
            
            st.success("✅ **Bond Ladder Management**")
            st.write("• Automatic maturity tracking")
            st.write("• Reinvestment planning")
            st.write("• Separate SIPP and ISA strategies")
        
        with col2:
            st.success("✅ **Tax Optimization**")
            st.write("• Minimizes higher-rate tax exposure")
            st.write("• Maximizes tax-free withdrawals")
            st.write("• Inflation-adjusted calculations")
            
            st.success("✅ **Professional Outputs**")
            st.write("• Downloadable Excel reports")
            st.write("• Interactive charts and analysis")
            st.write("• Implementation guidance")

if __name__ == "__main__":
    main()
