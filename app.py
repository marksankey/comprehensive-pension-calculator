# Enhanced Features to Add to Your Calculator

# 1. Real-time Gilt Data Integration
import yfinance as yf
import requests

def get_current_gilt_yields():
    """Fetch current UK gilt yields from financial APIs"""
    try:
        # Example: Fetch 10-year gilt yield
        uk_10y = yf.Ticker("^TNX-GB")  # UK 10-year treasury
        current_yield = uk_10y.info.get('regularMarketPrice', 4.5)
        return current_yield
    except:
        return 4.5  # Fallback default

# 2. Scenario Analysis
def add_scenario_analysis():
    """Add scenario analysis with different economic conditions"""
    st.subheader("🎭 Scenario Analysis")
    
    scenarios = {
        'Base Case': {'inflation': 2.5, 'growth': 4.0, 'yields': 4.5},
        'High Inflation': {'inflation': 5.0, 'growth': 2.0, 'yields': 6.0},
        'Low Growth': {'inflation': 1.5, 'growth': 1.5, 'yields': 3.0},
        'Market Stress': {'inflation': 3.0, 'growth': -2.0, 'yields': 5.5}
    }
    
    scenario_results = {}
    for scenario_name, params in scenarios.items():
        # Run calculation with scenario parameters
        scenario_params = base_params.copy()
        scenario_params.update(params)
        results = calc.simulate_comprehensive_strategy(scenario_params)
        scenario_results[scenario_name] = results
    
    # Display comparison chart
    fig_scenarios = go.Figure()
    for scenario_name, results in scenario_results.items():
        annual_data = results[0]
        years = [data['year'] for data in annual_data]
        net_incomes = [data['total_net_income'] for data in annual_data]
        
        fig_scenarios.add_trace(go.Scatter(
            x=years,
            y=net_incomes,
            mode='lines',
            name=scenario_name,
            line=dict(width=2)
        ))
    
    fig_scenarios.update_layout(
        title='Scenario Analysis: Net Income Over Time',
        xaxis_title='Year',
        yaxis_title='Net Income (£)',
        height=400
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)

# 3. Monte Carlo Simulation
import random

def monte_carlo_simulation(params, num_simulations=1000):
    """Run Monte Carlo simulation with variable returns"""
    results = []
    
    for simulation in range(num_simulations):
        # Add random variation to key parameters
        sim_params = params.copy()
        sim_params['investment_growth'] += random.normalvariate(0, 2)  # ±2% std dev
        sim_params['inflation_rate'] += random.normalvariate(0, 1)    # ±1% std dev
        sim_params['sipp_yield'] += random.normalvariate(0, 0.5)      # ±0.5% std dev
        
        # Run simulation
        try:
            annual_data, _, _ = calc.simulate_comprehensive_strategy(sim_params)
            if annual_data:
                final_pot = annual_data[-1]['total_remaining_pots']
                avg_income = np.mean([year['total_net_income'] for year in annual_data])
                results.append({'final_pot': final_pot, 'avg_income': avg_income})
        except:
            continue
    
    return results

# 4. Tax-Year Optimizer
def optimize_tax_efficiency():
    """Suggest tax-efficient withdrawal strategies"""
    st.subheader("🎯 Tax Optimization Suggestions")
    
    suggestions = []
    
    # Check if income hits higher rate threshold
    if annual_data[0]['total_taxable_income'] > 50270:
        suggestions.append("⚠️ Consider increasing ISA withdrawals to reduce taxable income")
    
    # Check personal allowance tapering
    if annual_data[0]['total_taxable_income'] > 100000:
        suggestions.append("🚨 Personal allowance tapering applies - very tax inefficient zone")
    
    # Check if state pension timing could be optimized
    if state_pension_start_year <= 5:
        suggestions.append("💡 Consider deferring state pension for higher payments")
    
    for suggestion in suggestions:
        st.info(suggestion)

# 5. Implementation Timeline
def create_implementation_timeline():
    """Create a detailed implementation timeline"""
    st.subheader("📅 Implementation Timeline")
    
    timeline_data = [
        {"Phase": "6-12 months before retirement", "Action": "Open II SIPP and ISA accounts", "Status": "Planning"},
        {"Phase": "6-12 months before retirement", "Action": "Transfer existing pensions", "Status": "Planning"},
        {"Phase": "3-6 months before retirement", "Action": "Research and select specific gilts", "Status": "Pending"},
        {"Phase": "3-6 months before retirement", "Action": "Begin purchasing bonds", "Status": "Pending"},
        {"Phase": "1-3 months before retirement", "Action": "Complete bond purchases", "Status": "Pending"},
        {"Phase": "1-3 months before retirement", "Action": "Set up drawdown arrangements", "Status": "Pending"},
        {"Phase": "Retirement date", "Action": "Begin income withdrawals", "Status": "Future"},
        {"Phase": "Annual", "Action": "Review and rebalance", "Status": "Ongoing"}
    ]
    
    df_timeline = pd.DataFrame(timeline_data)
    st.dataframe(df_timeline, use_container_width=True)

# 6. Alert System
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

# 7. Export to Excel with Multiple Sheets
def create_excel_export(annual_data, sipp_ladder, isa_ladder):
    """Create comprehensive Excel export"""
    from io import BytesIO
    import pandas as pd
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Annual data
        pd.DataFrame(annual_data).to_excel(writer, sheet_name='Annual Analysis', index=False)
        
        # Bond ladders
        sipp_ladder.to_excel(writer, sheet_name='SIPP Bonds', index=False)
        isa_ladder.to_excel(writer, sheet_name='ISA Bonds', index=False)
        
        # Summary statistics
        summary_data = {
            'Metric': ['Total Net Income (Year 1)', 'Average Tax Rate', 'Final Pot Value'],
            'Value': [annual_data[0]['total_net_income'], 
                     np.mean([y['effective_tax_rate'] for y in annual_data]),
                     annual_data[-1]['total_remaining_pots']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# Add to main app
def enhanced_main():
    # ... existing main function code ...
    
    # Add enhanced features after main calculation
    if st.sidebar.button("🚀 Calculate Strategy", type="primary"):
        # ... existing calculation code ...
        
        # Add enhanced features
        create_alerts(annual_data)
        optimize_tax_efficiency()
        add_scenario_analysis()
        create_implementation_timeline()
        
        # Enhanced export
        st.subheader("📊 Enhanced Export")
        excel_data = create_excel_export(annual_data, sipp_ladder, isa_ladder)
        st.download_button(
            label="📊 Download Comprehensive Excel Report",
            data=excel_data,
            file_name=f"comprehensive_pension_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
