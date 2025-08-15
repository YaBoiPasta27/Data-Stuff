import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


st.plotly_chart(fig_regional, use_container_width=True)

# Regional Energy Trends Over Time
st.subheader("⚡ Energy Use Per Person by Region Over Time")

@st.cache_data
def generate_regional_energy_data():
    """Generate regional energy use data over time"""
    regions = [
        'South Asia', 'East Asia and Pacific', 'Europe and Central Asia',
        'North America', 'Middle East and North Africa', 'Sub-Saharan Africa',
        'Latin America and Caribbean'
    ]
    
    data = []
    np.random.seed(42)
    
    # Base energy consumption levels by region (kWh per person)
    base_levels = {
        'North America': 12000,
        'Europe and Central Asia': 6000,
        'Middle East and North Africa': 4000,
        'East Asia and Pacific': 3500,
        'Latin America and Caribbean': 2000,
        'South Asia': 800,
        'Sub-Saharan Africa': 500
    }
    
    for year in range(1990, 2018):
        for region in regions:
            base = base_levels[region]
            # Add growth trend and some variation
            growth_factor = 1 + (year - 1990) * 0.025
            seasonal_variation = np.sin((year - 1990) * 0.3) * 0.1
            noise = np.random.normal(0, 0.05)
            
            value = base * growth_factor * (1 + seasonal_variation + noise)
            value = max(100, value)  # Ensure minimum value
            
            data.append({
                'region': region,
                'year': year,
                'value': round(value, 2)
            })
    
    return pd.DataFrame(data)

df_regional_energy = generate_regional_energy_data()

# Filter for years before 2018
region_trend = df_regional_energy[df_regional_energy['year'] < 2018]

# Create the line plot
fig_energy_trend = go.Figure()

colors = px.colors.qualitative.Set1
for i, region in enumerate(region_trend['region'].unique()):
    subset = region_trend[region_trend['region'] == region]
    fig_energy_trend.add_trace(go.Scatter(
        x=subset['year'],
        y=subset['value'],
        mode='lines+markers',
        name=region,
        line=dict(width=3, color=colors[i % len(colors)]),
        marker=dict(size=5)
    ))

fig_energy_trend.update_layout(
    title='Energy Use Per Person by Region Over Time (Before 2018)',
    xaxis_title='Year',
    yaxis_title='kWh per Person',
    height=500,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_energy_trend, use_container_width=True)

# Multi-metric comparison: India vs Rest of World
st.subheader("🇮🇳 India vs Rest of World - Multi-Metric Analysis")

@st.cache_data
def generate_comparison_data():
    """Generate comparison data for multiple metrics"""
    # Generate GDP data
    gdp_data = []
    np.random.seed(42)
    countries = ['India', 'China', 'United States', 'Russia', 'Japan', 'Germany']
    
    for year in range(1990, 2015):
        for country in countries:
            if country == 'India':
                base_gdp = 400 + (year - 1990) * 25 + np.random.normal(0, 20)
            elif country == 'China':
                base_gdp = 350 + (year - 1990) * 35 + np.random.normal(0, 25)
            elif country == 'United States':
                base_gdp = 25000 + (year - 1990) * 800 + np.random.normal(0, 500)
            else:
                base_gdp = np.random.uniform(15000, 30000) + (year - 1990) * np.random.uniform(200, 600)
            
            gdp_data.append({
                'country': country,
                'year': year,
                'value': max(100, base_gdp)
            })
    
    # Generate Energy data (reuse some emissions logic but scale differently)
    energy_data = []
    for year in range(1990, 2015):
        for country in countries:
            if country == 'India':
                base_energy = 400 + (year - 1990) * 15 + np.random.normal(0, 30)
            elif country == 'China':
                base_energy = 800 + (year - 1990) * 45 + np.random.normal(0, 50)
            elif country == 'United States':
                base_energy = 12000 + (year - 1990) * 100 + np.random.normal(0, 200)
            else:
                base_energy = np.random.uniform(2000, 8000) + (year - 1990) * np.random.uniform(50, 150)
            
            energy_data.append({
                'country': country,
                'year': year,
                'value': max(100, base_energy)
            })
    
    return pd.DataFrame(gdp_data), pd.DataFrame(energy_data)

df_gdp, df_energy = generate_comparison_data()

# Create the multi-metric comparison
metrics_data = [
    ('CO2 Emissions (tonnes)', df_emissions[df_emissions['year'] >= 1990]),
    ('Energy Use (kWh per person)', df_energy),
    ('GDP per capita (USD)', df_gdp)
]

# Create subplots
fig_comparison = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        'CO2 Emissions - Rest of World', 'CO2 Emissions - India',
        'Energy Use - Rest of World', 'Energy Use - India', 
        'GDP per capita - Rest of World', 'GDP per capita - India'
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.1
)

for i, (metric_name, df_metric) in enumerate(metrics_data):
    row = i + 1
    
    # Filter data for the selected year range
    df_filtered = df_metric[
        (df_metric['year'] >= year_range[0]) & 
        (df_metric['year'] <= year_range[1])
    ]
    
    # Countries excluding India
    countries_excl_india = [c for c in df_filtered['country'].unique() if c != 'India']
    
    # Plot other countries (left column)
    for country in countries_excl_india:
        df_country = df_filtered[df_filtered['country'] == country]
        fig_comparison.add_trace(
            go.Scatter(
                x=df_country['year'],
                y=df_country['value'],
                mode='lines',
                name=country,
                line=dict(color='rgba(0,0,0,0.3)', width=2),
                showlegend=False
            ),
            row=row, col=1
        )
    
    # Plot India (right column)
    df_india = df_filtered[df_filtered['country'] == 'India']
    if not df_india.empty:
        fig_comparison.add_trace(
            go.Scatter(
                x=df_india['year'],
                y=df_india['value'],
                mode='lines+markers',
                name='India',
                line=dict(color='#dc2626', width=3),
                marker=dict(size=6),
                showlegend=False
            ),
            row=row, col=2
        )

# Update layout
fig_comparison.update_layout(
    height=800,
    title_text="Distribution of Indicators by Year and Value",
    title_x=0.5,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Update x-axis labels for bottom row
fig_comparison.update_xaxes(title_text="Year", row=3, col=1)
fig_comparison.update_xaxes(title_text="Year", row=3, col=2)

# Update y-axis labels
fig_comparison.update_yaxes(title_text="CO2 Emissions", row=1, col=1)
fig_comparison.update_yaxes(title_text="Energy Use", row=2, col=1)
fig_comparison.update_yaxes(title_text="GDP per capita", row=3, col=1)

st.plotly_chart(fig_comparison, use_container_width=True)

# Code example - FIXED
st.subheader("💻 Sample Analysis Code")
st.markdown("""
