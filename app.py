import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="CO2 Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED for better visibility
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        color: #333333;
    }
    .main-header h1 {
        color: #1f2937;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .main-header p {
        color: #4b5563;
        font-size: 1.2em;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #374151;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card h2 {
        color: #1f2937;
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-increase {
        color: #dc2626 !important;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .metric-decrease {
        color: #059669 !important;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .metric-neutral {
        color: #6b7280 !important;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .code-block {
        background-color: #1f2937;
        color: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.75rem;
        font-family: 'Courier New', monospace;
        border: 1px solid #374151;
        line-height: 1.6;
    }
    .purpose-box {
        background-color: #eff6ff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid #bfdbfe;
    }
    .purpose-box strong {
        color: #1e40af;
    }
    .correlation-display {
        text-align: center;
        padding: 2rem;
        background-color: #f8fafc;
        border-radius: 0.75rem;
        border: 2px solid #e2e8f0;
    }
    .correlation-number {
        color: #1e40af;
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .correlation-label {
        color: #334155;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .correlation-description {
        background-color: #dbeafe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        color: #1e40af;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Data generation functions
@st.cache_data
def generate_emissions_data():
    """Generate CO2 emissions data for multiple countries"""
    countries = ['India', 'China', 'United States', 'Russia', 'Japan', 
                'Germany', 'Iran', 'South Korea', 'Saudi Arabia', 'Canada']
    data = []
    
    np.random.seed(42)  # For reproducible results
    
    for year in range(1980, 2015):
        for idx, country in enumerate(countries):
            base_value = 500000 + idx * 200000
            if country == 'China':
                base_value = 8000000
            elif country == 'United States':
                base_value = 5000000
            elif country == 'India':
                base_value = 2000000
            
            growth_factor = 1 + (year - 1980) * 0.02
            noise = np.random.normal(1, 0.1)
            value = int(base_value * growth_factor * noise)
            
            data.append({
                'country': country,
                'year': year,
                'value': max(10000, value),
                'log_value': np.log10(max(10000, value))
            })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_temperature_data():
    """Generate temperature data for India"""
    data = []
    np.random.seed(42)
    
    for year in range(1980, 2015):
        temp = 25.2 + (year - 1980) * 0.04 + np.sin((year - 1980) * 0.3) * 0.8 + np.random.normal(0, 0.3)
        emissions = 800000 + (year - 1980) * 45000 + np.random.normal(0, 150000)
        
        data.append({
            'year': year,
            'temperature': round(temp, 2),
            'emissions': max(500000, int(emissions)),
            'scaled_temp': (temp - 25.5) / 1.2,
            'scaled_emissions': (emissions - 1600000) / 600000
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_regional_data():
    """Generate regional CO2 data"""
    regions = [
        'South Asia', 'East Asia and Pacific', 'Europe and Central Asia',
        'North America', 'Middle East and North Africa', 'Sub-Saharan Africa',
        'Latin America and Caribbean'
    ]
    
    data = []
    values = [1.8, 7.2, 6.8, 15.5, 8.9, 0.8, 3.1]
    emissions = [2500000, 15000000, 4200000, 6800000, 2100000, 800000, 1500000]
    
    for i, region in enumerate(regions):
        data.append({
            'region': region,
            'co2_per_capita': values[i],
            'total_emissions': emissions[i]
        })
    
    return pd.DataFrame(data)

# Load data
df_emissions = generate_emissions_data()
df_temperature = generate_temperature_data()
df_regional = generate_regional_data()

# Main header - FIXED
st.markdown("""
<div class="main-header">
    <h1>🌍 CO2 Emissions and Climate Analysis Dashboard</h1>
    <p>Interactive Analysis Platform for Sustainability Data</p>
    <div class="purpose-box">
        <strong>Purpose:</strong> Monitor environmental metrics and provide data access for climate analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ Interactive Controls")

# Country selection
selected_country = st.sidebar.selectbox(
    "Select Country",
    options=['India', 'China', 'United States', 'Russia', 'Japan', 
            'Germany', 'Iran', 'South Korea', 'Saudi Arabia', 'Canada'],
    index=0
)

# Metrics selection
selected_metrics = st.sidebar.multiselect(
    "Select Metrics",
    options=['CO2', 'Temperature', 'GDP', 'Population'],
    default=['CO2', 'Temperature']
)

# Log scale toggle
show_log_scale = st.sidebar.checkbox("Use Log Scale")

# Year range
year_range = st.sidebar.slider(
    "Year Range",
    min_value=1980,
    max_value=2014,
    value=(1980, 2014),
    step=1
)

# Animation controls
st.sidebar.subheader("🎬 Animation Controls")
if st.sidebar.button("Play Animation"):
    st.sidebar.info("Animation would cycle through years here")

current_year = st.sidebar.slider("Current Year", 1980, 2014, 2000)

# Main content area
# Metrics row - FIXED with better visibility
st.subheader("📊 Key Metrics Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Avg Temperature</h3>
        <h2>26.8°C</h2>
        <p class="metric-increase">▲ 1.2°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>CO2 Emissions</h3>
        <h2>2.1M tonnes</h2>
        <p class="metric-increase">▲ 15.3%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Growth Rate</h3>
        <h2>4.2%</h2>
        <p class="metric-decrease">▼ 0.8%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Countries</h3>
        <h2>195</h2>
        <p class="metric-neutral">—</p>
    </div>
    """, unsafe_allow_html=True)

# Climate model equation
st.subheader("🧮 Climate Model Equation")
st.latex(r"Temperature = \beta_0 + \beta_1 \times CO2 + \beta_2 \times GDP + \varepsilon")
st.caption("Linear relationship between temperature rise, emissions, and economic factors")

# Main emissions chart
st.subheader(f"📈 CO2 Emissions Over Time - {selected_country} Focus")

# Filter data based on selections
filtered_emissions = df_emissions[
    (df_emissions['year'] >= year_range[0]) & 
    (df_emissions['year'] <= year_range[1])
]

# Create the main emissions plot
fig_emissions = go.Figure()

# Add lines for all countries, highlighting selected
countries = df_emissions['country'].unique()
for country in countries:
    country_data = filtered_emissions[filtered_emissions['country'] == country]
    
    y_values = country_data['log_value'] if show_log_scale else country_data['value']
    
    fig_emissions.add_trace(go.Scatter(
        x=country_data['year'],
        y=y_values,
        mode='lines+markers',
        name=country,
        line=dict(
            width=4 if country == selected_country else 2,
            color='#dc2626' if country == selected_country else '#64748b'
        ),
        marker=dict(size=6 if country == selected_country else 4),
        opacity=1.0 if country == selected_country else 0.6
    ))

fig_emissions.update_layout(
    title=f"CO2 Emissions Trends ({year_range[0]}-{year_range[1]})",
    xaxis_title="Year",
    yaxis_title="Log10(Emissions)" if show_log_scale else "Emissions (tonnes)",
    height=500,
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_emissions, use_container_width=True)

# Temperature correlation analysis
if 'CO2' in selected_metrics and 'Temperature' in selected_metrics:
    st.subheader("🌡️ Temperature vs CO2 Correlation - India")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        filtered_temp = df_temperature[
            (df_temperature['year'] >= year_range[0]) & 
            (df_temperature['year'] <= year_range[1])
        ]
        
        fig_corr = px.scatter(
            filtered_temp,
            x='scaled_emissions',
            y='scaled_temp',
            title='Scaled Emissions vs Scaled Temperature',
            labels={'scaled_emissions': 'Scaled CO2 Emissions', 'scaled_temp': 'Scaled Temperature'},
            color_discrete_sequence=['#2563eb']
        )
        
        # Add trend line
        fig_corr.add_trace(go.Scatter(
            x=filtered_temp['scaled_emissions'],
            y=np.poly1d(np.polyfit(filtered_temp['scaled_emissions'], filtered_temp['scaled_temp'], 1))(filtered_temp['scaled_emissions']),
            mode='lines',
            name='Trend Line',
            line=dict(color='#dc2626', width=3)
        ))
        
        fig_corr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        correlation = np.corrcoef(filtered_temp['scaled_emissions'], filtered_temp['scaled_temp'])[0, 1]
        
        st.markdown(f"""
        <div class="correlation-display">
            <div class="correlation-number">{correlation:.3f}</div>
            <div class="correlation-label">Correlation Coefficient</div>
            <div class="correlation-description">
                Strong positive correlation between emissions and temperature rise
            </div>
        </div>
        """, unsafe_allow_html=True)

# Animated bar chart for current year
st.subheader(f"🏆 Top 10 Emitters in {current_year}")

current_year_data = df_emissions[df_emissions['year'] == current_year].nlargest(10, 'value')

# Create color map
colors = ['#dc2626' if country == selected_country else '#2563eb' for country in current_year_data['country']]

fig_bar = px.bar(
    current_year_data,
    x='value',
    y='country',
    orientation='h',
    title=f'CO2 Emissions by Country in {current_year}',
    labels={'value': 'CO2 Emissions (tonnes)', 'country': 'Country'},
    color='country',
    color_discrete_sequence=colors
)

fig_bar.update_layout(
    height=500,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

st.plotly_chart(fig_bar, use_container_width=True)

# Regional analysis
st.subheader("🌐 Regional CO2 Per Capita Analysis")

fig_regional = px.bar(
    df_regional,
    x='region',
    y='co2_per_capita',
    title='CO2 Emissions Per Capita by World Bank Region',
    labels={'co2_per_capita': 'CO2 Per Capita (tonnes)', 'region': 'Region'},
    color='co2_per_capita',
    color_continuous_scale='Reds'
)

fig_regional.update_layout(
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis={'tickangle': 45}
)

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
<div class="code-block">
# Data preparation for sustainability analysis<br>
import pandas as pd<br>
import streamlit as st<br>
import plotly.express as px<br>
<br>
# Load CO2 emissions data<br>
df = pd.read_csv('emissions_data.csv')<br>
<br>
# Calculate correlation<br>
correlation = df['emissions'].corr(df['temperature'])<br>
st.write(f"Correlation: {correlation:.3f}")<br>
<br>
# Create interactive plot<br>
fig = px.scatter(df, x='emissions', y='temperature')<br>
st.plotly_chart(fig)
</div>
""", unsafe_allow_html=True)

# Export section
st.subheader("📤 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Download CSV", use_container_width=True):
        csv = df_emissions.to_csv(index=False)
        st.download_button(
            label="Download Emissions Data",
            data=csv,
            file_name="co2_emissions_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Charts", use_container_width=True):
        st.info("Chart export functionality would be implemented here")

with col3:
    if st.button("📋 Generate Report", use_container_width=True):
        st.info("Report generation functionality would be implemented here")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; padding: 1rem; font-weight: 500;'>"
    "🎓 ENVECON 105: Data Tools for Sustainability • Streamlit Dashboard"
    "</div>", 
    unsafe_allow_html=True
)

# Additional information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About This Dashboard")
st.sidebar.info(
    "This dashboard demonstrates interactive data visualization "
    "for sustainability analysis using Streamlit. It includes "
    "CO2 emissions trends, temperature correlations, and "
    "regional analysis features."
)

st.sidebar.subheader("📊 Data Sources")
st.sidebar.caption(
    "Simulated data based on real-world patterns for "
    "educational purposes. In production, this would "
    "connect to live environmental databases."
)
