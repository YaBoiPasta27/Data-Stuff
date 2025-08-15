import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page configuration
st.set_page_config(
    page_title="CO2 Emissions Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
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

# Main header
st.markdown("""
<div class="main-header">
    <h1>CO2 Emissions and Climate Analysis Dashboard</h1>
    <p style="font-size: 1.2em; color: #666;">
        Interactive Analysis Platform for Sustainability Data
    </p>
    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; max-width: 600px; margin-left: auto; margin-right: auto;">
        <strong>Purpose:</strong> Monitor environmental metrics and provide data access for climate analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Interactive Controls")

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
st.sidebar.subheader("Animation Controls")
if st.sidebar.button("Play Animation"):
    # Placeholder for animation - in actual Streamlit, you'd use st.empty() and update it
    st.sidebar.info("Animation would cycle through years here")

current_year = st.sidebar.slider("Current Year", 1980, 2014, 2000)

# Main content area
# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Avg Temperature</h3>
        <h2>26.8°C</h2>
        <p style="color: #d32f2f;">▲ 1.2°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>CO2 Emissions</h3>
        <h2>2.1M tonnes</h2>
        <p style="color: #d32f2f;">▲ 15.3%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Growth Rate</h3>
        <h2>4.2%</h2>
        <p style="color: #388e3c;">▼ 0.8%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Countries</h3>
        <h2>195</h2>
    </div>
    """, unsafe_allow_html=True)

# Climate model equation
st.subheader("Climate Model Equation")
st.latex(r"Temperature = \beta_0 + \beta_1 \times CO2 + \beta_2 \times GDP + \varepsilon")
st.caption("Linear relationship between temperature rise, emissions, and economic factors")

# Main emissions chart
st.subheader(f"CO2 Emissions Over Time - {selected_country} Focus")

# Filter data based on selections
filtered_emissions = df_emissions[
    (df_emissions['year'] >= year_range[0]) & 
    (df_emissions['year'] <= year_range[1])
]

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot all countries with highlighting
countries = df_emissions['country'].unique()
for country in countries:
    country_data = filtered_emissions[filtered_emissions['country'] == country]
    
    y_values = country_data['log_value'] if show_log_scale else country_data['value']
    
    if country == selected_country:
        ax.plot(country_data['year'], y_values, 
                linewidth=3, color='#2563eb', label=country, alpha=1.0)
    else:
        ax.plot(country_data['year'], y_values, 
                linewidth=1, color='#64748b', alpha=0.4)

ax.set_title(f"CO2 Emissions Trends ({year_range[0]}-{year_range[1]})")
ax.set_xlabel("Year")
ax.set_ylabel("Log10(Emissions)" if show_log_scale else "Emissions (tonnes)")
ax.grid(True, alpha=0.3)
ax.legend()

if not show_log_scale:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

st.pyplot(fig)

# Temperature correlation analysis
if 'CO2' in selected_metrics and 'Temperature' in selected_metrics:
    st.subheader("Temperature vs CO2 Correlation - India")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        filtered_temp = df_temperature[
            (df_temperature['year'] >= year_range[0]) & 
            (df_temperature['year'] <= year_range[1])
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(filtered_temp['scaled_emissions'], filtered_temp['scaled_temp'], 
                  alpha=0.7, color='#2563eb', s=50)
        
        # Trend line
        z = np.polyfit(filtered_temp['scaled_emissions'], filtered_temp['scaled_temp'], 1)
        p = np.poly1d(z)
        ax.plot(filtered_temp['scaled_emissions'], p(filtered_temp['scaled_emissions']), 
                color='red', linewidth=2, linestyle='--', label='Trend Line')
        
        ax.set_xlabel('Scaled CO2 Emissions')
        ax.set_ylabel('Scaled Temperature')
        ax.set_title('Scaled Emissions vs Scaled Temperature')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    with col2:
        correlation = np.corrcoef(filtered_temp['scaled_emissions'], filtered_temp['scaled_temp'])[0, 1]
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #2563eb; font-size: 3em;">{correlation:.3f}</h1>
            <h3>Correlation Coefficient</h3>
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <p>Strong positive correlation between emissions and temperature rise</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Animated bar chart for current year
st.subheader(f"Top 10 Emitters in {current_year}")

current_year_data = df_emissions[df_emissions['year'] == current_year].nlargest(10, 'value')

fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart
countries_list = current_year_data['country'].tolist()
values_list = current_year_data['value'].tolist()

# Color bars, highlighting selected country
colors = ['#dc2626' if country == selected_country else '#2563eb' for country in countries_list]

bars = ax.barh(countries_list, values_list, color=colors)

ax.set_xlabel('CO2 Emissions (tonnes)')
ax.set_title(f'CO2 Emissions by Country in {current_year}')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

# Add value labels on bars
for i, (country, value) in enumerate(zip(countries_list, values_list)):
    ax.text(value + max(values_list)*0.01, i, f'{value/1e6:.1f}M', 
            va='center', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# Regional analysis
st.subheader("Regional CO2 Per Capita Analysis")

fig, ax = plt.subplots(figsize=(12, 6))

regions = df_regional['region'].tolist()
values = df_regional['co2_per_capita'].tolist()

# Create color map
colors = plt.cm.Reds(np.linspace(0.3, 1, len(regions)))

bars = ax.bar(regions, values, color=colors)

ax.set_ylabel('CO2 Per Capita (tonnes)')
ax.set_title('CO2 Emissions Per Capita by World Bank Region')
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{value:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# Code example
st.subheader("Sample Analysis Code")
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
st.subheader("Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Download CSV"):
        csv = df_emissions.to_csv(index=False)
        st.download_button(
            label="Download Emissions Data",
            data=csv,
            file_name="co2_emissions_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Charts"):
        st.info("Chart export functionality would be implemented here")

with col3:
    if st.button("Generate Report"):
        st.info("Report generation functionality would be implemented here")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "ENVECON 105: Data Tools for Sustainability • Streamlit Dashboard"
    "</div>", 
    unsafe_allow_html=True
)

# Additional information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.info(
    "This dashboard demonstrates interactive data visualization "
    "for sustainability analysis using Streamlit. It includes "
    "CO2 emissions trends, temperature correlations, and "
    "regional analysis features."
)

st.sidebar.subheader("Data Sources")
st.sidebar.caption(
    "Simulated data based on real-world patterns for "
    "educational purposes. In production, this would "
    "connect to live environmental databases."
)
