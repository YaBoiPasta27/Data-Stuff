# Enhanced line graph sections with better visibility for selected country

# Main emissions chart
st.subheader(f"CO2 Emissions Over Time - {selected_country} Focus")

# Filter data based on selections
filtered_emissions = df_emissions[
    (df_emissions['year'] >= year_range[0]) & 
    (df_emissions['year'] <= year_range[1])
]

if not filtered_emissions.empty:
    # Create the main emissions plot
    fig_emissions = go.Figure()

    # Get unique countries in the filtered data
    countries = filtered_emissions['country'].unique()
    
    # First pass: Add all OTHER countries (background lines)
    for country in countries:
        if country != selected_country:  # Skip selected country for now
            country_data = filtered_emissions[filtered_emissions['country'] == country]
            
            if not country_data.empty:
                y_values = country_data['log_value'] if show_log_scale else country_data['value']
                
                fig_emissions.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=y_values,
                    mode='lines',
                    name=country,
                    line=dict(
                        width=0.8,  # Thinner background lines
                        color='#cbd5e1'  # Light gray
                    ),
                    opacity=0.3,  # More transparent
                    hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>Emissions: %{{y:.2e}}<extra></extra>',
                    showlegend=False  # Hide from legend to reduce clutter
                ))
    
    # Second pass: Add the SELECTED country on top
    if selected_country in countries:
        selected_data = filtered_emissions[filtered_emissions['country'] == selected_country]
        
        if not selected_data.empty:
            y_values = selected_data['log_value'] if show_log_scale else selected_data['value']
            
            fig_emissions.add_trace(go.Scatter(
                x=selected_data['year'],
                y=y_values,
                mode='lines+markers',  # Add markers for selected country
                name=f"{selected_country} (Selected)",
                line=dict(
                    width=4,  # Much thicker line
                    color='#dc2626'  # Bright red
                ),
                marker=dict(
                    size=6,
                    color='#dc2626',
                    symbol='circle',
                    line=dict(width=2, color='white')  # White border on markers
                ),
                opacity=1.0,
                hovertemplate=f'<b>{selected_country}</b><br>Year: %{{x}}<br>Emissions: %{{y:.2e}}<extra></extra>'
            ))

    fig_emissions.update_layout(
        title=f"CO2 Emissions Trends ({year_range[0]}-{year_range[1]}) - {selected_country} Highlighted",
        xaxis_title="Year",
        yaxis_title="Log10(Emissions)" if show_log_scale else "Emissions (tonnes)",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_emissions, use_container_width=True)
else:
    st.warning("No emissions data available for the selected time period.")

# EUPP Over Time Chart
st.subheader(f"Energy Use Per Person (EUPP) Over Time - {selected_country} Focus")

# Filter EUPP data
filtered_eupp = df_eupp[
    (df_eupp['year'] >= year_range[0]) & 
    (df_eupp['year'] <= year_range[1])
]

if not filtered_eupp.empty:
    fig_eupp = go.Figure()

    # Get unique countries in the filtered EUPP data
    eupp_countries = filtered_eupp['country'].unique()
    
    # First pass: Add all OTHER countries (background lines)
    for country in eupp_countries:
        if country != selected_country:  # Skip selected country for now
            country_data = filtered_eupp[filtered_eupp['country'] == country]
            
            if not country_data.empty:
                # Add log transformation if needed
                if show_log_scale:
                    country_data = country_data.copy()
                    country_data['log_value'] = np.log10(country_data['value'].replace(0, np.nan))
                    y_values = country_data['log_value']
                else:
                    y_values = country_data['value']
                
                fig_eupp.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=y_values,
                    mode='lines',
                    name=country,
                    line=dict(
                        width=0.8,  # Thinner background lines
                        color='#e2e8f0'  # Light gray
                    ),
                    opacity=0.3,  # More transparent
                    hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>EUPP: %{{y:.1f}}<extra></extra>',
                    showlegend=False  # Hide from legend
                ))

    # Second pass: Add the SELECTED country on top
    if selected_country in eupp_countries:
        selected_data = filtered_eupp[filtered_eupp['country'] == selected_country]
        
        if not selected_data.empty:
            # Add log transformation if needed
            if show_log_scale:
                selected_data = selected_data.copy()
                selected_data['log_value'] = np.log10(selected_data['value'].replace(0, np.nan))
                y_values = selected_data['log_value']
            else:
                y_values = selected_data['value']
            
            fig_eupp.add_trace(go.Scatter(
                x=selected_data['year'],
                y=y_values,
                mode='lines+markers',  # Add markers for selected country
                name=f"{selected_country} (Selected)",
                line=dict(
                    width=4,  # Much thicker line
                    color='#2563eb'  # Bright blue
                ),
                marker=dict(
                    size=6,
                    color='#2563eb',
                    symbol='circle',
                    line=dict(width=2, color='white')  # White border on markers
                ),
                opacity=1.0,
                hovertemplate=f'<b>{selected_country}</b><br>Year: %{{x}}<br>EUPP: %{{y:.1f}}<extra></extra>'
            ))

    fig_eupp.update_layout(
        title=f"Energy Use Per Person Trends ({year_range[0]}-{year_range[1]}) - {selected_country} Highlighted",
        xaxis_title="Year",
        yaxis_title="Log10(EUPP)" if show_log_scale else "Energy Use Per Person (GJ/capita/year)",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_eupp, use_container_width=True)
else:
    st.warning("No EUPP data available for the selected time period.")

# GDP Change Over Time Chart
st.subheader(f"GDP Growth Rate Over Time - {selected_country} Focus")

# Filter GDP data
filtered_gdp = df_gdp[
    (df_gdp['year'] >= year_range[0]) & 
    (df_gdp['year'] <= year_range[1])
]

if not filtered_gdp.empty:
    fig_gdp = go.Figure()

    # Get unique countries in the filtered GDP data
    gdp_countries = filtered_gdp['country'].unique()
    
    # First pass: Add all OTHER countries (background lines)
    for country in gdp_countries:
        if country != selected_country:  # Skip selected country for now
            country_data = filtered_gdp[filtered_gdp['country'] == country]
            
            if not country_data.empty:
                # Convert to percentage for display
                y_values = country_data['value'] * 100
                
                fig_gdp.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=y_values,
                    mode='lines',
                    name=country,
                    line=dict(
                        width=0.8,  # Thinner background lines
                        color='#f1f5f9'  # Light gray
                    ),
                    opacity=0.3,  # More transparent
                    hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>GDP Growth: %{{y:.1f}}%<extra></extra>',
                    showlegend=False  # Hide from legend
                ))

    # Second pass: Add the SELECTED country on top
    if selected_country in gdp_countries:
        selected_data = filtered_gdp[filtered_gdp['country'] == selected_country]
        
        if not selected_data.empty:
            # Convert to percentage for display
            y_values = selected_data['value'] * 100
            
            fig_gdp.add_trace(go.Scatter(
                x=selected_data['year'],
                y=y_values,
                mode='lines+markers',  # Add markers for selected country
                name=f"{selected_country} (Selected)",
                line=dict(
                    width=4,  # Much thicker line
                    color='#16a34a'  # Bright green
                ),
                marker=dict(
                    size=6,
                    color='#16a34a',
                    symbol='circle',
                    line=dict(width=2, color='white')  # White border on markers
                ),
                opacity=1.0,
                hovertemplate=f'<b>{selected_country}</b><br>Year: %{{x}}<br>GDP Growth: %{{y:.1f}}%<extra></extra>'
            ))

    # Add horizontal line at 0% growth
    fig_gdp.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.7, line_width=2)

    fig_gdp.update_layout(
        title=f"GDP Growth Rate Trends ({year_range[0]}-{year_range[1]}) - {selected_country} Highlighted",
        xaxis_title="Year",
        yaxis_title="GDP Growth Rate (%)",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_gdp, use_container_width=True)
else:
    st.warning("No GDP data available for the selected time period.")
