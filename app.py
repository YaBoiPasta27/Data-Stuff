import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, Globe, Thermometer, BarChart3, Map, Calculator, Upload, Download, Play, Pause } from 'lucide-react';

const Dashboard = () => {
  const [selectedCountry, setSelectedCountry] = useState('India');
  const [selectedRegion, setSelectedRegion] = useState('South Asia');
  const [showLogScale, setShowLogScale] = useState(false);
  const [selectedMetrics, setSelectedMetrics] = useState(['CO2', 'Temperature']);
  const [yearRange, setYearRange] = useState([1980, 2014]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentYear, setCurrentYear] = useState(1980);

  // Data generation functions based on your analysis
  const generateEmissionsData = () => {
    const countries = ['India', 'China', 'United States', 'Russia', 'Japan', 'Germany', 'Iran', 'South Korea', 'Saudi Arabia', 'Canada'];
    const data = [];
    
    for (let year = 1900; year <= 2020; year++) {
      countries.forEach(country => {
        let baseValue = Math.random() * 500000;
        if (country === 'China') baseValue *= 12;
        if (country === 'United States') baseValue *= 10;
        if (country === 'India') baseValue *= 6;
        
        const growthFactor = Math.pow(1.025, year - 1950);
        const value = baseValue * growthFactor * (1 + Math.sin((year - 1950) / 10) * 0.1);
        
        data.push({
          country,
          year,
          value: Math.round(Math.max(10000, value)),
          logValue: Math.log10(Math.max(10000, value))
        });
      });
    }
    return data;
  };

  const generateTemperatureData = () => {
    const data = [];
    for (let year = 1980; year <= 2014; year++) {
      const temp = 25.2 + (year - 1980) * 0.04 + Math.sin((year - 1980) * 0.3) * 0.8 + Math.random() * 0.3;
      const emissions = 800000 + (year - 1980) * 45000 + Math.random() * 150000;
      data.push({
        year,
        temperature: parseFloat(temp.toFixed(2)),
        emissions: Math.round(emissions),
        scaledTemp: ((temp - 25.5) / 1.2),
        scaledEmissions: ((emissions - 1600000) / 600000)
      });
    }
    return data;
  };

  const generateRegionalData = () => {
    const regions = [
      'South Asia', 'East Asia & Pacific', 'Europe & Central Asia', 
      'North America', 'Middle East & North Africa', 'Sub-Saharan Africa', 
      'Latin America & Caribbean'
    ];
    return regions.map(region => ({
      region,
      CO2: Math.random() * 2000000 + 500000,
      GDP: Math.random() * 15000 + 5000,
      Population: Math.random() * 1000 + 100,
      CO2PerCapita: Math.random() * 8 + 2
    }));
  };

  const emissionsData = useMemo(generateEmissionsData, []);
  const temperatureData = useMemo(generateTemperatureData, []);
  const regionalData = useMemo(generateRegionalData, []);

  // Animation effect for year progression
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentYear(prev => {
          if (prev >= yearRange[1]) {
            setIsPlaying(false);
            return yearRange[0];
          }
          return prev + 1;
        });
      }, 200);
    }
    return () => clearInterval(interval);
  }, [isPlaying, yearRange]);

  // Streamlit-style metric component
  const StreamlitMetric = ({ label, value, delta, unit = "" }) => (
    <div className="bg-white p-4 rounded-lg border border-gray-200 text-center">
      <div className="text-sm text-gray-600 font-medium mb-1">{label}</div>
      <div className="text-2xl font-bold text-gray-900 mb-1">
        {typeof value === 'number' ? value.toLocaleString() : value} {unit}
      </div>
      {delta && (
        <div className={`text-sm font-medium ${delta > 0 ? 'text-red-500' : 'text-green-500'}`}>
          {delta > 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}{unit === '%' ? 'pp' : unit}
        </div>
      )}
    </div>
  );

  // Streamlit-style sidebar
  const Sidebar = () => (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">Interactive Controls</h3>
      
      {/* Country Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Country
        </label>
        <select 
          value={selectedCountry} 
          onChange={(e) => setSelectedCountry(e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {['India', 'China', 'United States', 'Russia', 'Japan', 'Germany', 'Iran', 'South Korea', 'Saudi Arabia', 'Canada'].map(country => (
            <option key={country} value={country}>{country}</option>
          ))}
        </select>
      </div>

      {/* Multi-select for Metrics */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Metrics
        </label>
        <div className="space-y-2">
          {['CO2', 'Temperature', 'GDP', 'Population'].map(metric => (
            <label key={metric} className="flex items-center">
              <input
                type="checkbox"
                checked={selectedMetrics.includes(metric)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedMetrics([...selectedMetrics, metric]);
                  } else {
                    setSelectedMetrics(selectedMetrics.filter(m => m !== metric));
                  }
                }}
                className="mr-2"
              />
              <span className="text-sm">{metric}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Log Scale Toggle */}
      <div className="mb-6">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={showLogScale}
            onChange={(e) => setShowLogScale(e.target.checked)}
            className="mr-2"
          />
          <span className="text-sm font-medium">Use Log Scale</span>
        </label>
      </div>

      {/* Year Range Slider */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Year Range: {yearRange[0]} - {yearRange[1]}
        </label>
        <div className="space-y-2">
          <input
            type="range"
            min="1980"
            max="2014"
            value={yearRange[0]}
            onChange={(e) => setYearRange([parseInt(e.target.value), yearRange[1]])}
            className="w-full"
          />
          <input
            type="range"
            min="1980"
            max="2014"
            value={yearRange[1]}
            onChange={(e) => setYearRange([yearRange[0], parseInt(e.target.value)])}
            className="w-full"
          />
        </div>
      </div>

      {/* Animation Controls */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Animation Controls
        </label>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            <span className="ml-1">{isPlaying ? 'Pause' : 'Play'}</span>
          </button>
          <span className="text-sm text-gray-600">Year: {currentYear}</span>
        </div>
      </div>

      {/* File Upload Simulation */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Data
        </label>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center">
          <Upload className="mx-auto text-gray-400 mb-2" size={24} />
          <p className="text-sm text-gray-600">Upload CSV data file</p>
          <p className="text-xs text-gray-400">(Simulation only)</p>
        </div>
      </div>
    </div>
  );

  const filteredTemperatureData = temperatureData.filter(d => 
    d.year >= yearRange[0] && d.year <= yearRange[1]
  );

  const currentYearData = emissionsData
    .filter(d => d.year === currentYear)
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Streamlit-style Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Data Tools for Sustainability Dashboard
          </h1>
          <p className="text-gray-600 text-lg mb-4">
            Interactive CO2 Emissions & Climate Analysis Platform
          </p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-2xl mx-auto">
            <p className="text-blue-800 text-sm">
              <strong>Purpose:</strong> Deliver sustainability insights, monitor environmental metrics, 
              and provide interactive data access for policy makers and researchers.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Streamlit-style Sidebar */}
          <div className="lg:col-span-1">
            <Sidebar />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Streamlit-style Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <StreamlitMetric 
                label="Avg Temperature" 
                value={26.8} 
                delta={1.2}
                unit="°C" 
              />
              <StreamlitMetric 
                label="CO2 Emissions" 
                value="2.1M" 
                delta={15.3}
                unit="tonnes" 
              />
              <StreamlitMetric 
                label="Growth Rate" 
                value={4.2} 
                delta={-0.8}
                unit="%" 
              />
              <StreamlitMetric 
                label="Countries" 
                value={195} 
                unit="" 
              />
            </div>

            {/* LaTeX Equation Display (Streamlit style) */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">📐 Climate Model Equation</h3>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-lg font-mono">
                  Temperature = β₀ + β₁ × CO2 + β₂ × GDP + ε
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Linear relationship between temperature rise, emissions, and economic factors
                </p>
              </div>
            </div>

            {/* Interactive Chart with Country Highlighting */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">
                CO2 Emissions Over Time - {selectedCountry} Focus
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="year" domain={[1980, 2014]} type="number" />
                  <YAxis 
                    scale={showLogScale ? 'log' : 'linear'}
                    domain={showLogScale ? ['auto', 'auto'] : [0, 'dataMax']}
                    tickFormatter={(value) => 
                      showLogScale ? `10^${value.toFixed(1)}` : `${(value/1e6).toFixed(1)}M`
                    }
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      showLogScale ? `10^${value.toFixed(2)}` : `${(value/1e6).toFixed(1)}M tonnes`, 
                      name
                    ]}
                  />
                  <Legend />
                  {['India', 'China', 'United States', 'Russia', 'Japan'].map((country, index) => {
                    const countryData = emissionsData
                      .filter(d => d.country === country && d.year >= yearRange[0] && d.year <= yearRange[1]);
                    return (
                      <Line 
                        key={country}
                        dataKey={showLogScale ? "logValue" : "value"}
                        data={countryData}
                        stroke={country === selectedCountry ? '#2563eb' : '#64748b'}
                        strokeWidth={country === selectedCountry ? 3 : 1}
                        opacity={country === selectedCountry ? 1 : 0.4}
                        dot={false}
                        name={country}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Correlation Analysis */}
            {selectedMetrics.includes('CO2') && selectedMetrics.includes('Temperature') && (
              <div className="bg-white p-6 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-4">
                  🔗 Temperature vs CO2 Correlation - {selectedCountry}
                </h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={filteredTemperatureData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="scaledEmissions" name="Scaled Emissions" />
                      <YAxis dataKey="scaledTemp" name="Scaled Temperature" />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        formatter={(value, name) => [value.toFixed(3), name]}
                      />
                      <Scatter fill="#2563eb" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <div className="flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-4xl font-bold text-blue-600 mb-2">0.847</div>
                      <div className="text-gray-600 mb-4">Correlation Coefficient</div>
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <p className="text-sm text-gray-700">
                          Strong positive correlation indicates significant relationship between 
                          emissions and temperature rise in {selectedCountry}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Animated Bar Chart */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">
                Top 10 Emitters in {currentYear} (Animation: {isPlaying ? 'Playing' : 'Paused'})
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={currentYearData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" tickFormatter={(value) => `${(value/1e6).toFixed(1)}M`} />
                  <YAxis dataKey="country" type="category" width={100} />
                  <Tooltip formatter={(value) => [`${(value/1e6).toFixed(1)}M tonnes`, 'Emissions']} />
                  <Bar dataKey="value" fill="#2563eb">
                    {currentYearData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.country === selectedCountry ? '#dc2626' : '#2563eb'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Regional Stacked Area Chart */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Regional Analysis</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={regionalData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="region" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="CO2PerCapita" fill="#ef4444" name="CO2 Per Capita (tons)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Code Display (Streamlit style) */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Sample Analysis Code</h3>
              <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                <div># Data preparation for sustainability analysis</div>
                <div>import pandas as pd</div>
                <div>import streamlit as st</div>
                <div>import plotly.express as px</div>
                <div></div>
                <div># Load CO2 emissions data</div>
                <div>df = pd.read_csv('emissions_data.csv')</div>
                <div></div>
                <div># Calculate correlation</div>
                <div>correlation = df['emissions'].corr(df['temperature'])</div>
                <div>st.write(f"Correlation: {'{correlation:.3f}'}")</div>
              </div>
            </div>

            {/* Download Section */}
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Export Results</h3>
              <div className="flex space-x-4">
                <button className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                  <Download size={16} className="mr-2" />
                  Download CSV
                </button>
                <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                  <Download size={16} className="mr-2" />
                  Export Charts
                </button>
                <button className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                  <Download size={16} className="mr-2" />
                  Generate Report
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Built with React • Inspired by Streamlit Design Principles • ENVECON 105: Data Tools for Sustainability</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
