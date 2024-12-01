import streamlit as st
import math
from scipy.stats import norm 
import mibian as ml
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

st.title("Black-Scholes Theoretical Option Pricer")

# Variables
st.sidebar.write("## Input Data")
S = st.sidebar.number_input(label="Price of underlying:", value=100.00, min_value=0.01, max_value=None)  # Price of underlying
K = st.sidebar.number_input(label="Strike price:", value=95.00,min_value=0.01, max_value=None)  # Strike price
T = st.sidebar.number_input('Time to Maturity (days):', value= 365,min_value=1, max_value=None)  # Time to maturity years
r = st.sidebar.number_input(label="Risk free rate (%):", value=3.50,min_value=None, max_value=None)  # Risk free rate
vol = st.sidebar.number_input(label="Volatility (σ %):",value=25.00,min_value=0.01, max_value=None)  # Volatility (σ)
st.sidebar.text("By Senan Skalkos")

r = r / 100
vol = vol / 100
T = T / 365

# Calculate d1 and d2
d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
d2 = d1 - (vol * math.sqrt(T))

# Calculate the call option price
C = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# Calculate the put option price
P = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Display price results
col1, col2 = st.columns(2)
col1.metric(label="Call price", value=f"${C: .2f}")
col2.metric(label="Put price", value=f"${P: .2f}")

# Function to color negative values in red and positive values in green in the dataframe
def color_negatives(val):
    color = ('#FA7070' if val < 0 else '#799351')  
    return f'color: {color}'

# Preparing data for mibianlib to calculate the Greeks using the correct units
option_data = [S, K, r * 100, T * 365]  # Spot price, strike price, risk-free rate (%), time to expiration (days)
call_option = ml.BS(option_data, vol * 100)
put_option = ml.BS(option_data, vol * 100)

# Extracting the Greeks for both Call and Put options
call_greeks = {
    'Delta': call_option.callDelta,
    'Gamma': call_option.gamma,
    'Theta': call_option.callTheta,
    'Vega': call_option.vega,
    'Rho': call_option.callRho
}

put_greeks = {
    'Delta': put_option.putDelta,
    'Gamma': put_option.gamma,
    'Theta': put_option.putTheta,
    'Vega': put_option.vega,
    'Rho': put_option.putRho
}

# Create a table to display the Greeks
greek_df = pd.DataFrame({'Call Greeks': call_greeks, 'Put Greeks': put_greeks})

# Displaying the Greeks table
st.subheader("Option Greeks Summary")
st.dataframe(greek_df.style.applymap(color_negatives), use_container_width=True)

# Greek Visualisation Function
def greek_visualisation(spot, strike, r, T, sigma, greek):
    fig = go.Figure()
    min_s = spot * (0.5)
    max_s = spot * (1.5)
    
    spot_values = np.linspace(min_s, max_s, 200)

    call_greek_values = []
    put_greek_values = []
    
    for s in spot_values:
        option = ml.BS([s, strike, r * 100, T * 365], sigma * 100) 
        call_greek_values.append(option.callDelta if greek == 'Delta' else
                                  option.gamma if greek == 'Gamma' else
                                  option.callTheta if greek == 'Theta' else
                                  option.vega if greek == 'Vega' else
                                  option.callRho)
        
        put_greek_values.append(option.putDelta if greek == 'Delta' else
                                 option.gamma if greek == 'Gamma' else
                                 option.putTheta if greek == 'Theta' else
                                 option.vega if greek == 'Vega' else
                                 option.putRho)

    if greek in ['Gamma', 'Vega']:
        fig.add_trace(go.Scatter(x=spot_values, y=call_greek_values, mode='lines', name=f'{greek.capitalize()}', line=dict(color='#799351', width=3)))
    else:
        fig.add_trace(go.Scatter(x=spot_values, y=call_greek_values, mode='lines', name=f'Call {greek.capitalize()}', line=dict(color='#799351', width=3)))
        fig.add_trace(go.Scatter(x=spot_values, y=put_greek_values, mode='lines', name=f'Put {greek.capitalize()}', line=dict(color='#FA7070', width=3)))

    current_option = ml.BS([spot, strike, r * 100, T * 365], sigma * 100)
    current_call_greek_value = current_option.callDelta if greek == 'Delta' else \
                                current_option.gamma if greek == 'Gamma' else \
                                current_option.callTheta if greek == 'Theta' else \
                                current_option.vega if greek == 'Vega' else \
                                current_option.callRho

    current_put_greek_value = current_option.putDelta if greek == 'Delta' else \
                               current_option.gamma if greek == 'Gamma' else \
                               current_option.putTheta if greek == 'Theta' else \
                               current_option.vega if greek == 'Vega' else \
                               current_option.putRho

    fig.add_trace(go.Scatter(x=[spot], y=[current_call_greek_value], mode='markers', name=f'Current Call {greek.capitalize()}', marker=dict(color='red', size=7)))
    fig.add_trace(go.Scatter(x=[spot], y=[current_put_greek_value], mode='markers', name=f'Current Put {greek.capitalize()}', marker=dict(color='green', size=7)))

    fig.update_layout(title=f'{greek.capitalize()} vs Spot Price',
                      xaxis_title='Spot Price',
                      yaxis_title=greek.capitalize())

    return fig

st.subheader("Visualisation of Greeks")

greeks = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
for i in range(0, len(greeks), 2):
    
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(greek_visualisation(S, K, r, T, vol, greeks[i]), use_container_width=True)


    if i + 1 < len(greeks):
        with col2:
            st.plotly_chart(greek_visualisation(S, K, r, T, vol, greeks[i + 1]), use_container_width=True)

# Price shock analysis
# Slider controls
st.header("Price Heatmap")
st.subheader("Shock Parameters")
col1, col2 = st.columns(2)
spot_min = col1.slider("Minimum Spot Price", value=S*0.95, min_value=S*0.25, max_value=S*1.5)
spot_max = col2.slider("Maximum Spot Price", value=S*1.05, min_value=S*0.25, max_value=S*1.5)
vol_min = col1.slider("Minimum Volatility", value=vol*0.80, min_value=vol*0.25, max_value=vol*1.5)
vol_max = col2.slider("Maximum Volatility", value=vol*1.2, min_value=vol*0.25, max_value=vol*1.5)

if spot_min > spot_max:
    st.error("Error: Min Spot > Max Spot")

if vol_min > vol_max:
    st.error("Error: Min Vol > Max Vol")

# Generate grid of spot prices and volatilities
spot_prices = np.linspace(spot_min, spot_max, 10)
volatilities = np.linspace(vol_min, vol_max, 10)

# Calculate option prices for each combination of spot and volatility
call_prices = np.array([[ml.BS([S_i, K, r * 100, T * 365], sigma * 100).callPrice for sigma in volatilities] for S_i in spot_prices])
put_prices = np.array([[ml.BS([S_i, K, r * 100, T * 365], sigma * 100).putPrice for sigma in volatilities] for S_i in spot_prices])

def plot_heatmap(data, title):
    data = data[::-1]
    plt.figure(figsize=(16, 12))
    sb.heatmap(data,
               yticklabels=np.round(spot_prices[::-1], 2),  
               xticklabels=np.round(volatilities, 2),
               cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title(title, fontsize=16)
    plt.ylabel("Spot Price", fontsize=14)
    plt.xlabel("Volatility", fontsize=14)
    st.pyplot(plt)

# Display heatmaps
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option Price Heatmap")
    plot_heatmap(call_prices, "Call Option Prices")

with col2:
    st.subheader("Put Option Price Heatmap")
    plot_heatmap(put_prices, "Put Option Prices")