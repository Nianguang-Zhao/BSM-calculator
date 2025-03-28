import streamlit as st
import numpy as np
import plotly.graph_objs as plt
from scipy.stats import norm

def black_scholes_merton(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    
    return option_price, delta, gamma, theta, vega, rho

def create_profit_loss_chart(current_price, strike_price, option_price):
    # Generate price range for the chart
    price_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
    
    # Calculate Profit/Loss at Expiration
    pl_values = np.maximum(price_range - strike_price, 0) - option_price
    
    # Create Plotly figure
    fig = plt.Figure(data=[plt.Scatter(x=price_range, y=pl_values, mode='lines', line=dict(color='blue'))])
    
    fig.update_layout(
        title='Buy Call Option P/L at Expiration',
        xaxis_title='Stock Price at Expiration',
        yaxis_title='Profit/Loss (USD)',
        template='plotly_white',
        height=400
    )
    
    # Add vertical lines for current price and breakeven
    fig.add_vline(x=current_price, line_dash='dash', line_color='green', annotation_text='Current Price: {:.2f}'.format(current_price))
    breakeven = strike_price + option_price
    fig.add_vline(x=breakeven, line_dash='dash', line_color='red', annotation_text='Breakeven: {:.2f}'.format(breakeven))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_color='black', line_width=1)
    
    return fig

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f4f4f4;
}
.main-container {
    display: flex;
    gap: 20px;
}
.input-section {
    flex: 1;
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.results-section {
    flex: 1;
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 30px;
    font-weight: 600;
}
.stSelectbox, .stNumberInput {
    margin-bottom: 15px;
}
.pricing-metrics, .strategy-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}
.metric-box {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #e0e0e0;
}
.metric-label {
    color: #555;
    font-size: 0.9em;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.1em;
    font-weight: bold;
    color: #2c3e50;
}
.stButton>button {
    width: 100%;
    background-color: #3498db;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1>Advanced Option Strategy Calculator</h1>", unsafe_allow_html=True)
    
    # Main container with two columns
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # Input Section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("<h2>Input Parameters</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        option_type = st.selectbox("Option Type", ["Call Option", "Put Option"])
        side = st.selectbox("Side", ["Buy", "Sell"])
    
    with col2:
        current_stock_price = st.number_input("Current Stock Price (S)", value=44.0, step=0.1)
        strike_price = st.number_input("Strike Price (K)", value=33.0, step=0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        time_to_maturity = st.number_input("Time to Maturity (Days)", value=55)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.0, step=0.1)
    
    with col4:
        volatility = st.number_input("Volatility (%)", value=33.0, step=0.1)
    
    calculate_button = st.button("Calculate Option Strategy", help="Click to calculate option pricing metrics")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results Section
    st.markdown("<div class='results-section'>", unsafe_allow_html=True)
    
    if calculate_button:
        # Convert inputs
        T = time_to_maturity / 365  # Convert days to years
        r = risk_free_rate / 100
        sigma = volatility / 100
        
        # Calculate option pricing
        option_price, delta, gamma, theta, vega, rho = black_scholes_merton(
            current_stock_price, strike_price, T, r, sigma, 
            option_type='call' if option_type == "Call Option" else 'put'
        )
        
        # Option Pricing Metrics
        st.markdown("<h2>Pricing Metrics</h2>", unsafe_allow_html=True)
        st.markdown("<div class='pricing-metrics'>", unsafe_allow_html=True)
        pricing_metrics = [
            ("Option Price", f"${option_price:.4f}"),
            ("Delta", f"{delta:.4f}"),
            ("Gamma", f"{gamma:.4f}"),
            ("Theta", f"{theta:.4f}"),
            ("Vega", f"{vega:.4f}"),
            ("Rho", f"{rho:.4f}")
        ]
        
        for label, value in pricing_metrics:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Option Strategy Analysis
        st.markdown("<h2>Strategy Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<div class='strategy-metrics'>", unsafe_allow_html=True)
        strategy_analysis = [
            ("Max Profit", "Unlimited"),
            ("Max Loss", f"${option_price:.4f}")
        ]
        
        for label, value in strategy_analysis:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Profit/Loss at Expiration Chart
        st.markdown("<h2>Profit/Loss Chart</h2>", unsafe_allow_html=True)
        fig = create_profit_loss_chart(current_stock_price, strike_price, option_price)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Close main container
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
