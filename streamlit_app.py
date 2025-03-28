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
    fig = plt.Figure(data=[plt.Scatter(x=price_range, y=pl_values, mode='lines', line=dict(color='blue', width=3))])
    
    fig.update_layout(
        title='Option Profit/Loss at Expiration',
        xaxis_title='Stock Price at Expiration',
        yaxis_title='Profit/Loss (USD)',
        template='plotly_white',
        height=400,
        width=800,
        font=dict(size=12),
        title_font_size=16,
        title_x=0.5
    )
    
    # Add vertical lines for current price and breakeven
    fig.add_vline(x=current_price, line_dash='dash', line_color='green', 
                  annotation_text='Current Price', annotation_position='top right')
    breakeven = strike_price + option_price
    fig.add_vline(x=breakeven, line_dash='dash', line_color='red', 
                  annotation_text='Breakeven', annotation_position='top right')
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_color='black', line_width=1)
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Option Strategy Calculator", 
        page_icon=":chart_with_upwards_trend:", 
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stNumberInput > div > div > input {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    .stSelectbox > div > div > div {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .metric-box {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 18%;
    }
    .metric-label {
        font-size: 0.8em;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Advanced Option Strategy Calculator</h1>", unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Option Parameters")
        option_type = st.selectbox("Option Type", ["Call Option", "Put Option"])
        side = st.selectbox("Side", ["Buy", "Sell"])
        current_stock_price = st.number_input("Current Stock Price (S)", value=44.0, step=0.1, format="%.2f")
        strike_price = st.number_input("Strike Price (K)", value=33.0, step=0.1, format="%.2f")
    
    with col2:
        st.markdown("### Market Conditions")
        time_to_maturity = st.number_input("Time to Maturity (Days)", value=55)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.0, step=0.1, format="%.2f")
        volatility = st.number_input("Volatility (%)", value=33.0, step=0.1, format="%.2f")
    
    # Calculate button
    calculate_button = st.button("Calculate Option Strategy", use_container_width=True)
    
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
        
        # Metrics display
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        metrics = [
            ("Option Price", f"${option_price:.4f}"),
            ("Delta", f"{delta:.4f}"),
            ("Gamma", f"{gamma:.4f}"),
            ("Theta", f"{theta:.4f}"),
            ("Vega", f"{vega:.4f}"),
            ("Rho", f"{rho:.4f}")
        ]
        
        for label, value in metrics:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Profit/Loss Chart
        st.markdown("### Profit/Loss Analysis")
        fig = create_profit_loss_chart(current_stock_price, strike_price, option_price)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy Summary
        st.markdown("### Strategy Summary")
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.metric("Max Profit", "Unlimited")
        
        with col_summary2:
            st.metric("Max Loss", f"${option_price:.4f}")

def __runpy__():
    """
    Special function to run the Streamlit app
    This is optional and can be removed when running directly
    """
    main()

if __name__ == "__main__":
    main()
