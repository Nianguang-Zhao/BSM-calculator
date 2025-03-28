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
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
}
.stApp {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stTitle {
    text-align: center;
    color: #333;
    margin-bottom: 20px;
}
.strategy-analysis {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.strategy-box {
    background-color: #f1f1f1;
    padding: 15px;
    text-align: center;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='stTitle'>Advanced Option Strategy Calculator</h1>", unsafe_allow_html=True)
    
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
    
    if st.button("Calculate Option Strategy", help="Click to calculate option pricing metrics"):
        # Convert inputs
        T = time_to_maturity / 365  # Convert days to years
        r = risk_free_rate / 100
        sigma = volatility / 100
        
        # Calculate option pricing
        option_price, delta, gamma, theta, vega, rho = black_scholes_merton(
            current_stock_price, strike_price, T, r, sigma, 
            option_type='call' if option_type == "Call Option" else 'put'
        )
        
        # Option Strategy Analysis
        st.markdown("<div class='strategy-analysis'>", unsafe_allow_html=True)
        strategy_analysis = [
            ("Max Profit", "Unlimited"),
            ("Max Loss", f"${option_price:.4f}"),
            ("Probability of Profit", "N/A")
        ]
        
        for label, value in strategy_analysis:
            st.markdown(f"""
            <div class='strategy-box'>
                <strong>{label}</strong><br>
                {value}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Profit/Loss at Expiration Chart
        fig = create_profit_loss_chart(current_stock_price, strike_price, option_price)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pricing Results
        st.markdown("<div class='strategy-analysis'>", unsafe_allow_html=True)
        result_metrics = [
            ("Option Price", f"${option_price:.4f}"),
            ("Delta", f"{delta:.4f}"),
            ("Gamma", f"{gamma:.4f}"),
            ("Theta", f"{theta:.4f}"),
            ("Vega", f"{vega:.4f}"),
            ("Rho", f"{rho:.4f}")
        ]
        
        for label, value in result_metrics:
            st.markdown(f"""
            <div class='strategy-box'>
                <strong>{label}</strong><br>
                {value}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
