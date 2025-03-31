import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate the Black-Scholes option price for both call and put options."""
    T = T / 365  # Convert days to years
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Corrected Greeks calculations
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Corrected Theta calculation (annual terms)
    if option_type.lower() == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))/365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))/365
    
    # Corrected Vega calculation
    vega = S * norm.pdf(d1) * np.sqrt(T)/100
    
    # Corrected Rho calculation
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)/100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)/100
    
    return {
        'price': round(option_price, 4),
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4)
    }

def calculate_profit_loss(S, K, option_price, option_type, side, premium_paid):
    """Calculate maximum profit and loss for the option strategy."""
    # Ensure premium_paid is a float and handle zero case
    premium_paid = float(premium_paid) if premium_paid is not None else 0
    
    if side.lower() == 'buy':
        if option_type.lower() == 'call':
            # Buyer of a call option
            max_profit = 'Unlimited'  # Unlimited upside
            max_loss = premium_paid
        else:
            # Buyer of a put option
            max_profit = K - premium_paid
            max_loss = premium_paid
    else:  # sell
        if option_type.lower() == 'call':
            # Seller of a call option
            max_profit = premium_paid
            max_loss = 'Unlimited'  # Unlimited downside
        else:
            # Seller of a put option
            max_profit = premium_paid
            max_loss = K - premium_paid
    
    return {
        'max_profit': max_profit,
        'max_loss': max_loss,
    }

def generate_price_chart(S, K, T, r, sigma, option_type):
    """Generate a comprehensive price chart for options."""
    # Generate a range of underlying prices
    ST_range = np.linspace(max(0.1, S * 0.5), S * 1.5, 100)
    
    # Calculate option prices for each stock price
    option_prices = [black_scholes(st, K, T, r, sigma, option_type)['price'] for st in ST_range]
    
    # Create the plot with improved styling
    plt.figure(figsize=(10, 6))
    plt.plot(ST_range, option_prices, color='#007bff', linewidth=2, label='Option Price')
    
    # Add vertical lines for key price points
    plt.axvline(x=S, color='green', linestyle='--', label='Current Stock Price')
    plt.axvline(x=K, color='red', linestyle='--', label='Strike Price')
    
    # Title and labels with more context
    plt.title(f'{option_type.capitalize()} Option Pricing Sensitivity', fontsize=15)
    plt.xlabel('Underlying Stock Price', fontsize=12)
    plt.ylabel('Option Price', fontsize=12)
    
    # Enhanced grid and aesthetics
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    # Annotate current parameters
    plt.text(0.05, 0.95, 
             f'S: {S:.2f}, K: {K:.2f}, T: {T} days\n'
             f'r: {r*100:.2f}%, σ: {sigma*100:.2f}%', 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot to a base64 encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_profit_loss_chart(S, K, option_price, option_type, side, premium_paid):
    """Generate a comprehensive profit/loss chart for option strategies."""
    try:
        # Generate a range of underlying prices with more granularity
        ST_range = np.linspace(max(0.1, S * 0.5), S * 1.5, 100)
        
        # Calculate profit/loss for each price
        pl_at_expiry = []
        for st in ST_range:
            if side.lower() == 'buy':
                if option_type.lower() == 'call':
                    # Buyer of a call option
                    pl = max(st - K, 0) - premium_paid
                else:
                    # Buyer of a put option
                    pl = max(K - st, 0) - premium_paid
            else:  # sell
                if option_type.lower() == 'call':
                    # Seller of a call option
                    pl = premium_paid - max(st - K, 0)
                else:
                    # Seller of a put option
                    pl = premium_paid - max(K - st, 0)
            pl_at_expiry.append(pl)
        
        # Create the plot with improved styling
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Color palette for better visibility
        colors = {
            'call_buy': '#2196F3',    # Bright Blue
            'call_sell': '#FF9800',   # Orange
            'put_buy': '#4CAF50',     # Green
            'put_sell': '#F44336'     # Red
        }
        
        color_key = f"{option_type.lower()}_{side.lower()}"
        line_color = colors.get(color_key, 'purple')
        
        # Plot the profit/loss line with chosen color
        plt.plot(ST_range, pl_at_expiry, color=line_color, linewidth=2.5, label='Strategy P/L')
        
        # Horizontal and vertical reference lines
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(x=S, color='black', linestyle=':', linewidth=1)
        plt.axvline(x=K, color='black', linestyle='-', linewidth=2)
        
        # Calculate breakeven point
        if option_type.lower() == 'call':
            breakeven = K + premium_paid
        else:
            breakeven = K - premium_paid
        
        # Rich annotations
        plt.annotate(f'Current Price: {S:.2f}', 
                     xy=(S, 0), 
                     xytext=(10, 20), 
                     textcoords='offset points',
                     fontsize=9,
                     color='darkgreen')
        
        plt.annotate(f'Strike Price: {K:.2f}', 
                     xy=(K, 0), 
                     xytext=(10, -20), 
                     textcoords='offset points',
                     fontsize=9,
                     color='darkred')
        
        plt.annotate(f'Breakeven: {breakeven:.2f}', 
                     xy=(breakeven, 0), 
                     xytext=(10, 10), 
                     textcoords='offset points',
                     fontsize=9,
                     color='purple')
        
        plt.title(f'{side.capitalize()} {option_type.capitalize()} Option P/L at Expiration', fontsize=12)
        plt.xlabel('Underlying Stock Price at Expiration', fontsize=10)
        plt.ylabel('Profit or Loss (USD)', fontsize=10)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Dynamic axis scaling with more robust handling
        y_max = max(pl_at_expiry)
        y_min = min(pl_at_expiry)
        plt.ylim(min(y_min * 1.2, y_min - 50), max(y_max * 1.2, y_max + 50))
        plt.xlim(min(ST_range), max(ST_range))
        
        # Save plot to a base64 encoded image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    except Exception as e:
        st.error(f"Error in profit/loss chart generation: {e}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="Options Pricing Calculator")
    
    # Custom CSS to reduce input width and adjust font sizes
    st.markdown("""
    <style>
    /* Reduce width of number input containers */
    .stNumberInput > div > div {
        width: 150px !important;
        min-width: 150px !important;
    }
    
    /* Reduce font size for metrics */
    .metric-container {
        font-size: 0.8rem !important;
    }
    
    /* Adjust chart captions */
    .caption {
        font-size: 0.7rem !important;
    }
    
    /* Reduce select box width */
    .stSelectbox > div > div {
        width: 150px !important;
        min-width: 150px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("BSM Options Pricing and Strategy Analyzer")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])  # Adjust column proportions
    
    with col1:
        st.header("Input Parameters")
        
        # Stock price input
        S = st.number_input("Current Stock Price ($)", min_value=0.01, value=110.0, step=0.1, format="%.2f")
        
        # Strike price input
        K = st.number_input("Strike Price ($)", min_value=0.01, value=100.0, step=0.1, format="%.2f")
        
        # Time to expiration input
        T = st.number_input("Time to Expiration (Days)", min_value=1, value=30, step=1)
        
        # Risk-free rate input
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=4.0, step=0.1, format="%.2f")
        
        # Volatility input
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=0.1, format="%.2f")
        
        # Option type selection
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        # Side selection
        side = st.selectbox("Strategy Side", ["Buy", "Sell"])
        
        # Calculate button
        calculate_button = st.button("Calculate Option Metrics")
    
    with col2:
        st.header("Option Analysis Results")
        
        if calculate_button:
            try:
                # Perform calculations
                result = black_scholes(S, K, T, r/100, sigma/100, option_type.lower())
                
                # Create columns for Greeks with reduced width
                greek_cols = st.columns(6)
                greek_names = ['Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
                greek_values = [
                    result['price'], 
                    result['delta'], 
                    result['gamma'], 
                    result['theta'], 
                    result['vega'], 
                    result['rho']
                ]
                
                # Display Greeks with custom formatting
                for col, name, value in zip(greek_cols, greek_names, greek_values):
                    col.markdown(f"""
                    <div style='text-align:center;'>
                    <small style='color:gray;'>{name}</small><br>
                    <span style='font-size:1rem;'>{value:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Calculate profit/loss
                profit_loss = calculate_profit_loss(S, K, result['price'], option_type.lower(), side.lower(), result['price'])
                
                # Display Max Profit and Max Loss
                st.subheader("Strategy Analysis")
                col1, col2 = st.columns(2)
                col1.markdown("""
                <div style='text-align:center;'>
                <small style='color:gray;'>Max Profit</small><br>
                <span style='font-size:1rem;'>{}</span>
                </div>
                """.format(str(profit_loss['max_profit'])), unsafe_allow_html=True)
                
                col2.markdown("""
                <div style='text-align:center;'>
                <small style='color:gray;'>Max Loss</small><br>
                <span style='font-size:1rem;'>{}</span>
                </div>
                """.format(str(profit_loss['max_loss'])), unsafe_allow_html=True)
                
                # Generate and display charts
                st.subheader("Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(f"data:image/png;base64,{generate_price_chart(S, K, T, r/100, sigma/100, option_type.lower())}")
                    st.markdown("<div class='caption' style='text-align:center;'>Option Price Sensitivity</div>", unsafe_allow_html=True)
                
                with col2:
                    st.image(f"data:image/png;base64,{generate_profit_loss_chart(S, K, result['price'], option_type.lower(), side.lower(), result['price'])}")
                    st.markdown("<div class='caption' style='text-align:center;'>Profit/Loss at Expiration</div>", unsafe_allow_html=True)
                
            st.sidebar.markdown("""
<div style="text-align: center; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/nianguang-zhao/" target="_blank" style="text-decoration: none; color: #0077b5; display: inline-flex; align-items: center;">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="16" height="16" style="margin-right: 5px;"> 
        Nianguang Zhao
    </a>
</div>
""", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
