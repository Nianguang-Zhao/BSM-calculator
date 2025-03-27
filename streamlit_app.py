from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import traceback

app = Flask(__name__)


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
    """
    Generate a comprehensive price chart for options.
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
    
    Returns:
        str: Base64 encoded PNG image of the option price chart
    """
    # Generate a range of underlying prices
    ST_range = np.linspace(max(0.1, S * 0.5), S * 1.5, 100)
    
    # Calculate option prices for each stock price
    option_prices = [black_scholes(st, K, T, r, sigma, option_type)['price'] for st in ST_range]
    
    # Create the plot with improved styling
    plt.figure(figsize=(12, 7))
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
    """
    Generate a comprehensive profit/loss chart for option strategies.
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        option_price (float): Option premium
        option_type (str): 'call' or 'put'
        side (str): 'buy' or 'sell'
        premium_paid (float): Premium paid/received
    
    Returns:
        str: Base64 encoded PNG image of the profit/loss chart
    """
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
        print(f"Error in profit/loss chart generation: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        # Validate inputs
        S = float(request.form["S"])
        K = float(request.form["K"])
        T = float(request.form["T"])
        r = float(request.form["r"]) / 100  # Convert percentage to decimal
        sigma = float(request.form["sigma"]) / 100  # Convert percentage to decimal
        option_type = request.form.get("option_type", "call")
        side = request.form.get("side", "buy")
        
        # Ensure matplotlib operations are thread-safe
        plt.close('all')
        
        # Calculate option price and Greeks
        result = black_scholes(S, K, T, r, sigma, option_type)
        
        # Use the calculated option price as the premium
        premium_paid = result['price']
        
        # Calculate profit and loss
        profit_loss = calculate_profit_loss(S, K, result['price'], option_type, side, premium_paid)
        
        # Generate price chart
        chart = generate_price_chart(S, K, T, r, sigma, option_type)
        
        # Generate profit/loss chart
        pl_chart = generate_profit_loss_chart(S, K, result['price'], option_type, side, premium_paid)
        
        # Custom JSON response to handle Infinity
        response = {
            "result": result,
            "profit_loss": {
                "max_profit": "Unlimited" if profit_loss['max_profit'] == float('inf') else profit_loss['max_profit'],
                "max_loss": "Unlimited" if profit_loss['max_loss'] == float('inf') else profit_loss['max_loss'],
                # "pop": profit_loss['pop']
            },
            "chart": chart,
            "pl_chart": pl_chart
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Detailed error: {traceback.format_exc()}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400
    finally:
        plt.close('all')  # Ensure plots are closed after each request
app.config['TEMPLATES_AUTO_RELOAD'] = True

# if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=5000)
