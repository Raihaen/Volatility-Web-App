import scipy.interpolate
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.optimize import brentq


def ameribor():
    sofr = yf.Ticker("^AMERIBOR")
    sofr_hist = sofr.history(period="1y").tail()['Close']/100
    return float(sofr_hist.iloc[0])

def CallDF(ticker):
    
    stock = Stockinfo(ticker) #we get the stockinfo
    expirations = stock.options #we get available expiration dates
    calls_map = []
    if not expirations :
        return pd.DataFrame()
    for expiry in expirations : 
        options_chain = stock.option_chain(expiry)
        calls = options_chain.calls #we use here call price for implied vol calculation, see read me file
        calls['expiry'] = expiry #to sort call series through expiry date and save them in one big table
        calls_map.append(calls)

    calls_df = pd.concat(calls_map, ignore_index=True) #we create our database using pandas
    calls_df['timeToMaturity'] = (pd.to_datetime(calls_df['expiry']) - pd.to_datetime("today")).dt.days / 365 #we add a column for time to maturity
    return calls_df

def LatestClose(ticker):
    stock = Stockinfo(ticker)
    history = stock.history(period="1d")
    latest_close = history['Close'].iloc[-1]
    #latest_close = yf.Ticker(ticker).info['previousClose']  #we use lastest close data instead of real time data because the yfinance option data we have is at close only
    return latest_close

def Stockinfo(ticker):
    #we choose a ticker
    stock = yf.Ticker(ticker)   #we get stock info through yahoo finance
    return stock

def getindicators(calls_df):
    sigma = calls_df['impliedVolatility']
    K = calls_df['strike']
    T = calls_df['timeToMaturity']
    return (sigma,K,T)

def dividend_yield(ticker) :
    dividend_yield = yf.Ticker(ticker).info.get("dividendYield")
    if type(dividend_yield) != float :
        dividend_yield = 0
    return dividend_yield/100 # to turn it into a float instead of a percentage


def CreateGraph(sigma, K, T, gauss=False):
    K_grid, T_grid = np.meshgrid(np.unique(K), np.unique(T))  # Grid for plotting
    sigma_grid = scipy.interpolate.griddata((K, T), sigma, (K_grid, T_grid), method='linear')  # Interpolated surface

    # Apply Gaussian smoothing to sigma_grid (volatility surface)
    if gauss == True :
        sigma_grid = gaussian_filter(sigma_grid, sigma=1) 
    # Create a Plotly figure
    fig = go.Figure()

    # Add a 3D surface plot
    fig.add_trace(go.Surface(z=sigma_grid, x=K_grid, y=T_grid, colorscale='viridis'))

    # Update layout for better visualization
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Time to Maturity (T)",
            zaxis_title="Implied Volatility (σ)",
        )
    )
    return fig


def Krangecreator(df, ticker, lowerbound=0, upperbound=2000): #we add lower and higher bound default values 
    S = LatestClose(ticker)
    lowerbound = S * lowerbound/100
    upperbound = S * upperbound/100
    filtered_df = df[(df['strike'] >= lowerbound) & (df['strike'] <= upperbound)]
    return filtered_df

def Trangecreator(df, lowerbound=0, upperbound=10000): #we add lower and higher bound default values 
    lowerbound = lowerbound/365
    upperbound = upperbound/365
    filtered_df = df[(df['timeToMaturity'] >= lowerbound) & (df['timeToMaturity'] <= upperbound)]
    return filtered_df

def LiquidityFilter(df, bound=100):
    filtered_df = df[(df['volume'] >= bound)]
    return filtered_df




### Ameribor vs SPIBOR : 

def calculate_SPIBOR(expiration_date):
    # Get the SPX options data
    ticker = yf.Ticker("^SPX")
    options_data = CallDF("^SPX")
    
    # Find the spot price (current price of the underlying SPX index)
    S0 = ticker.history(period='1d')['Close'].iloc[-1]
    
    # Find the ATM strike price (closest to the spot price)
    strikes = options_data.calls['strike']
    K = min(strikes, key=lambda x: abs(x - S0))  # Find the ATM strike
    
    # Get the corresponding ATM call and put option prices
    C = options_data.calls[options_data.calls['strike'] == K]['lastPrice'].iloc[0]
    P = options_data.puts[options_data.puts['strike'] == K]['lastPrice'].iloc[0]
    
    # Time to expiration in years (T)
    expiration_date = pd.to_datetime(expiration_date)
    current_date = pd.to_datetime('today')
    T = (expiration_date - current_date).days / 365.0
    
    # Calculate the implied risk-free interest rate (r) using the put-call parity formula
    implied_rate = np.log(K/(S0 + P - C))/T

    return implied_rate


# Example usage
# expiration_date = "2025-03-31"  # Choose an expiration date (format: YYYY-MM-DD)

# implied_rate = calculate_SPIBOR(expiration_date)
# print(f"Implied risk-free interest rate: {implied_rate*100:.2f}%")
# print((ameribor() - implied_rate)/implied_rate*100)





