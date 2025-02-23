import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from DataFunctions import *
from DataImpLib import * 

def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
    
    try:
        return brentq(objective, 1e-6, 5)  # Assuming volatility is between 0.0001% and 500%
    except ValueError:
        return np.nan  # Return NaN if no solution is found




def ManualIV(calldfFiltred,ticker):
    r = ameribor()
    d = dividend_yield(ticker)
    S0 = LatestClose(ticker)
    for index, row in calldfFiltred.iterrows():
        price = (row['bid'] + row['ask']) / 2
        S0 = LatestClose(ticker)
        implied_vol = implied_volatility(price, S0, row['strike'], row['timeToMaturity'], r-d, 'call')
        calldfFiltred.at[index, 'implied_vol'] = implied_vol
    return calldfFiltred




###### EXAMPLE CODE : #######

# ticker = "AAPL" 


# calldf = CallDF(ticker)
# calldfFiltred = Krangecreator(calldf, ticker, 0, 200)
# calldfFiltred = Trangecreator(calldfFiltred, 0, 200)
# calldfFiltred = LiquidityFilter(calldfFiltred, 0)

# calldfFiltred = ManualIV(calldfFiltred, ticker)
# print(calldfFiltred.head(50))

# graph = CreateGraphofindicator('implied_vol',calldfFiltred)
# graph.show()
# graph = CreateGraphofindicator('impliedVolatility',calldfFiltred)
# graph.show()