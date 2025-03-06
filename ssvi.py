import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objs as go
from DataFunctions import *
from DataImpLib import *

class SSVIModel:
    def __init__(self, f, t):
        self.f = f  # Forward price
        self.t = t  # Time to expiration

    def ssvi_model(self, k, theta, rho, phi):
        return (theta / 2) * (1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + (1 - rho**2)))

    def phi_function(self, theta, a, b):
        return a*theta **(-b)  # tested multiple versions of phi function
        #return a / (theta**(b)*((1+theta)**(1-b)))
        #return (1/(a*theta) * (1-(1-np.exp(-a * theta))/(a*theta)) )

    def calibrate(self, K, market_vols):
        def objective(params):
            a, b, rho = params
            theta = np.mean(market_vols**2 * self.t)
            phi = self.phi_function(theta, a, b)
            k = np.log(K / self.f)
            model_vols = np.sqrt(self.ssvi_model(k, theta, rho, phi) / self.t)
            return np.sum((model_vols - market_vols)**2)

        # Initial guess and bounds
        x0 = [0.1, 0.2, -0.5]  # Initial guess for [a, b, rho]
        #bounds = [(0.000001, 1), (0.0000001, 1/2), (-0.9999, 0.9999)] # for 2nd phi function
        bounds = [(0.000001, 999), (0.0000001, 1), (-0.9999, 0.9999)]         

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        return result.x

def SSVICalculator(calldfFiltred, ticker):
    unique_T = calldfFiltred['timeToMaturity'].unique()
    calldfFiltred['ssvi_vol'] = np.nan

    d = dividend_yield(ticker)
    r = ameribor()
    S0 = LatestClose(ticker)

    for T in unique_T:
        df_T = calldfFiltred[calldfFiltred['timeToMaturity'] == T].copy()
        f = S0 * np.exp((r-d)*T)
        ssvi = SSVIModel(f, T)
        a, b, rho = ssvi.calibrate(df_T['strike'].values, df_T['impliedVolatility'].values)
        
        theta = np.mean(df_T['impliedVolatility'].values**2 * T)
        phi = ssvi.phi_function(theta, a, b)
        k = np.log(df_T['strike'].values / f)
        df_T.loc[:, 'ssvi_vol'] = np.sqrt(ssvi.ssvi_model(k, theta, rho, phi) / T)

        # Reset the index to ensure we can access rows by integer location
        df_T = df_T.reset_index(drop=True)
        
        if len(df_T) > 1:
            df_T.loc[0, 'ssvi_vol'] = df_T.loc[1, 'ssvi_vol']
        else:
            df_T.loc[0, 'ssvi_vol'] = 0  # or some other appropriate value

        ogdf_indexed = calldfFiltred.set_index('contractSymbol')
        df_indexed = df_T.set_index('contractSymbol')

        ogdf_indexed.update(df_indexed[['ssvi_vol']])

        # Reset the index to make 'contractSymbol' a column again
        calldfFiltred = ogdf_indexed.reset_index()
    return calldfFiltred

# def CreateSSVISurface(calldfFiltred, ticker):
#     d = dividend_yield(ticker)
#     r = ameribor()
#     S0 = LatestClose(ticker)

#     unique_T = np.sort(calldfFiltred['timeToMaturity'].unique())
#     unique_K = np.sort(calldfFiltred['strike'].unique())

#     X, Y = np.meshgrid(unique_K, unique_T)
#     Z = np.zeros_like(X)

#     for i, T in enumerate(unique_T):
#         df_T = calldfFiltred[calldfFiltred['timeToMaturity'] == T]
#         f = S0 * np.exp((r-d)*T)
#         ssvi = SSVIModel(f, T)
#         a, b, rho = ssvi.calibrate(df_T['strike'].values, df_T['impliedVolatility'].values)
        
#         theta = np.mean(df_T['impliedVolatility'].values**2 * T)
#         phi = ssvi.phi_function(theta, a, b)
#         k = np.log(unique_K / f)
#         Z[i, :] = np.sqrt(ssvi.ssvi_model(k, theta, rho, phi) / T)

#     surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')
#     layout = go.Layout(
#         title=f'SSVI Volatility Surface for {ticker}',
#         scene=dict(
#             xaxis_title='Strike',
#             yaxis_title='Time to Maturity',
#             zaxis_title='Implied Volatility'
#         )
#     )
#     fig = go.Figure(data=[surface], layout=layout)
#     return fig

#Usage example:
# ticker = "TSLA"
# calldf = CallDF(ticker)
# calldfFiltred = Krangecreator(calldf, ticker, 70, 130)
# calldfFiltred = Trangecreator(calldfFiltred, 180, 360)
# calldfFiltred = LiquidityFilter(calldfFiltred, 0)
# calldfFiltred = SSVICalculator(calldfFiltred, ticker)
# calldfFiltred['diff'] = (calldfFiltred['ssvi_vol'] - calldfFiltred['impliedVolatility'])/calldfFiltred['impliedVolatility']*100

# graph = CreateGraphofindicator('impliedVolatility', calldfFiltred)
# graph.show()
# graph = CreateGraphofindicator('ssvi_vol', calldfFiltred)
# graph.show()
# graph = CreateGraphofindicator('diff', calldfFiltred)
# graph.show()

# surface = CreateSSVISurface(calldfFiltred, ticker)
# surface.show()