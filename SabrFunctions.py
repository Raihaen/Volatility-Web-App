from DataFunctions import *
from DataImpLib import *
import streamlit as st
import plotly.io as pio
import plotly.tools as tls
from pathlib import Path
from scipy.optimize import minimize


class SABRModel:
    def __init__(self, f, t, beta=1.0):
        self.f = f  # Forward price
        self.t = t  # Time to expiration
        self.beta = beta

    def HaganFormulaWithBeta(self, K, alpha, rho, nu) :
        # Implement Hagan's approximation formula (to implement beta != 1)
        F = self.f
        T = self.t
        B = self.beta
        log_FK = np.log(F/K)
        A = (F*K)**((1-B)/2)
        z = nu/alpha * A * log_FK
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        return alpha * log_FK / x * (1 + 
            ((1-B)**2/24 * log_FK**2 + 1/4*rho*B*nu*alpha*log_FK + (2-3*rho**2)/24*nu**2) * T)  

    def HaganFormula(self, K, alpha, rho, nu) :
        # Implement Hagan's formula for beta = 1 (lots of factors are removed)
        F = self.f
        T = self.t
        B = self.beta
        log_FK = np.log(F/K)
        A = (F*K)**((1-B)/2)
        z = nu/alpha * A * log_FK
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        return alpha * z / x * (1 + 
            ( 1/4*rho*B*nu*alpha + (2-3*rho**2)/24*nu**2) * T)  
      
    
    def calibrate(self, K, market_vols):
        def objective(params):
            alpha, rho, nu = params
            model_vols = self.HaganFormula(K, alpha, rho, nu)
            return np.sum((model_vols - market_vols)**2)

        # Initial guess and bounds
        x0 = [0.2, 0, 0.8]  # Initial guess for [alpha, rho, nu]
        bounds = [(0.0001, 1), (-0.9999, 0.9999), (0.0001, 1)]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        #result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds)
        #result = minimize(objective, x0, method='trust-constr', bounds=bounds)

        
        return result.x


# ticker = "AAPL" 
# S0 = LatestClose(ticker)
# dividend_yield = yf.Ticker(ticker).info.get("dividendYield")
# if type(dividend_yield) != float :
#     dividend_yield = 0


# calldf = CallDF(ticker)
# calldfFiltred = Krangecreator(calldf, ticker, 80, 120)
# calldfFiltred = Trangecreator(calldfFiltred, 0, 200)
# calldfFiltred = LiquidityFilter(calldfFiltred, 0)
# print(calldfFiltred.head(50))

def SABRCalculator(calldfFiltred,ticker) :
    unique_T = calldfFiltred['timeToMaturity'].unique()
    calldfFiltred['sabr_vol'] = np.nan

    d = dividend_yield(ticker)
    r = ameribor()
    S0 = LatestClose(ticker)
    for T in unique_T:
        df_T = calldfFiltred[calldfFiltred['timeToMaturity'] == T].copy()
        f = S0*np.exp((r-d)*T)
        sabr = SABRModel(f, T, beta=1.0)
        alpha, rho, nu = sabr.calibrate(df_T['strike'].values, df_T['impliedVolatility'].values)
        
            
        df_T.loc[:, 'sabr_vol'] = sabr.HaganFormula(df_T['strike'].values, alpha, rho, nu)
        # Reset the index to ensure we can access rows by integer location
        df_T = df_T.reset_index(drop=True)
        
        if len(df_T) > 1:
            df_T.loc[0, 'sabr_vol'] = df_T.loc[1, 'sabr_vol']
        else:
            df_T.loc[0, 'sabr_vol'] = 0  # or some other appropriate value

        ogdf_indexed = calldfFiltred.set_index('contractSymbol')
        df_indexed = df_T.set_index('contractSymbol')

        ogdf_indexed.update(df_indexed[['sabr_vol']])

        # Reset the index to make 'contractSymbol' a column again
        calldfFiltred = ogdf_indexed.reset_index()
    return calldfFiltred
# calldfFiltred = SABRCalculator(calldfFiltred)
# calldfFiltred['diff'] = (calldfFiltred['sabr_vol'] - calldfFiltred['impliedVolatility'])/calldfFiltred['impliedVolatility']*100
# print(calldfFiltred.head(50))

# graph = CreateGraphofindicator('impliedVolatility',calldfFiltred)
# graph.show()
# graph = CreateGraphofindicator('sabr_vol',calldfFiltred)
# graph.show()
# graph = CreateGraphofindicator('diff',calldfFiltred)
# graph.show()