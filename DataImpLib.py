import numpy as np
import QuantLib as ql
import scipy.interpolate
from DataFunctions import *
from scipy.interpolate import RectBivariateSpline
import mpmath as mp
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

def getindicatorsvolume(calls_df):
    Volume = calls_df['volume']
    K = calls_df['strike']
    T = calls_df['timeToMaturity']
    return (Volume,K,T)

def compute_volatility_derivatives(df):
    # Compute dσ/dK using finite differences
    df["d_sigma_dk"] = df.groupby("timeToMaturity")["impliedVolatility"].diff() / df.groupby("timeToMaturity")["strike"].diff()
    # Compute d²σ/dK² using second-order finite differences
    df["d2_sigma_dk2"] = df.groupby("timeToMaturity")["d_sigma_dk"].diff() / df.groupby("timeToMaturity")["strike"].diff()
    # Compute dσ/dt using finite differences
    df.sort_values(["strike", "timeToMaturity"], inplace=True)  # Ensure sorting for correct differentiation
    df["d_sigma_dt"] = df.groupby("strike")["impliedVolatility"].diff() / df.groupby("strike")["timeToMaturity"].diff()
    return df

def CreateGraphofindicator(str_, df,gauss=False):
    K = df['strike']
    T = df['timeToMaturity']
    volume = df[str_]
    K_grid, T_grid = np.meshgrid(np.unique(K), np.unique(T))  # Grid for plotting
    volume_grid = scipy.interpolate.griddata((K, T), volume, (K_grid, T_grid), method='nearest')  # Interpolated surface

    # Apply Gaussian smoothing to sigma_grid (volatility surface)
    if gauss == True :
        volume_grid = gaussian_filter(volume_grid, sigma=1) 
    # Create a Plotly figure
    fig = go.Figure()

    # Add a 3D surface plot
    fig.add_trace(go.Surface(z=volume_grid, x=K_grid, y=T_grid, colorscale='viridis'))

    # Update layout for better visualization
    fig.update_layout(
        title=str_ + " Surface",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Time to Maturity (T)",
            zaxis_title=str_,
        )   
    )
    fig.update_layout(
    width=800,  # Increase width
    height=800  # Increase height 
    )
    return fig

def calculate_derivatives(ogdf):
    #part 2 : dsdk
    vals = ogdf['timeToMaturity'].unique()
    ogdf['dsdk'] = np.nan
    for i in vals:
        df = ogdf[ogdf['timeToMaturity'] == i].copy()
            
        df.loc[:, 'dsdk'] = (df['impliedVolatility'] - df['impliedVolatility'].shift(1))/(df['strike'] - df['strike'].shift(1))
        
        # Reset the index to ensure we can access rows by integer location
        df = df.reset_index(drop=True)
        
        if len(df) > 1:
            df.loc[0, 'dsdk'] = df.loc[1, 'dsdk']
        else:
            df.loc[0, 'dsdk'] = 0  # or some other appropriate value

        ogdf_indexed = ogdf.set_index('contractSymbol')
        df_indexed = df.set_index('contractSymbol')

        ogdf_indexed.update(df_indexed[['dsdk']])

        # Reset the index to make 'contractSymbol' a column again
        ogdf = ogdf_indexed.reset_index()
    #part 1 dsdt
    vals = ogdf['strike'].unique()
    ogdf['dsdt'] = np.nan
    for i in vals:
        df = ogdf[ogdf['strike'] == i].copy()
        df.loc[:, 'dsdt'] = (df['impliedVolatility'] - df['impliedVolatility'].shift(1))/(df['timeToMaturity'] - df['timeToMaturity'].shift(1))
        
        # Reset the index to ensure we can access rows by integer location
        df = df.reset_index(drop=True)
        
        if len(df) > 1:
            df.loc[0, 'dsdt'] = df.loc[1, 'dsdt']
        else:
            df.loc[0, 'dsdt'] = 0  # or some other appropriate value

        ogdf_indexed = ogdf.set_index('contractSymbol')
        df_indexed = df.set_index('contractSymbol')

        ogdf_indexed.update(df_indexed[['dsdt']])

        # Reset the index to make 'contractSymbol' a column again
        ogdf = ogdf_indexed.reset_index()
    

    #part3 d2sdk2
    vals = ogdf['timeToMaturity'].unique()
    ogdf['d2sdk2'] = np.nan
    for i in vals:
        df = ogdf[ogdf['timeToMaturity'] == i].copy()
            
        df.loc[:, 'd2sdk2'] = (df['dsdk'] - df['dsdk'].shift(1))/(df['strike'] - df['strike'].shift(1))
        
        # Reset the index to ensure we can access rows by integer location
        df = df.reset_index(drop=True)
        
        if len(df) > 1:
            df.loc[0, 'd2sdk2'] = df.loc[1, 'd2sdk2']
        else:
            df.loc[0, 'd2sdk2'] = 0  # or some other appropriate value

        ogdf_indexed = ogdf.set_index('contractSymbol')
        df_indexed = df.set_index('contractSymbol')

        ogdf_indexed.update(df_indexed[['d2sdk2']])

        # Reset the index to make 'contractSymbol' a column again
        ogdf = ogdf_indexed.reset_index()    

    return ogdf

# def CreateMultiGraphs(st1,st2,st3, df):
#     K = df['strike']
#     T = df['timeToMaturity']
#     v1 = df[st1]
#     v2 = df[st2]
#     v3 = df[st3]

#     K_grid, T_grid = np.meshgrid(np.unique(K), np.unique(T))  # Grid for plotting
#     v1grid = scipy.interpolate.griddata((K, T), v1, (K_grid, T_grid), method='linear')  # Interpolated surface
#     v2grid = scipy.interpolate.griddata((K, T), v2, (K_grid, T_grid), method='linear')  # Interpolated surface
#     v3grid = scipy.interpolate.griddata((K, T), v3, (K_grid, T_grid), method='linear')  # Interpolated surface


#     # Apply Gaussian smoothing to sigma_grid (volatility surface)
#     v1grid = gaussian_filter(v1grid, sigma=0.5) 
#     v2grid = gaussian_filter(v2grid, sigma=0.5) 
#     v3grid = gaussian_filter(v3grid, sigma=0.5) 
    
#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add a 3D surface plot
#     fig.add_trace(go.Surface(z=v1grid, x=K_grid, y=T_grid, colorscale='viridis', name=st1,showscale=False))
#     fig.add_trace(go.Surface(z=v2grid, x=K_grid, y=T_grid, colorscale='plasma', name=st2,showscale=False))
#     fig.add_trace(go.Surface(z=v3grid, x=K_grid, y=T_grid, colorscale='cividis', name=st3,showscale=False))


#     # Update layout for better visualization
#     fig.update_layout(
#         title="Implied Volatility Surface",
#         scene=dict(
#             xaxis_title="Strike Price (K)",
#             yaxis_title="Time to Maturity (T)",
#             zaxis_title="Indicators",
#         )
#     )
#     return fig

def local_volatility(df, S0, r=0.0443):
    # Compute d1 for Black-Scholes model
    df["d1"] = (np.log(S0 / df["strike"]) + (r + (df["impliedVolatility"] ** 2) / 2) * df["timeToMaturity"]) / (df["impliedVolatility"] * np.sqrt(df["timeToMaturity"]))

    # Extract derivatives
    dsdk = df["dsdk"]
    d2sdk2 = df["d2sdk2"]
    dsdt = df["dsdt"]
    d1 = df["d1"]

    # Compute numerator and denominator for local volatility
    numerator = df["impliedVolatility"] ** 2 + 2 * df["timeToMaturity"] * df["impliedVolatility"] * dsdt + 2 * r * df["strike"] * df["timeToMaturity"] * df["impliedVolatility"] * dsdk
    denominator = (1 + df["strike"] * d1 * np.sqrt(df["timeToMaturity"]) * dsdk) ** 2 + (df["strike"] ** 2) * df["timeToMaturity"] * df["impliedVolatility"] * (d2sdk2 - d1 * np.sqrt(df["timeToMaturity"]) * (dsdk ** 2))

    # Handle division by zero
    df["LocalVolatility"] = np.where(denominator != 0, numerator / denominator, df["impliedVolatility"])
    return df



# ticker = "AAPL" 
# S0 = LatestClose(ticker)
# dividend_yield = yf.Ticker(ticker).info.get("dividendYield")
# if type(dividend_yield) != float :
#     dividend_yield = 0

# print(dividend_yield)

# calldf = CallDF(ticker)
# calldfFiltred = Krangecreator(calldf, ticker, 80, 120)
# calldfFiltred = Trangecreator(calldfFiltred, 0, 60)
# calldfFiltred = LiquidityFilter(calldfFiltred, 0)
# calldfFiltred = calculate_derivatives(calldfFiltred)



# graph = CreateGraphofindicator('impliedVolatility',calldfFiltred)
# graph.show()

# r_d= 0.0443 - dividend_yield
# calldfFiltred = local_volatility(calldfFiltred, S0, r_d)

# graph = CreateGraphofindicator('LocalVolatility',calldfFiltred)
# graph.show()

# calldfFiltred['difference'] = (calldfFiltred['impliedVolatility'] - calldfFiltred['LocalVolatility'])/calldfFiltred['impliedVolatility'] * 100
# graph = CreateGraphofindicator('difference',calldfFiltred)
# graph.show()

