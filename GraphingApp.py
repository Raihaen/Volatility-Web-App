from DataFunctions import *
import streamlit as st
import plotly.io as pio
import plotly.tools as tls
from pathlib import Path
from DataImpLib import *
from SabrFunctions import *
from ManualIV import *



def read_markdown_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def check_ticker_valid(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Try fetching data for the ticker
        data = stock.history(period="1d")
        if data.empty:
            return False  # No data found, invalid ticker
        return True  # Ticker is valid
    except ValueError:
        return False  # Ticker is invalid
    except Exception as e:
        print(f"Error: {e}")
        return False 



if "page" not in st.session_state:
    st.session_state.page = "home"

# Function to change the page
def go_to_page(page_name):
    st.session_state.page = page_name

# Page navigation
if st.session_state.page == "home":


    st.title("Implied Vol sruface generator")
    DescriptionParagraph = """
                This web app allows visualizing the market's **implied volatility**, the **SABR volatility**, as well as **Dupire's local volatility**.
                
                Select a type of Vol in the slider and fill the necessary data. Click here to visit my [github](https://https://github.com/Raihaen/Volatility-Web-App)
                """
    st.markdown(  DescriptionParagraph , unsafe_allow_html=True)
    

    with st.sidebar: #we create a sidebar to input user parameters

        option1 = 'Implied Volatility'
        option2 = 'SABR Local Volatility'
        option3 = 'Local Volatility - Dupire'
        option = st.selectbox("Pick a sruface", [option1,option2 ,option3 ], index=1 ) #Show SADT First
        ticker = st.text_input("Enter the ticker name",value='TSLA')
        K_min = st.number_input("Enter the minimal moneyness value, (for 70% type 70)",value=70)
        K_max = st.number_input("Enter the maximal moneyness value, (for 130% type 130)",value=130)

        T_min = st.number_input("Enter the minimal Time to Expiery value, (in days)",value=180)
        T_max = st.number_input("Enter the maximal Time to Expiery value, (in days)", value=360)

        bound = st.number_input("Enter the minimal liquidity to take into account", value=0)
        gauss = st.checkbox("Use a guassian filter to smooth the functions", value= False)
    #    submit = st.button("Update")
        







    #if submit & (check_ticker_valid(ticker)!= False):

    if  (check_ticker_valid(ticker)!= False):
        calldf = CallDF(ticker)
        if calldf.empty :  
            st.text("No options found on this ticker")
        else :
            calldfFiltred = Krangecreator(calldf, ticker, K_min, K_max)
            calldfFiltred = Trangecreator(calldfFiltred, T_min, T_max)
            calldfFiltred = LiquidityFilter(calldfFiltred, bound)        
            if option == 'Implied Volatility' :
                if calldfFiltred.empty:
                    st.text("Dataset is empty")
                else:
                    ## implied vol provided by Yahoo Finance :
                    #st.markdown("### The implied volatiliy graph")
                    sigma, K, T = getindicators(calldfFiltred)
                    graph = CreateGraph(sigma, K, T, gauss)
                    #check if df is empty or not before creating the plot
                    st.plotly_chart(graph, use_container_width=True)

        #this part includes the manual implied vol. It doesn't figue in the app because it's super slow. You're welcome to fork this and try it on your computer !

                    # st.markdown("### The implied volatiliy graph created using numerical methods (takes a bit of time to load):")
                    # calldfFiltred = ManualIV(calldfFiltred, ticker)
                    # graph = CreateGraph('implied_vol', K, T,gauss)
                    # st.plotly_chart(graph, use_container_width=True)


            elif option == option3 :
                if calldfFiltred.empty:
                    st.text("Dataset is empty")
                else:
                    calldfFiltred = calculate_derivatives(calldfFiltred)
                    r_d= 0.0443 - dividend_yield(ticker)
                    calldfFiltred = local_volatility(calldfFiltred, LatestClose(ticker), r_d)
                    graph = CreateGraphofindicator('LocalVolatility',calldfFiltred,gauss)
                    st.plotly_chart(graph, use_container_width=True)
                    st.text("This was created by applying Dupire's formula; with a deterministic volatility function.")
                    st.text("-> The robustess of the model has been done by calculating the inegral of local volatility values through the most probable path of the stock with expiery at K.")
                    st.markdown("## Here are the derivatives used :")
                    graph = CreateGraphofindicator('dsdk',calldfFiltred,gauss)
                    st.plotly_chart(graph, use_container_width=True)  
                    st.markdown(r"Above is the $\frac{\partial S}{\partial K}$ Derivative.")          
                    graph = CreateGraphofindicator('dsdt',calldfFiltred,gauss)
                    st.plotly_chart(graph, use_container_width=True)
                    st.markdown(r"Above is the $\frac{\partial S}{\partial T}$ Derivative.")          
                    graph = CreateGraphofindicator('d2sdk2',calldfFiltred,gauss)
                    st.plotly_chart(graph, use_container_width=True) 
                    st.markdown(r"Above is the $$ \frac{\partial^2 S}{\partial K^2} $$ Derivative.")          


                # text = Path(r"readme.md").read_text()
                # st.markdown(text, unsafe_allow_html=True)
            elif option == option2 :
                if calldfFiltred.empty:
                    st.text("Dataset is empty")
                else:            
                    calldfFiltred = SABRCalculator(calldfFiltred, ticker)
                    calldfFiltred['(IV-SABR)/IV (in %)'] = (calldfFiltred['impliedVolatility']-calldfFiltred['sabr_vol'])/calldfFiltred['impliedVolatility']*100
                    graph = CreateGraphofindicator('sabr_vol',calldfFiltred,gauss)
                    st.plotly_chart(graph,use_container_width=True)
                    st.markdown('### Comparison with IV')
                    graph = CreateGraphofindicator('impliedVolatility',calldfFiltred,gauss)
                    st.markdown("### The values below are in Percentages :")
                    st.plotly_chart(graph,use_container_width=True)                               
                    graph = CreateGraphofindicator('(IV-SABR)/IV (in %)',calldfFiltred,gauss)
                    st.plotly_chart(graph,use_container_width=True)
                    st.markdown("-> Notice that the difference very small near $S_0$, except for option with inaccurate implied volatilities (probably due to non accurate pricing).")


    elif check_ticker_valid(ticker) == False :
        st.text("Please enter a valid ticker")
    # if st.button("Check the git hub"):
    #     go_to_page("page2") 


# if st.session_state.page == "page2":
#     text = Path(r"readme.md").read_text()
#     st.markdown(text, unsafe_allow_html=True)
#     if st.button("Go to Home"):
#         go_to_page("home") 
