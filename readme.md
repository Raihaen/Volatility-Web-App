
# Vol APP

This project is an evolving work where I try to apply some of the Markets knowledge I'm cultivating through studying Trading books in real-life-like applications.

My current objective is to construct an app to view the implied volatility surface by the market and use a local volatility model to price some exotic options using the data collected.

## Part 1 : Creating an Implied Volatility Surface

In this part, we develop the implied volatility surface app. We will be mainly using `streamlit` for the interface as well as `scipy` for dynamic graphs.

We first start by importing market data using APIs from the Yahoo Finance website. This was chosen mainly for it being free and simple. However, it should be noted that Yahoo Finance does not provide real-time data but end-of-day option chains. For real-life usages, we would be opting for data provided through Bloomberg or Refinitiv Eikon. The app's architecture, however, would not change greatly. We will just need to log in and handle changes in API structure.

Here is an example of the code that would enable a connection to the Bloomberg Server (B-PIPE) :

```python
import blpapi
from xbbg import blp

# We define server parameters depending on our account

host = "our_bloomberg_server_ip"
port = 8194  # the Default Bloomberg B-PIPE port

# We create a session

sessionOptions = blpapi.SessionOptions()
sessionOptions.setServerHost(host)
sessionOptions.setServerPort(port)

# We start the session

session = blpapi.Session(sessionOptions)
if not session.start():
    print("Failed to start B-PIPE session.")
    exit()

print("Connected to Bloomberg B-PIPE.")

# Real-time price for AAPL

data = blp.live("AAPL US Equity", "Last_Price")
print(data)
```

Once we import our data, we then treat it conveniently and run a reverse Black-Scholes formula to get the implied volatility per price. In theory, and because of the **put-call parity**, using either puts or calls would (theoretically) yield the same implied volatility for either of them given the same strike and time to maturity, this is however not the exact case in reality, a better approach will be using both call and put prices... For simplicity however, we use calls in our code. Extracting the implied volatility can be done either using the API directly, through specialized libraries, or, more primitively, through numerical methods...

### Finding Implied Volatility through Numerical Methods

Due to the complexity of the Black-Scholes equation, there is no closed formula for the reverse function that spits implied volatility values. Instead, we need to use 'smart guesses' to find it, and this is done through numerical methods: mathematical algorithms that try multiple values and converge to the implied volatility value. An example would be the Newton-Raphson method, one of the fastest converging methods with its quadratic convergence rate: $o(h^2)$.

The idea is to start with an initial guess $\sigma$ and then replace it with a new guess $\sigma_{new}$ such that:
$$ \sigma_{new} = \sigma_{old} - \frac{f(\sigma_{old})}{f'(\sigma_{old})}$$
where :

- $f(\sigma_{old}) = C_{B-S}(\sigma_{old})-C_{market}$

- $f'(\sigma_{old})=\nu(C_{B-S})$
  
- $C_{B-S}(\sigma_{old})$ being the theoretical price given by the Black-Scholes formula

- $\nu(C_{B-S}) =\frac{\partial f}{\partial \sigma} =\frac{\partial C}{\partial \sigma}$ aka the call's Vega, since $C_{market}$ is constant.

### Volatility Surface

We can find in the code a function that gets the implied volatility for a specific call (check `ManualIV.py`).
However, and for performance reasons, we will be using in our graph the implied vol values already provided by Yahoo Finance.

We use the ```python matplotlib``` library to graph our surface by creating a `meshgrid`. This uses our K and T values to create a 2D grid that we will be graphing the implied volatility values on. We then use `griddata` to create the Implied Volatility Surface. We will not be using any interpolation/curve fitting, so we will just use the regular `method='linear'`.

Once that is done, we will now go to part 2 where we will turn our script into functions to then build the interface that lets the user manipulate the graph and the info surrounding it using ```python streamlit```.

## Part 2 : Creating the Streamlit Web App

The first thing that needs attention is displaying the graph. `matplotlib`'s graphs are shown as static in `streamlit` and so, we need to use `pyplot` instead. Thankfully, this does not require any big changes. We then proceed to create a slider to enable the user to input the different parameters.

After doing that, we will add some options to the user:

- We will add a new column to our dataframe called moneyness, and add an option to either show strike price or moneyness.

- We also add a range option to graph around the moneyness.

We add a submit button to start the generation and an error text to show the user in case of any problems.

Now that we have a working app, we can move to a much more interesting topic: graphing local volatility.

## Part 3 : Local Volatility (Deterministic - Dupire)

Local Volatility is a type of modeling in which we try to model a continuous volatility surface based on the market's implied Vol to use in pricing Exotic options. Two families of local volatility models are used: exact time fitting and surface modeling, depending on whether the exotic you are pricing is time-dependent or not.

To get our local volatility, we will be using a Dupire formula, as illustrated in *The Art of Modeling Financial Options: Monte Carlo Simulation*, a paper published by Antonie Kotze and Rudolf Oosthuizen. The main Dupire Formula uses Call prices and their derivatives; however, one can extract an equation linking local volatility and implied volatility. The Dupire formula is as follows:

$$ \sigma_{loc}^2(S_0,M,\tau) = \frac{\sigma_{imp}+ 2\tau \sigma_{imp} \frac{\partial \sigma_{imp}}{\partial\tau} + 2 (r-d)K\tau \sigma_{imp} \frac{\partial \sigma_{imp}}{\partial K}}{(1+K d_1 \sqrt{\tau} \frac{\partial\sigma_{imp}}{\partial K})^2 + K^2 \tau \sigma_{imp}(\frac{\partial^2 \sigma_{imp}}{\partial K^2} - d_1 \sqrt{\tau}(\frac{\partial \sigma_{imp}}{\partial K})^2)}$$

`Quantlib` offers a readily available set of tools to create our local vol surface. However, we will be creating our surface without it.

To implement the theory we just discussed, we start by calculating Implied Vol derivatives. This can be done using finite differences as is done in the code, by first filtering through unique values of variable $x$ and then applying $\frac{\partial \sigma}{\partial y} \big|_{i}= \frac{\sigma_{i+1}-\sigma_i}{y_{i+1}-y_i}$.
We use $(i,i-1)$ instead of $(i+1, i-1)$ because of the difference in traded volume between strikes (which I noticed kind of has a $mod\space 2$ trend), this way we "smooth-out" the derivative function.

In our calculations, we will be using the AMERIBOR as a value for $r$ and the dividend yield provided by `yfinance` as $d$.

The robustness of this model can be checked by calculating the integral of local volatility values through the most probable path of the stock with expiry at K (Gatheral, The Volatility Surface: A Practitioner's Guide).

## Part 4 : Local Volatility (Stochastic - SABR)

The SABR Model stands for S Alpha Beta Rho. In this part, I will be relying on Hagan's mythical 2002 paper *MANAGING SMILE RISK*.
We are aiming for a dynamic SADT model, since we have multiple T's. This is done by applying a different calibration for each T and then create the vol surface.
We will be using a lognormal SADT, since (according to my research, a &\beta& =1$ is best suited for equities and derivatives, except in special market conditions). We then calibrate the values for $\alpha$ , $\nu$, and $\rho$. This is can be done through multiple methods of minimization, so as to make the SADT vol as close as possible to the market's implied vol, we minimize the square of the difference between the SADT Vol and the market's IV. after experimenting with minimization algos : `L-BFGS-B`,`Nelder-Mead` and `trust-constr` (all are available in the code). My final choice was `L-BFGS-B` as it's the most suited for our function.
-> For the SADT vol function, thanks to our $\beta$ beign equal to 1, a lot of terms in equation $(2.17a)$ go to zero (the full formula is also inculded to experiment with different values of $\beta$).
As for $r$ and $d$ values, I used Ameribor and dividens obtained via `yfinance`.
The app allows checking both the market IV and the SADT vol, as well as their difference relative to the IV expressed in percentages. We notice that it fits very well, with less than 1% difference near strike, except for misspriced options (probably due to low volumes and so innacurate pricings).

This concludes this part of the project, the next goal will be pricing different sets of exotic options !
