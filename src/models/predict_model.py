# Intent: Output whatever data is necessary for out-of-sample predictions.
# Should train model using ALL processed data.
# Input path: /data/processed
# Output path: /reports
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import minimize
import numpy as np
from train_model import train_model
import click
import torch


def bs_price(right, S, K, T, sigma, r):
    """
    Return's option price via Black-Scholes

    right: "P" or "C"
    S: Underlying price
    K: Strike price
    T: time to expiration (in fractions of a year)
    sigma: volatility of the underlying
    r: interest rate (in annual terms)
    """
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S / K) + (r + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    if right == "C":
        price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        return price

    if right == "P":
        price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
        return price


def log_wealth_optim(f, pnl):
    """
    Returns the negative of log wealth for optimization
    """
    return -np.mean(np.log(1 + f * pnl))


class ShortIronCondor:
    def __init__(self, put, call, hedge_put, hedge_call):
        """
        Class defining a short Iron Condor options position.
        put: short put
        call: short call
        hedge_put: long put
        hedge_call: long call
        """
        self.put = put
        self.call = call
        self.hedge_put = hedge_put
        self.hedge_call = hedge_call
        self.premium = (call["mid_price"] + put["mid_price"]) - (
            hedge_put["mid_price"] + hedge_call["mid_price"]
        )
        # Deflate the premium by 15% to be conservative and account for slippage
        self.premium = 0.85 * self.premium
        self.max_loss = put["strike"] - hedge_put["strike"] - self.premium

    def pnl(self, underlying_price):
        """
        Calculates profit and loss of position at expiration given the underlying price.
        """
        # If underlying is between short strikes
        if self.call["strike"] > underlying_price > self.put["strike"]:
            pnl = self.premium
        # If underlying is under the short put
        elif underlying_price < self.put["strike"]:
            pnl = max(
                (underlying_price - self.put["strike"]) + self.premium, -self.max_loss
            )
        # If underlying is above short call
        elif underlying_price > self.call["strike"]:
            pnl = max(
                (self.call["strike"] - underlying_price) + self.premium, -self.max_loss
            )

        return pnl


class OptionPosition:
    def __init__(self, chain, vols, dte, risk_free_rate):
        """
        Class for all information required to determine what option position to enter into
        and the kelly sizing percentage

        chain: pandas dataframe containing options chain
        vols: sample of future WEEKLY volatilities
        dte: days to expiration of options contracts given in chain
        risk_free_rate: current market risk free rate of return given in annualized terms
        """
        self.chain = chain
        self.vols = vols
        self.underlying_price = self.chain.iloc[0]["underprice"]
        self.dte = dte
        self.risk_free_rate = risk_free_rate

        self.calc_values()
        (
            self.short_put,
            self.short_call,
            self.long_put,
            self.long_call,
        ) = self.find_contracts()
        self.position = ShortIronCondor(
            self.short_put, self.short_call, self.long_put, self.long_call
        )

    def calc_values(self):
        """
        Calcuates Mid price and skew premium for each option in chain
        """
        atm_contract_index = (
            np.abs(self.chain["strike"] - self.underlying_price)
        ).idxmin()
        atm_impliedvol = self.chain.iloc[atm_contract_index]["impvol"]

        # Calculate option value for all options using ATM volatility
        self.chain["model_value"] = self.chain.apply(
            lambda x: bs_price(
                x["right"],
                x["underprice"],
                x["strike"],
                self.dte / 252,
                atm_impliedvol,
                self.risk_free_rate,
            ),
            axis=1,
        )
        self.chain["mid_price"] = (self.chain["bid"] + self.chain["ask"]) / 2
        self.chain["skew_premium"] = self.chain["mid_price"] - self.chain["model_value"]

    def find_contracts(self):
        """
        Finds put contract with highest skew premium, then call contract with closest delta.
        Then picks hedging contracts on either side so that required margin equals $1000
        Essentially, picks contracts for short Iron Condor position.
        """
        # Select a put to short that is OTM
        short_put = self.chain[
            (self.chain["right"] == "P")
            & (self.chain["strike"] < self.underlying_price)
        ]["skew_premium"].idxmax()
        short_put = self.chain.iloc[short_put]
        # Buy put option so our margin required is $1000
        long_put = self.chain[
            (self.chain["strike"] == (short_put["strike"] - 10))
            & (self.chain["right"] == "P")
        ].squeeze()

        # Find the corresponding call option to make the position delta neutral
        put_contract_delta = short_put["delta"]
        short_call = np.abs(
            self.chain[self.chain["right"] == "C"]["delta"] + put_contract_delta
        ).idxmin()
        short_call = self.chain[self.chain["right"] == "C"].iloc[short_call]
        # Find respective call hedge option
        long_call = self.chain[
            (self.chain["strike"] == (short_call["strike"] + 10))
            & (self.chain["right"] == "C")
        ].squeeze()

        return short_put, short_call, long_put, long_call

    def calc_kelly(self):
        """
        Simulates future returns and determines option position PNL.
        Returns the kelly criterion betting percentage by optimizing the log of wealth
        """
        # Sample returns from Student's T with 5 degrees of freedom to account
        # for kurtosis risk.
        returns = t.rvs(5, 0, self.vols)
        prices = self.underlying_price * (1 + returns)
        vfunc = np.vectorize(self.position.pnl)
        # Each option is 100 shares and return is based on an investment of $1000,
        # so 100 / 1000 = 10
        pnl = vfunc(prices) / 10

        print(f"Short put strike: {self.short_put['strike']}")
        print(f"Short call strike: {self.short_call['strike']}")
        print(f"Probability of Profit: {round(len(pnl[pnl > 0]) / len(pnl) * 100, 2)}%")
        print(f"Mean Profit on Win: {round(np.mean(pnl[pnl > 0]) * 100, 2)}%")
        print(f"Mean Profit on Loss: {round(np.mean(pnl[pnl < 0]) * 100, 2)}%")
        print(
            f"Forecasted 5-day Volatility (Annualized): {round(np.median(self.vols) * np.sqrt(252 / 5) * 100, 2)}%"
        )
        print(
            f"5 / 95 percentile Volatility: {round(np.percentile(self.vols, 5) * np.sqrt(252 /5) * 100, 2)}% :: {round(np.percentile(self.vols, 95) * np.sqrt(252 /5) * 100, 2)}%"
        )
        print(f"Forecasted 5-day Price: {round(np.median(prices), 2)}")
        print(
            f"5 / 95 percentile Price: {int(np.percentile(prices, 5))} :: {int(np.percentile(prices, 95))}"
        )

        initial = np.random.rand()
        result = minimize(log_wealth_optim, initial, (pnl))

        print(f"Kelly Bet Percentage: {round(result.x[0] * 100, 2)}%")


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def main(input_path, output_path):
    # TODO: Shift dropna to build_features.py
    indep_vars = pd.read_csv(
        input_path + "/indep_vars.csv", index_col="date", parse_dates=True
    ).dropna()
    dep_var = pd.read_csv(
        input_path + "/dep_var.csv", index_col="date", parse_dates=True
    ).dropna()

    # Find common index between our input and output data so everything has the same length.
    # This data is used for training the model.
    common_index = indep_vars.index.intersection(dep_var.index)
    indep_vars_train = indep_vars.loc[common_index]
    dep_var = dep_var.loc[common_index]
    # Convert to tensors
    indep_vars_train = torch.Tensor(indep_vars_train.values)
    # Model expects column vector input
    dep_var = torch.Tensor(dep_var.values).view(-1, 1)

    model = train_model(indep_vars_train, dep_var)
    # Select most recent data to predict future volatility
    # Model expects row vector as input
    indep_vars_predict = torch.Tensor(indep_vars.iloc[-1]).view(1, -1)

    vols = model(indep_vars_predict).sample([100000]).squeeze().numpy()
    # Un-transform vols to standard deviation
    vols = np.sqrt(np.exp(vols))
    # Inflate volatility by 10% to be conservative about bet sizing
    vols = vols * 1.10

    # Get Option chain data and define parameters
    # TODO: Make parameters as input instead of fixed
    chain = pd.read_csv(input_path + "/option_chain.csv")
    dte = 5
    risk_free_rate = 0.0006
    position = OptionPosition(chain, vols, dte, risk_free_rate)

    # Print date of last variable observation for error checking
    print(indep_vars.index[-1])

    # TODO: Make this better and output to a file in the reports section
    position.calc_kelly()


if __name__ == "__main__":
    main()