# Intent: Use raw data and create necessary features for training / predicting model
# Input path: /data/raw
# Output path: /data/processed
import numpy as np
import pandas as pd
import click


def rv_calc(data):
    """
    Calculates daily realized variance using intraday data.
    data: Pandas dataframe with datetime index

    Returns: Pandas series
    """
    results = {}

    for idx, data in data.groupby(data.index.date):
        returns = np.log(data["close"]) - np.log(data["close"].shift(1))
        results[idx] = np.sum(returns ** 2)

    return pd.Series(results)


def create_lags(series, lags, name="x"):
    """
    Creates a dataframe with lagged values of the given series.
    Generates columns named x_t-n which means the value of each row is the value of the original
    series lagged n times

    series: Pandas series
    lags: number of lagged values to include
    name: String to put as prefix for each column name

    Returns: Pandas dataframe
    """
    result = pd.DataFrame(index=series.index)
    result[f"{name}_t"] = series

    for n in range(lags):
        result[f"{name}_t-{n+1}"] = series.shift((n + 1))

    return result


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def main(input_path, output_path):
    spx_minute = pd.read_csv(
        input_path + "/SPX_1min.csv",
        header=0,
        names=["datetime", "open", "high", "low", "close"],
        index_col="datetime",
        parse_dates=True,
    )
    spx_variance = rv_calc(spx_minute)

    spx_returns = pd.read_csv(
        input_path + "/spx_returns.csv", index_col="date", parse_dates=True
    )

    vix = pd.read_csv(input_path + "/vix.csv", index_col="date", parse_dates=True)

    # Use log of vix and variance for better distributional qualities.
    # Maximum lag: 21 trading days
    vix_lags = create_lags(np.log(vix), 21, name="vix")
    return_lags = create_lags(spx_returns, 21, name="returns")
    rv_lags = create_lags(np.log(spx_variance), 21, name="variance")

    indep_vars = pd.concat([vix_lags, return_lags, rv_lags], axis=1)
    indep_vars.to_csv(output_path + "/indep_vars.csv", index_label="date")

    # What we cant to forecast is the volatility over the next 5 trading days
    # So our dependent variable is the rolling 5-day volatility shifted back 5 days
    dep_var = spx_variance.rolling(5).sum().shift(-5)
    # And logged for better distributional qualities
    dep_var = np.log(dep_var)
    dep_var.to_csv(
        output_path + "/dep_var.csv", index_label="date", header=["variance"]
    )


if __name__ == "__main__":
    main()
