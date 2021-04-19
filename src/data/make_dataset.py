# Intent: Do anything required to generate data needed for input features to the model.
# Also load environment variables. Output to /data/raw
import click
import os
import sqlite3
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("output_path", type=click.Path())
def main(output_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    conn = sqlite3.Connection(os.environ.get("DATABASE_PATH"))

    spx_data = pd.read_sql(
        "SELECT * FROM prices WHERE ticker='^GSPC'",
        conn,
        index_col="date",
        parse_dates="date",
    )
    spx_returns = np.log(spx_data["close"]) - np.log(spx_data["close"].shift(1))
    spx_returns.to_csv(output_path + "/spx_returns.csv", header=["spx_returns"])

    vix_data = pd.read_sql(
        "SELECT * FROM prices WHERE ticker='^VIX'",
        conn,
        index_col="date",
        parse_dates="date",
    )
    # This puts it into units of daily standard deviation
    vix = vix_data["close"] / np.sqrt(252) / 100
    vix.to_csv(output_path + "/vix.csv", header=["vix"])


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
