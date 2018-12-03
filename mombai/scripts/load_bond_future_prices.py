from mombai import Dictable
import pandas as pd
import numpy as np

"""
This is a simple starting scripts for loading bond data into a dictable. Once there, we can run the analysis per each market
"""
config = pd.read_html('https://github.com/robcarver17/pysystemtrade/blob/master/data/futures/csvconfig/instrumentconfig.csv')[0]
c = Dictable(config) - 'Unnamed: 0' ## get read of this silly column
bonds = c.inc(AssetClass = 'Bond') ## we just want bond futures
bonds = bonds(fn = lambda Instrument: 'https://github.com/robcarver17/pysystemtrade/blob/master/data/futures/adjusted_prices_csv/%s.csv'%Instrument)
bonds = bonds(adj = lambda fn: pd.read_html(fn)[0])
bonds[lambda adj: adj.columns]
bonds = bonds.do(lambda df: df[['DATETIME', 'PRICE']].set_index('DATETIME'), 'adj') ## we can operate on pd.DataFrames object, all living within bonds.




