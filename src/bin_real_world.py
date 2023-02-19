"""
@author Jack Ringer
Date: 2/17/2023
Description:
Script used to bin real-world data from:
https://github.com/owid/covid-19-data/tree/master/public/data/jhu
"""
import pandas as pd
import numpy as np


def main(cases_pth: str, deaths_pth: str, save_path: str):
    """
    Bin covid world data.
    :param cases_pth: str, path to new cases data
    :param deaths_pth: str, path to new deaths data
    :return: None
    """
    cases_df = pd.read_csv(cases_pth)
    deaths_df = pd.read_csv(deaths_pth)
    
    #world_cases_df = cases_df.rename(columns={'World': 'Cases'})['Cases']
    world_deaths_df = deaths_df.rename(columns={'World': 'Deaths'})['Deaths']
    df = pd.DataFrame([world_deaths_df]).T
    df_bin = df
    n_bins = 15
    labels = np.arange(n_bins)
    #df_bin['Cases'] = pd.cut(df['Cases'], labels=labels,
     #                               bins=n_bins)
    df_bin['Deaths'] = pd.cut(df['Deaths'], labels=labels, bins=n_bins)
    df.to_csv(save_path, header=None, index=None, sep=' ', mode='a')


if __name__ == "__main__":
    cases_path = "../data/real_world/new_cases.csv"
    deaths_path = "../data/real_world/new_deaths.csv"
    sv_path = "../data/real_world/binned_deaths.txt"
    main(cases_path, deaths_path, sv_path)
    
