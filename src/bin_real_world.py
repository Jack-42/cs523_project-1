"""
@author Jack Ringer
Date: 2/17/2023
Description:
Script used to bin real-world data from:
https://github.com/owid/covid-19-data/tree/master/public/data/jhu
"""
import pandas as pd


def main(cases_pth: str, deaths_pth: str):
    """
    Bin covid world data.
    :param cases_pth: str, path to new cases data
    :param deaths_pth: str, path to new deaths data
    :return: None
    """
    cases_df = pd.read_csv(cases_pth)
    deaths_df = pd.read_csv(deaths_pth)
    world_cases_df = cases_df['World']
    world_deaths_df = deaths_df['World']
    print(world_cases_df.head())
    print(world_deaths_df.head())


if __name__ == "__main__":
    cases_path = "../data/real_world/new_cases.csv"
    deaths_path = "../data/real_world/new_deaths.csv"
    main(cases_path, deaths_path)
