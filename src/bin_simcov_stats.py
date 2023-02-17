import numpy as np
import pandas as pd

"""
@author Jack Ringer
Date: 2/16/2023
Description:
Simple script to bin SimCov .stats data for analysis by JIDT. Will only save
information of interest (viral load and expressing cells.
"""


def main(file_path: str, save_path: str):
    """
    Bin .stats data and save to .txt file
    :param file_path: str, path to .stats file to be binned
    :param save_path: str, path to save binned data to
    :return: None
    """
    df = pd.read_csv(file_path, sep='\t')
    df['tot_infected'] = df['incb'] + df['expr']
    df = df[['tot_infected', 'virs']]
    # tot_infected col 1, virs col 2
    df_bin = df
    n_bins = 20
    labels = np.arange(n_bins)
    df_bin['tot_infected'] = pd.cut(df['tot_infected'], labels=labels,
                                    bins=n_bins)
    df_bin['virs'] = pd.cut(df['virs'], labels=labels, bins=n_bins)
    df_bin.to_csv(save_path, header=None, index=None, sep=' ', mode='a')
    print("Successfully saved binned data to:", save_path)
    return


if __name__ == "__main__":
    fpath = "../data/simcov/periodic/c7-1/simcov.stats"
    sv_path = "../data/simcov/binned/periodic.txt"
    main(fpath, sv_path)
