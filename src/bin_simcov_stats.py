import numpy as np
import pandas as pd

"""
@author Jack Ringer
Date: 2/16/2023
Description:
Simple script to bin SimCov .stats data for analysis by JIDT. Will only save
information of interest (viral load and expressing cells.
"""


def main(file_path: str, save_path: str, cutoff: int = None):
    """
    Bin .stats data and save to .txt file
    :param file_path: str, path to .stats file to be binned
    :param save_path: str, path to save binned data to
    :param cutoff: int (optional), use only data from time > cutoff
    :return: None
    """
    df = pd.read_csv(file_path, sep='\t')
    df = df.rename(columns={'# time': 'time'})
    if cutoff is not None:
        df = df.loc[df['time'] > cutoff]
    df['tot_infected'] = df['incb'] + df['expr']
    df = df[['tot_infected', 'virs']]
    # tot_infected col 1, virs col 2
    df['tot_infected'].plot()
    import matplotlib.pyplot as plt
    plt.show()
    df_bin = df
    n_bins = 20
    labels = np.arange(n_bins)
    df_bin['tot_infected'] = pd.cut(df['tot_infected'], labels=labels,
                                    bins=n_bins)
    df_bin['virs'] = pd.cut(df['virs'], labels=labels, bins=n_bins)
    df_bin.to_csv(save_path, header=None, index=None, sep=' ')
    print("Successfully saved binned data to:", save_path)
    return


if __name__ == "__main__":
    config = 'chaotic'
    fpath = "../data/simcov/%s/run-1/simcov.stats" % config
    sv_path = "../data/simcov/binned_cutoff/%s.txt" % config
    main(fpath, sv_path)
