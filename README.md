# CS523 Project 1

#### By Jack Ringer and James Morris

This repository contains the code and configuration files used for project 1 of
CS523. The project report contains an overview of our results.

## Project Guide
The "data\" directory contains the raw data used in making our report.

The configuration files used to generate stable, periodic, and chaotic dynamics
in SIMCoV may be found under the "simcov_config_files\" directory

All code snippets used to generate figures and logistic map data can be found
in the "src/" directory.

An overview of what is implemented in each of these files is given as follows:

* bin_real_world.py: script that bins data from real-world COVID-19 cases/deaths
* bin_simcov_stats.py: scripts that bins data from SIMCoV .stats files
* logistic_map.py: contains utilities for generating, plotting, and binning
  logistic map data
* make_figure1.py: script to generate figure 1 from our report. We note that
  this
  code is based upon the script
  from: https://github.com/AdaptiveComputationLab/simcov/blob/master/scripts/dynamic_plot.py
* make_figure2_n_4.py: script to generate figures 2 and 4 from our report
* make_figure5.py: script to generate figure 5 from our report

For more information please see the project report.

## Setup

All necessary packages to run the code in this project can be installed with:

pip install -r requirements.txt
