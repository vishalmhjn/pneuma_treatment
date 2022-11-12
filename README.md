# pneuma_open_traffic

Trajectory data collected from Drones over an urban area in Athens. For more info: https://open-traffic.epfl.ch/

<p align="center">
<img src="media/pneuma_overview.gif" alt="drawing" width="400" align="center"/>
</p>

This repository provides an automated way to treat the noise and anomalies (in the form of unrealistic peaks) in the acceleration values of a number of vehicles in the pneuma dataset. The algorithm uses a combination of low-pass filters (Savitzky-Golay filter, Gaussian filter) to remove the noise. A machine learning model (XGBoost) is used to reconstruct acceleration the time-series and detect the anomalies. If anomalies are detected, they are removed. The outputs are the processed time-series profiles of vehicle speeds and accelerations without high frequency noise and unrealistic acceleration peaks.

# Code 
You should be in [src](src/) folder when running the codes.
### Raw data to Long format
1. Specify the path to the raw data from pNEUMA website in the [data_formatter.py](src/data_formatter.py) and run it to convert the files from wide to long format. A [sample](data/sample_data.csv) output is provided.
### Treating noise and anomalies
2. Specify the path to the output file from step-1, other paths (to save output data and plots) in the [processor.py](src/processor.py). Run this file to get the treated speed and acceleration time series.

### Outputs
3. With the default paths, the plots are saved in [plots](plots/) and data is saved in [data/derived](data/derived).