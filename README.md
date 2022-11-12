# pneuma_open_traffic

Trajectory data collected from Drones over an urban area in Athens. For more info: https://open-traffic.epfl.ch/

<p align="center">
<img src="media/pneuma_overview.gif" alt="drawing" width="400" align="center"/>
</p>

This repository provides an automated way to treat the noise and anomalies (in the form of unrealistic peaks) in the acceleration values of a number of vehicles in the pneuma dataset. The outputs are the processed time-series profiles of vehicle speeds and accelerations.

# Code 
1. Specify the path to the raw data in the [data_formatter.py](src/data_formatter.py) and run it to convert the files from wide to long format. 
2. Specify the path to the output file from step-1, other paths (to save output data and plots) in the [processor.py](src/processor.py). Run this file to get the treated speed and acceleration.
