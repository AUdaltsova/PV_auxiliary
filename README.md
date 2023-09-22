# PV_auxillary
A study on the effects of auxillary data on the performance of an LSTM solar PV forecasting model

## Abstract
Short-term forecasting (nowcasting) of solar energy sources' outputs is an integral part of the successful decarbonization of energy grids. Without accurate prediction of solar power sources’ output, it is impossible to anticipate the amount of electricity available and provide a robust supply of energy to the consumers. 

Currently, the majority of forecasting techniques rely on previous solar photovoltaic (PV) sources' outputs for prediction, or sometimes take into account auxiliary data such as weather or outputs of the nearby solar power sources. However, combinations of those are rare, and the contribution of auxillary data to the overall predictive power of the solution is not evaluated. This project aims to test the advantage of using any of those sources of data, as well as whether any significant improvement can be achieved when using them in combination.

## Datasets
1. [openclimatefix/uk_pv](https://huggingface.co/datasets/openclimatefix/uk_pv) dataset was usead as the primary dataset of photovoltaic power generation
2. [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) dataset was used as the source of the meteorological data, namely total cloud cover, skin temperature, and surface solar radiation

The authors acknowledge the use of the UCL Myriad High Performance Computing Facility (Myriad@UCL), and associated support services, in the completion of this work.

## Hyperparameter tuning
Done via optuna on UCL Myriad Cluster. Scrips used can be found in [optuna_scripts](https://github.com/AUdaltsova/PV_auxillary/tree/main/optuna_scrips)

## Model training
Done in Google Colaboratory on V100 GPU. Scripts used can be found in [model_training](https://github.com/AUdaltsova/PV_auxillary/tree/main/model_training)
