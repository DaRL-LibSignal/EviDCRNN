# Deep Evidential Learning in Diffusion Convolutional Recurrent Neural Network

This repository contains the code and data for the paper **"Deep Evidential Learning in Diffusion Convolutional Recurrent Neural Network"**. 
The baseline codebase is sourced from [Spatiotemporal_UQ](https://github.com/Rose-STL-Lab/Spatiotemporal_UQ) and [evidential-deep-learning](https://github.com/aamini/evidential-deep-learning).

## Repository Structure

- `traffic_data`: Contains the dataset used for training and evaluation.
- `plot`: Contains scripts and tools for plotting results and visualizations.
- `sq_model`: Implementation of the  Spline Quantile regression.
- `sgmcmc_model`: Implementation of the Stochastic Gradient Markov Chain Monte Carlo model.
- `quantile_model`: Implementation of the quantile regression model.
- `point_model`: Implementation of the point estimate model.
- `maemis_model`: An Implementation of the baseline masmis method.
- `dropout_model`: An Implementation of the baseline dropout method.
- `dcrnn_edl_model_original`: Original implementation of the DCRNN model with evidential deep learning.
- `dcrnn_edl_model_fac_0.1`: Implementation of the DCRNN model with evidential deep learning with the hyperparameter of 0.1.
- `dcrnn_edl_model_60`: Main implementation of the DCRNN model with evidential deep learning.
- `dcrnn_MT_ENet_model_60`: Implementation of the DCRNN model with evidential depp learning with MT-loss.
- `dcrnn_edl_model_MSE_60`: Implementation of the DCRNN model with evidential deep learning with Mean Squared Error loss.

The notation '60' refers predicting changes in the graph for the next 60 minutes.

## Core Content

The core content of this repository is the implementation of the DCRNN with evidential deep learning (`dcrnn_edl`). This includes various versions and settings as detailed in the repository structure.

## Dataset

Download traffic/METR-LA to traffic/uq_model/data folder (uq_model includes dropout_model, maemis_model, quantile_model, and sq_model).

You can download the data from link of "https://drive.google.com/drive/folders/102QfowJq7zmyR3W5LjF1K_eNZ0eAA8RL"

## Model Training and Evaluation
```
# traffic (Different models require slight modifications)
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
python run_demo_pytorch.py
```

## Installation

You can install the necessary packages using `pip`:
```sh
pip install -r requirements.txt
```

