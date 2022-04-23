# ds-projects-capstone

Capstone project of the neue fische bootcamp on

***Predictive maintenance of HDDs in data centers***.

## Introduction

Predicting health of hard drive disks (HDDs) in data centers allows to maximize HDD usage and ensures data availability. We explore the [S.M.A.R.T. characteristics](https://en.wikipedia.org/wiki/S.M.A.R.T.) of HDDs using anomaly detection methods and develop a classification model that predicts imminent HDD failure. A [scalable model pipeline](https://hdd-predicitve-maintenance-api.herokuapp.com) powers the [customer dashboard](https://share.streamlit.io/felix-roc/ds-projects-capstone/deployment_streamlit).

## Content

This repository contains three branches.
- The `main` branch contains
  - the data
  - `notebooks` for [EDA](https://github.com/felix-roc/ds-projects-capstone/blob/main/notebooks/felix-EDA.ipynb), [anomaly detection](https://github.com/felix-roc/ds-projects-capstone/blob/main/notebooks/felix-anomaly.ipynb), [baseline model](https://github.com/felix-roc/ds-projects-capstone/blob/main/notebooks/felix-baseline.ipynb), [feature engineering](https://github.com/felix-roc/ds-projects-capstone/blob/main/notebooks/felix-features.ipynb) and [modeling](https://github.com/felix-roc/ds-projects-capstone/blob/main/notebooks/felix-modeling.ipynb)
  - refactored code in the `src` folder
  - images, logos and the [stakeholder presentations](https://github.com/felix-roc/ds-projects-capstone/blob/main/reports/presentation/To%20fail%20or%20not%20to%20fail%20-%20That's%20the%20question.pdf) in the `presentation` folder
  - trained `models` for deployment
- The `deployment-model` branch contains a container for model deployment to [Heroku](https://www.heroku.com). The model is interfaced via an [API](https://hdd-predicitve-maintenance-api.herokuapp.com) using FastAPI.
- The `deployment-dashboard` branch contains a Plotly Dash dashboard that uses predictions obtained from the deployed model. This branch is also [deployed to Heroku](https://www.heroku.com).
- The `deployment-streamlit` branch contains the Streamlit app with a [dashboard](https://share.streamlit.io/felix-roc/ds-projects-capstone/deployment_streamlit) to get predictions on sample data and upload custom data.

## Data

The dataset is a excerpt from the data published by [Backblaze](https://www.backblaze.com/b2/hard-drive-test-data.html). It contains:
* all drives of a certain model: Seagate ST4000DM000 that failed in 2020 or 2021
* daily reported drive stats for those drives between 2019 and drive failure

To facilitate reproducing the results, the data is included in the `data.zip` file inside this repository.

## Installation and Setup

The repository uses `venv` to create a virtual environment. For installation run
```
make setup
```
Install the refactored code with
```
python setup.py install
```
In order to preprocess the training data run
```
make data
```
Afterwards, the model is trained with `python -m src.train`, predictions on test data are obtained by running `python -m src.predict`.
