# vignette-anomaly-detection

Vignette on implementing anomaly detection using climate data to detect drastic changes, cloud computing data to detect malicious attacks, and gold price data to detect price spikes; created as a class project for PSTAT197A in Fall 2024

## Contributors

Justin Lang, Ivan Li, Sanchit Mehtora, Keon Dibley

## Vignette Abstract

This vignette utilizes a combination of machine learning techniques, including Isolation Forests and classical time series models to analyze and predict potential anomalies and trends in several datasets, showcasing their wide range of applications. Specifically, we analyzed climate data to find drastic environmental changes, cloud computing data to detect abnormal network attacks, credit card transaction data for fraud detection, and stock market data for irregular trading patterns. By identifying irregularities in data, businesses and companies can identify problems, use model results to employ effective solutions, and forecast trends to prepare for any issues.

## Repository Contents

data - a folder containing the raw and preprocessed datasets used for our methods

scripts - a folder containing separate Jupyter notebooks and Python files used for the analyses

vignette_anomaly.ipynb - the primary vignette document, which is Jupyter notebook containing all integrated code, results, and analyses. This document also functions as a script with line annotations that replicates all the results shown, assuming that the user's environment have the datasets loaded properly

## Reference List

1) T. L. Yasarathna and L. Munasinghe, "Anomaly detection in cloud network data," 2020 International Research Conference on Smart Computing and Systems Engineering (SCSE), Colombo, Sri Lanka, 2020, pp. 62-67, doi: 10.1109/SCSE49731.2020.9313014. keywords: {Cloud computing;Anomaly detection;Data models;Modeling;Security;Machine learning;Support vector machines;Anomaly Detection;Cloud Computing;Machine Learning;One-Class Classification}

2) Tanja Hagemann and Katerina Katsarou. 2021. A Systematic Review on Anomaly Detection for Cloud Computing Environments. In Proceedings of the 2020 3rd Artificial Intelligence and Cloud Computing Conference (AICCC '20). Association for Computing Machinery, New York, NY, USA, 83–96. https://doi.org/10.1145/3442536.3442550

3) Lorenzo Menculini, Andrea Marini, Proietti and Marcello Marconi, 2021. Comparing Prophet and Deep Learning to ARIMA in Forecasting Wholesale Food Prices. https://www.researchgate.net/publication/354631436_Comparing_Prophet_and_Deep_Learning_to_ARIMA_in_Forecasting_Wholesale_Food_Prices.

4) Gold Prices: Working data - https://www.kaggle.com/code/gvyshnya/gold-future-prices-ts-comprehesive-eda?select=future-gc00-daily-prices.csv

5) Cloud Network: Working data - https://www.kaggle.com/datasets/garystafford/ping-data

6) Climate Analysis: Working data - https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?resource=download%7D
