# Gold Price Prediction

A machine learning application built in Python to analyze historical gold market data and predict gold prices using supervised learning.

## Project Overview

This project explores the use of machine learning for commodity price prediction. The application collects and processes historical gold price data, creates predictive features, trains a machine learning model, and evaluates its performance against actual market prices.

The goal of the project was to gain hands-on experience with data preprocessing, feature engineering, machine learning, model evaluation, and data visualization.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Matplotlib
- Machine Learning / Supervised Learning

## Model Performance

The final model achieved:

- **R² Score:** 0.9430
- **Mean Absolute Error (MAE):** $60.22
- **Root Mean Squared Error (RMSE):** $85.59
- **Direction Accuracy:** 48.78%

The R² score indicates that the model explained approximately 94% of the variation in gold prices within the test data.

## Project Files

- `gold_price_predictor.py` - Main Python application and machine learning pipeline
- `gold_historical_data.csv` - Historical market data used by the project
- `gold_predictions.csv` - Actual and predicted gold price results
- `actual_vs_predicted.png` - Visualization comparing actual and predicted prices
- `feature_importance.png` - Visualization of model feature importance
- `requirements.txt` - Python dependencies required to run the project

## What I Learned

This project gave me practical experience working through an end-to-end machine learning workflow, including preparing financial data, engineering features, training a predictive model, evaluating model performance, and presenting results through visualizations.

It also helped me better understand the difference between predicting price values and predicting market direction. While the model achieved a strong R² score for price prediction, its direction accuracy demonstrates the difficulty of forecasting short-term market movements.

## Running the Project

1. Clone or download this repository.
2. Install the required dependencies:

   pip install -r requirements.txt

3. Run the Python application:

   python gold_price_predictor.py

## Disclaimer

This project was created for educational and portfolio purposes. The predictions are not intended to provide financial or investment advice.
