GOLDPREDICT AI
================

Purpose
-------
GoldPredict AI is a Python machine-learning application that uses historical
gold futures prices to predict the next trading day's closing price.

Files
-----
gold_price_predictor.py  Main application
requirements.txt         Python dependencies
GoldPredict_AI_Report.txt Project report
gold_historical_data.csv Dataset created when the application runs
gold_predictions.csv     Test-set predictions created when the application runs
actual_vs_predicted.png  Model-results chart created when the application runs
feature_importance.png   Feature-importance chart created when the application runs

Run Instructions
----------------
1. Install Python 3.10 or newer.
2. Open Terminal in this project folder.
3. Create a virtual environment:
       python -m venv .venv
4. Activate it:
   macOS/Linux:
       source .venv/bin/activate
   Windows:
       .venv\Scripts\activate
5. Install dependencies:
       pip install -r requirements.txt
6. Run the application:
       python gold_price_predictor.py

The application needs an internet connection because it downloads current
historical gold futures data from Yahoo Finance.

Important
---------
The exact dataset size, evaluation metrics, and next-day prediction are
generated when the program is run. Do not manually invent these values for
the final submission; copy the values printed by the program into the report.
