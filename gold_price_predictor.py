"""
GoldPredict AI
Machine Learning Application for Gold Price Prediction

This application downloads historical gold futures data and uses
machine learning to:

1. Predict the next trading day's gold closing price.
2. Test whether a separate classification model can predict whether
   the next trading day's price will move up or down.

Models:
- Random Forest Regressor
- Random Forest Classifier
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TICKER = "GC=F"
PERIOD = "5y"
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 500

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "gold_historical_data.csv"
PREDICTIONS_FILE = BASE_DIR / "gold_predictions.csv"
ACTUAL_PREDICTED_PLOT = BASE_DIR / "actual_vs_predicted.png"
FEATURE_IMPORTANCE_PLOT = BASE_DIR / "feature_importance.png"


FEATURES = [
    "Return_1d",
    "Return_2d",
    "Return_5d",
    "MA_5_Ratio",
    "MA_10_Ratio",
    "MA_20_Ratio",
    "Volatility_5d",
    "Volatility_20d",
    "High_Low_Range",
    "Open_Close_Change",
    "Volume_Change"
]


# --------------------------------------------------
# Load historical data
# --------------------------------------------------

def load_data():

    import yfinance as yf

    print("Downloading historical gold data...")

    data = yf.download(
        TICKER,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "No historical gold data was downloaded."
        )

    # yfinance may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[required_columns].copy()

    data.index = pd.to_datetime(data.index)

    data.sort_index(inplace=True)

    for column in required_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data.dropna(inplace=True)

    # Save the downloaded dataset.
    data.to_csv(DATA_FILE)

    print(
        f"Downloaded {len(data):,} daily records."
    )

    print(
        f"Date range: "
        f"{data.index.min().date()} "
        f"to {data.index.max().date()}"
    )

    return data


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def create_features(data):

    df = data.copy()

    # One-day return.
    df["Return_1d"] = (
        df["Close"].pct_change()
    )

    # Two-day and five-day returns.
    df["Return_2d"] = (
        df["Close"].pct_change(2)
    )

    df["Return_5d"] = (
        df["Close"].pct_change(5)
    )

    # Moving averages.
    df["MA_5"] = (
        df["Close"]
        .rolling(5)
        .mean()
    )

    df["MA_10"] = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    df["MA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    # Measure current price relative to moving averages.
    df["MA_5_Ratio"] = (
        df["Close"] / df["MA_5"] - 1
    )

    df["MA_10_Ratio"] = (
        df["Close"] / df["MA_10"] - 1
    )

    df["MA_20_Ratio"] = (
        df["Close"] / df["MA_20"] - 1
    )

    # Short-term and longer-term volatility.
    df["Volatility_5d"] = (
        df["Return_1d"]
        .rolling(5)
        .std()
    )

    df["Volatility_20d"] = (
        df["Return_1d"]
        .rolling(20)
        .std()
    )

    # Intraday high-to-low price range.
    df["High_Low_Range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    # Change between opening and closing price.
    df["Open_Close_Change"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    # Daily change in trading volume.
    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Regression target:
    # percentage return on the next trading day.
    df["Next_Day_Return"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    )

    # Classification target:
    # 1 = next day moves up
    # 0 = next day stays the same or moves down
    df["Next_Day_Direction"] = np.where(
        df["Next_Day_Return"].notna(),
        (df["Next_Day_Return"] > 0).astype(int),
        np.nan
    )

    # Actual next-day closing price.
    # This is used for model evaluation.
    df["Next_Day_Close"] = (
        df["Close"].shift(-1)
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Keep the newest row even though its future
    # target values are not known yet.
    df.dropna(
        subset=FEATURES,
        inplace=True
    )

    return df


# --------------------------------------------------
# Chronological train/test split
# --------------------------------------------------

def split_data(df):

    # Only known historical outcomes can be used
    # for training and testing.
    model_df = df.dropna(
        subset=[
            "Next_Day_Return",
            "Next_Day_Close",
            "Next_Day_Direction"
        ]
    ).copy()

    split_index = int(
        len(model_df)
        * (1 - TEST_SIZE)
    )

    # Earlier observations are training data.
    train = (
        model_df
        .iloc[:split_index]
        .copy()
    )

    # Most recent observations are test data.
    test = (
        model_df
        .iloc[split_index:]
        .copy()
    )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train["Next_Day_Return"]
    y_test = test["Next_Day_Return"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train,
        test
    )


# --------------------------------------------------
# Train price regression model
# --------------------------------------------------

def train_model(
    X_train,
    y_train
):

    print(
        "\nTraining Random Forest regression model..."
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=3
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# --------------------------------------------------
# Train direction classification model
# --------------------------------------------------

def train_direction_classifier(
    X_train,
    train
):

    print(
        "Training Random Forest "
        "direction classifier..."
    )

    y_direction = (
        train["Next_Day_Direction"]
        .astype(int)
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced"
    )

    classifier.fit(
        X_train,
        y_direction
    )

    return classifier


# --------------------------------------------------
# Evaluate price regression model
# --------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    test
):

    # Predict the next day's percentage return.
    predicted_returns = (
        model.predict(X_test)
    )

    # Convert the predicted return back
    # into a predicted dollar price.
    predicted_prices = (
        test["Close"].values
        * (1 + predicted_returns)
    )

    actual_prices = (
        test["Next_Day_Close"].values
    )

    # Calculate regression metrics.
    mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            predicted_prices
        )
    )

    r2 = r2_score(
        actual_prices,
        predicted_prices
    )

    # Direction implied by regression predictions.
    actual_direction = (
        y_test.values > 0
    )

    predicted_direction = (
        predicted_returns > 0
    )

    regression_direction_accuracy = np.mean(
        actual_direction
        == predicted_direction
    )

    # Save predictions for later review.
    results = pd.DataFrame(
        {
            "Current_Close":
                test["Close"].values,

            "Actual_Next_Close":
                actual_prices,

            "Predicted_Next_Close":
                predicted_prices,

            "Actual_Return":
                y_test.values,

            "Predicted_Return":
                predicted_returns,

            "Absolute_Error":
                np.abs(
                    actual_prices
                    - predicted_prices
                )
        },
        index=test.index
    )

    results.to_csv(
        PREDICTIONS_FILE
    )

    print(
        "\n--- Price Regression Results ---"
    )

    print(
        f"MAE: ${mae:,.2f}"
    )

    print(
        f"RMSE: ${rmse:,.2f}"
    )

    print(
        f"R²: {r2:.4f}"
    )

    print(
        "Regression Direction Accuracy: "
        f"{regression_direction_accuracy:.2%}"
    )

    return (
        predicted_prices,
        mae,
        rmse,
        r2,
        regression_direction_accuracy
    )


# --------------------------------------------------
# Evaluate direction classifier
# --------------------------------------------------

def evaluate_direction_classifier(
    classifier,
    X_test,
    test
):

    actual_direction = (
        test["Next_Day_Direction"]
        .astype(int)
    )

    predicted_direction = (
        classifier.predict(X_test)
    )

    accuracy = accuracy_score(
        actual_direction,
        predicted_direction
    )

    # Create a simple baseline that always
    # predicts the most common direction
    # in the test data.
    majority_class = (
        actual_direction
        .value_counts()
        .idxmax()
    )

    baseline_predictions = np.full(
        len(actual_direction),
        majority_class
    )

    baseline_accuracy = accuracy_score(
        actual_direction,
        baseline_predictions
    )

    print(
        "\n--- Direction Classification Results ---"
    )

    print(
        "Classifier Direction Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        "Majority-Class Baseline Accuracy: "
        f"{baseline_accuracy:.2%}"
    )

    if accuracy > baseline_accuracy:

        print(
            "Classifier beat the "
            "majority-class baseline."
        )

    elif accuracy < baseline_accuracy:

        print(
            "Classifier did not beat the "
            "majority-class baseline."
        )

    else:

        print(
            "Classifier matched the "
            "majority-class baseline."
        )

    return (
        accuracy,
        baseline_accuracy
    )


# --------------------------------------------------
# Predict next trading day
# --------------------------------------------------

def predict_next_day(
    model,
    classifier,
    df
):

    # The newest row contains all required
    # input features but does not yet have
    # a known next-day outcome.
    latest_features = (
        df[FEATURES]
        .iloc[[-1]]
    )

    predicted_return = (
        model.predict(
            latest_features
        )[0]
    )

    latest_close = (
        df["Close"].iloc[-1]
    )

    predicted_price = (
        latest_close
        * (1 + predicted_return)
    )

    latest_date = (
        df.index[-1]
    )

    # Direction predicted by the separate classifier.
    direction_prediction = int(
        classifier.predict(
            latest_features
        )[0]
    )

    direction_probability = (
        classifier.predict_proba(
            latest_features
        )[0]
    )

    class_list = list(
        classifier.classes_
    )

    up_index = class_list.index(1)

    probability_up = (
        direction_probability[up_index]
    )

    if direction_prediction == 1:
        direction_text = "UP"
    else:
        direction_text = "DOWN"

    print(
        "\n--- Next Trading Day Prediction ---"
    )

    print(
        "Latest available trading date: "
        f"{latest_date.date()}"
    )

    print(
        "Latest closing price: "
        f"${latest_close:,.2f}"
    )

    print(
        "Predicted next-day return: "
        f"{predicted_return:.3%}"
    )

    print(
        "Predicted next closing price: "
        f"${predicted_price:,.2f}"
    )

    print(
        "Classifier predicted direction: "
        f"{direction_text}"
    )

    print(
        "Classifier probability of UP: "
        f"{probability_up:.2%}"
    )

    return predicted_price


# --------------------------------------------------
# Create visualizations
# --------------------------------------------------

def create_visualizations(
    test,
    predicted_prices,
    model
):

    # Actual vs. predicted prices.
    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test.index,
        test["Next_Day_Close"],
        label="Actual Price"
    )

    plt.plot(
        test.index,
        predicted_prices,
        label="Predicted Price"
    )

    plt.title(
        "Gold Price: Actual vs. Predicted"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Gold Price (USD)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACTUAL_PREDICTED_PLOT,
        dpi=150
    )

    plt.close()

    # Regression feature importance.
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    )

    importance = (
        importance
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    importance.plot(
        kind="barh"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved chart: "
        f"{ACTUAL_PREDICTED_PLOT.name}"
    )

    print(
        f"Saved chart: "
        f"{FEATURE_IMPORTANCE_PLOT.name}"
    )


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():

    print("=" * 60)

    print(
        "GoldPredict AI - "
        "Gold Price Prediction Application"
    )

    print("=" * 60)

    # Download historical gold data.
    data = load_data()

    # Create machine-learning features.
    df = create_features(data)

    # Count historical records with known outcomes.
    modeling_records = df.dropna(
        subset=[
            "Next_Day_Return",
            "Next_Day_Close",
            "Next_Day_Direction"
        ]
    )

    print(
        "\nModeling records after "
        f"feature engineering: "
        f"{len(modeling_records):,}"
    )

    print(
        "Latest feature row available "
        f"for future prediction: "
        f"{df.index[-1].date()}"
    )

    print(
        f"Number of input features: "
        f"{len(FEATURES)}"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train,
        test
    ) = split_data(df)

    print(
        f"Training records: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing records: "
        f"{len(X_test):,}"
    )

    # Train the regression model.
    model = train_model(
        X_train,
        y_train
    )

    # Train the separate direction classifier.
    classifier = train_direction_classifier(
        X_train,
        train
    )

    # Evaluate price prediction.
    (
        predicted_prices,
        mae,
        rmse,
        r2,
        regression_direction_accuracy
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        test
    )

    # Evaluate direction prediction.
    (
        classifier_accuracy,
        baseline_accuracy
    ) = evaluate_direction_classifier(
        classifier,
        X_test,
        test
    )

    # Generate a prediction using the newest
    # available gold market data.
    predict_next_day(
        model,
        classifier,
        df
    )

    # Generate output charts.
    create_visualizations(
        test,
        predicted_prices,
        model
    )

    print(
        "\nApplication complete."
    )

    print(
        "\nFiles created:"
    )

    print(
        f"  - {DATA_FILE.name}"
    )

    print(
        f"  - {PREDICTIONS_FILE.name}"
    )

    print(
        f"  - {ACTUAL_PREDICTED_PLOT.name}"
    )

    print(
        f"  - {FEATURE_IMPORTANCE_PLOT.name}"
    )


if __name__ == "__main__":
    main()