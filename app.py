from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import cast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = Path("outputs")


@dataclass
class PredictionResult:
    dates: pd.Series
    actual: pd.Series
    predicted: pd.Series
    mae: float
    rmse: float
    r2: float


def load_stock_data(ticker: str, period: str) -> pd.DataFrame:
    raw_data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    data = pd.DataFrame(raw_data)
    if data.empty:
        raise ValueError(f"No data was returned for ticker '{ticker}'.")
    return data


def prepare_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if isinstance(data.columns, pd.MultiIndex):
        if data.columns.nlevels == 2:
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
        else:
            raise ValueError("Unsupported column format returned by data source.")

    required_features = ["Open", "High", "Low", "Volume"]
    missing_columns = [
        column for column in required_features + ["Close"] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    dataset = data.copy()
    dataset["TargetClose"] = dataset["Close"].shift(-1)
    dataset = dataset.dropna(subset=required_features + ["TargetClose"])

    x = cast(pd.DataFrame, dataset[required_features])
    y = cast(pd.Series, dataset["TargetClose"])
    dates = cast(pd.Series, dataset.index.to_series())
    return x, y, dates


def train_and_evaluate(
    x: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_name: str,
    test_ratio: float,
) -> PredictionResult:
    split_index = int(len(x) * (1 - test_ratio))
    if split_index <= 0 or split_index >= len(x):
        raise ValueError(
            "test_ratio produced an invalid train/test split. Try a value such as 0.2."
        )

    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_dates = dates.iloc[split_index:]

    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        raise ValueError("Unsupported model. Use 'linear' or 'random_forest'.")

    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=y_test.index)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return PredictionResult(
        dates=test_dates,
        actual=y_test,
        predicted=predictions,
        mae=mae,
        rmse=rmse,
        r2=r2,
    )


def plot_results(
    ticker: str, result: PredictionResult, output_file: str | None = None
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(result.dates, result.actual, label="Actual Next Close", linewidth=2)
    plt.plot(
        result.dates, result.predicted, label="Predicted Next Close", linestyle="--"
    )
    plt.title(f"{ticker} - Actual vs Predicted Next-Day Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(OUTPUT_DIR / output_file, dpi=150)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict next-day stock close price using regression models."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (e.g., AAPL, TSLA).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Yahoo Finance period string (e.g., 1y, 5y, max).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "random_forest"],
        default="random_forest",
        help="Model to train: linear or random_forest.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data used for testing (chronological split).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional output path for saving the plot image.",
    )
    return parser.parse_args()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    args = parse_args()

    data = load_stock_data(args.ticker, args.period)
    x, y, dates = prepare_dataset(data)
    result = train_and_evaluate(x, y, dates, args.model, args.test_ratio)

    print(f"Ticker: {args.ticker}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(x)}")
    print(f"MAE: {result.mae:.4f}")
    print(f"RMSE: {result.rmse:.4f}")
    print(f"R²: {result.r2:.4f}")

    plot_results(args.ticker, result, args.save_plot)

    if args.save_plot:
        print(f"Plot saved in: {OUTPUT_DIR.resolve()}\\{args.save_plot}")


if __name__ == "__main__":
    main()
