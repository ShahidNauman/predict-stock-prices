# Predict Future Stock Prices (Short-Term)

## Objective:

Use historical stock data to predict the next day's closing price.

## Dataset:

Stock market data from Yahoo Finance (retrieved using the yfinance Python library)
Instructions:

- [x] Select a stock (e.g., Apple, Tesla).
- [x] Load historical data using the yfinance library.
- [x] Use features like Open, High, Low, and Volume to predict the next Close price.
- [x] Train a Linear Regression or Random Forest model.
- [x] Plot actual vs predicted closing prices for comparison.

## Skills:

- Time series data handling
- Regression modeling
- Data fetching using APIs (yfinance)
- Plotting predictions vs real data

## Python Implementation

This workspace includes `app.py` that fulfills the requirements using Yahoo Finance data.

### Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py --ticker AAPL --model random_forest --period 5y
```

### Useful options

- `--ticker` stock symbol (example: `AAPL`, `TSLA`)
- `--model` one of: `linear`, `random_forest`
- `--period` Yahoo Finance period string (example: `1y`, `5y`, `max`)
- `--test-ratio` proportion for test split (default: `0.2`)
- `--save-plot` save chart to file (example: `plot.png`)

### Example

```bash
python app.py --ticker TSLA --model linear --period 2y --test-ratio 0.25 --save-plot tsla_plot.png
```
