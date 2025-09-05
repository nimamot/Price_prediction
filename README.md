# 🔍 Bitcoin Price Prediction Project

This project aims to predict the future prices of Bitcoin using machine learning models with **real Bitcoin data** from Yahoo Finance (2016-2024). Initially, various traditional regression models were tested, including Linear Regression, KNN, Random Forest, and XGBoost. However, due to the sequential nature of the data and temporal dependencies, these models did not perform well. Therefore, the focus shifted to a time-series-based approach using an LSTM (Long Short-Term Memory) model.

## 🚀 Why Time-Lagging Was Used

To better capture the temporal dependencies in the data, lagged features were created, which allowed the model to leverage previous values as predictors for future prices. This step was crucial for improving the model's performance by providing a sense of history in the time-series data.

## 🧠 LSTM Model Overview

An enhanced LSTM network was built using:
- 🔄 Two LSTM layers (100 and 50 neurons) to capture complex time-series patterns
- ❌ Dropout layers (0.3) for regularization
- 🔢 Dense layers with ReLU activation for feature extraction
- 📊 Multiple time-lagged features (1, 2, 3, 7-day lags)
- 📈 Rolling statistics (7-day mean and standard deviation)
- 📉 Price change and volatility features

The model was evaluated using comprehensive metrics including **MSE, RMSE, MAE, MAPE, R² Score, and Directional Accuracy**. The LSTM successfully captured general trends and shows improved performance on real Bitcoin market data.

## 📊 Results

The LSTM model demonstrates strong performance on real Bitcoin data:
- **R² Score**: High variance explanation indicating good model fit
- **Directional Accuracy**: Measures how often the model correctly predicts price direction
- **MAPE**: Percentage-based error measurement for relative accuracy
- **RMSE/MAE**: Absolute error metrics in USD terms
- The model successfully captures Bitcoin's high volatility and long-term trends
- Predictions show good correlation with actual price movements

![Bitcoin Price Prediction](https://github.com/nimamot/Price_prediction/blob/main/LSTM)


## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv btc_env
   source btc_env/bin/activate  # On Windows: btc_env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost yfinance matplotlib seaborn tensorflow jupyter
   ```

### Running the Analysis
- **Jupyter Notebook**: `jupyter notebook btc.ipynb`
- **Python Script**: `python run_btc_analysis.py`

## 🔧 Next Steps

Possible improvements include:
- Adding more lagged features or external variables (e.g., trading volume, sentiment analysis, macroeconomic indicators)
- Experimenting with deeper LSTM layers or more advanced time-series models like GRUs or Transformers
- Implementing ensemble methods combining multiple models
- Adding technical indicators (RSI, MACD, Bollinger Bands)
- Incorporating market sentiment data from social media

## 📁 Project Structure
- `btc.ipynb` - Main Jupyter notebook with complete analysis
- `run_btc_analysis.py` - Standalone Python script for running the analysis
- `btc_env/` - Virtual environment directory
- `bitcoin_prediction_results.png` - Generated visualization results

---

**💡 Note**: The initial models like Linear Regression, KNN, and Random Forest failed to capture the temporal dependencies due to the lack of time-series-specific feature handling, making LSTM a more suitable choice for this project. The current implementation uses real Bitcoin data from Yahoo Finance, providing more realistic and meaningful predictions.
