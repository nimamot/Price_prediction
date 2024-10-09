# 🔍 Bitcoin Price Prediction Project

This project aims to predict the future prices of Bitcoin using machine learning models. Initially, various traditional regression models were tested, including Linear Regression, KNN, Random Forest, and XGBoost. However, due to the sequential nature of the data and temporal dependencies, these models did not perform well. Therefore, the focus shifted to a time-series-based approach using an LSTM (Long Short-Term Memory) model.

## 🚀 Why Time-Lagging Was Used

To better capture the temporal dependencies in the data, lagged features were created, which allowed the model to leverage previous values as predictors for future prices. This step was crucial for improving the model's performance by providing a sense of history in the time-series data.

## 🧠 LSTM Model Overview

An LSTM network was built using:
- 🔄 Two LSTM layers to capture complex time-series patterns.
- ❌ Dropout layers for regularization.
- 🔢 A Dense layer for final price prediction.

The model was evaluated using **Mean Squared Error (MSE)** and visualized using a line plot to compare actual vs. predicted prices. The LSTM successfully captured general trends but struggled with the short-term fluctuations inherent to Bitcoin prices.

## 📊 Results

- **Mean Squared Error (MSE)**: The LSTM model's MSE was lower compared to traditional regression models, demonstrating its effectiveness for this problem.
- The predictions generally followed the trend but underestimated sharp price changes, indicating a need for further tuning or additional features.
![Bitcoin Price Prediction](https://github.com/nimamot/Price_prediction/blob/main/LSTM)


## 🔧 Next Steps

Possible improvements include:
- Adding more lagged features or external variables (e.g., trading volume or sentiment analysis).
- Experimenting with deeper LSTM layers or more advanced time-series models like GRUs.

---

**💡 Note**: The initial models like Linear Regression, KNN, and Random Forest failed to capture the temporal dependencies due to the lack of time-series-specific feature handling, making LSTM a more suitable choice for this project.
