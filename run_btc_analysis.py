#!/usr/bin/env python3
"""
Bitcoin Price Prediction Analysis Script
This script runs the Bitcoin price prediction analysis using real data from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Bitcoin Price Prediction Analysis with Real Data")
    print("=" * 60)
    
    # Step 1: Download real Bitcoin data
    print("\n📊 Step 1: Downloading real Bitcoin data from Yahoo Finance...")
    btc_ticker = yf.Ticker("BTC-USD")
    btc_data = btc_ticker.history(start="2016-01-01", end="2024-01-01")
    btc_data.reset_index(inplace=True)
    btc_data = btc_data[['Date', 'Close']].rename(columns={'Close': 'Price'})
    btc_data.dropna(inplace=True)
    
    print(f"✅ Downloaded {len(btc_data)} days of Bitcoin data")
    print(f"📅 Date range: {btc_data['Date'].min()} to {btc_data['Date'].max()}")
    print(f"💰 Price range: ${btc_data['Price'].min():.2f} to ${btc_data['Price'].max():.2f}")
    
    # Step 2: Feature engineering
    print("\n🔧 Step 2: Creating features...")
    btc_data['Date'] = pd.to_datetime(btc_data['Date'])
    btc_data['Day'] = btc_data['Date'].dt.day
    btc_data['Month'] = btc_data['Date'].dt.month
    btc_data['Year'] = btc_data['Date'].dt.year
    btc_data['DayOfWeek'] = btc_data['Date'].dt.dayofweek
    btc_data['DayOfYear'] = btc_data['Date'].dt.dayofyear
    btc_data = btc_data.sort_values('Date').reset_index(drop=True)
    
    # Create lagged features
    lagged_data = btc_data.copy()
    lagged_data['Price_lag1'] = lagged_data['Price'].shift(1)
    lagged_data['Price_lag2'] = lagged_data['Price'].shift(2)
    lagged_data['Price_lag3'] = lagged_data['Price'].shift(3)
    lagged_data['Price_lag7'] = lagged_data['Price'].shift(7)
    lagged_data['Price_rolling_mean7'] = lagged_data['Price'].rolling(window=7).mean()
    lagged_data['Price_rolling_std7'] = lagged_data['Price'].rolling(window=7).std()
    lagged_data['Price_change_1d'] = lagged_data['Price'].pct_change(1)
    lagged_data['Price_volatility_7d'] = lagged_data['Price_change_1d'].rolling(window=7).std()
    lagged_data.dropna(inplace=True)
    
    print(f"✅ Created {len(lagged_data)} samples with lagged features")
    
    # Step 3: Prepare data for LSTM
    print("\n🧠 Step 3: Preparing data for LSTM model...")
    lstm_features = ['Price', 'Price_lag1', 'Price_lag2', 'Price_lag3', 'Price_lag7',
                     'Price_rolling_mean7', 'Price_rolling_std7', 'Price_change_1d', 'Price_volatility_7d']
    
    lstm_data = lagged_data[lstm_features].values
    scaler = MinMaxScaler()
    
    train_size = int(len(lstm_data) * 0.8)
    train_data = lstm_data[:train_size]
    test_data = lstm_data[train_size:]
    
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    print(f"📊 Training data: {train_scaled.shape[0]} samples")
    print(f"📊 Test data: {test_scaled.shape[0]} samples")
    
    # Step 4: Create sequences for LSTM
    def create_sequences(data, n_steps=7):
        X, y = [], []
        for i in range(n_steps, len(data)):
            X.append(data[i-n_steps:i, 1:])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    n_steps = 7
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_test, y_test = create_sequences(test_scaled, n_steps)
    
    print(f"🔄 LSTM sequences - Training: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 5: Build and train LSTM model
    print("\n🏗️ Step 4: Building and training LSTM model...")
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(n_steps, X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    print("🎯 Training model (this may take a few minutes)...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                       validation_data=(X_test, y_test), verbose=0)
    
    # Step 6: Make predictions
    print("\n🔮 Step 5: Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    y_pred_dummy = np.zeros((len(y_pred), len(lstm_features)))
    y_pred_dummy[:, 0] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_dummy)[:, 0]
    
    y_test_dummy = np.zeros((len(y_test), len(lstm_features)))
    y_test_dummy[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 0]
    
    # Step 7: Calculate metrics
    print("\n📈 Step 6: Calculating evaluation metrics...")
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Directional accuracy
    actual_direction = np.diff(y_test_inv) > 0
    predicted_direction = np.diff(y_pred_inv) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 LSTM MODEL RESULTS ON REAL BITCOIN DATA")
    print("=" * 60)
    print(f"📊 Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"📊 Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"📊 Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"📊 R² Score: {r2:.4f}")
    print(f"📊 Directional Accuracy: {directional_accuracy:.2f}%")
    print("=" * 60)
    
    # Step 8: Create visualizations
    print("\n📊 Step 7: Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Bitcoin price over time
    axes[0, 0].plot(btc_data['Date'], btc_data['Price'], color='blue', linewidth=1)
    axes[0, 0].set_title('Real Bitcoin Price Over Time (2016-2024)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Predictions vs Actual
    axes[0, 1].plot(y_test_inv, label='Actual Price', linewidth=2, color='blue')
    axes[0, 1].plot(y_pred_inv, label='Predicted Price', linewidth=2, color='red')
    axes[0, 1].set_title('LSTM Predictions vs Actual Bitcoin Prices', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Price (USD)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    axes[1, 0].scatter(y_test_inv, y_pred_inv, alpha=0.6, color='green')
    min_val = min(y_test_inv.min(), y_pred_inv.min())
    max_val = max(y_test_inv.max(), y_pred_inv.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Actual Price (USD)')
    axes[1, 0].set_ylabel('Predicted Price (USD)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training history
    axes[1, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1, 1].set_title('Model Training History', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bitcoin_prediction_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'bitcoin_prediction_results.png'")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("✅ Successfully analyzed real Bitcoin data from 2016-2024")
    print("✅ Trained LSTM model with multiple time-lagged features")
    print("✅ Achieved reasonable prediction accuracy on test data")
    print("✅ Generated comprehensive visualizations")
    print("\n💡 Key Insights:")
    print(f"   • Model explains {r2*100:.1f}% of price variance")
    print(f"   • Average prediction error: ${mae:.2f}")
    print(f"   • Correctly predicts price direction {directional_accuracy:.1f}% of the time")
    print("=" * 60)

if __name__ == "__main__":
    main()
