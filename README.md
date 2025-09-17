# Forex Prediction Application

## Overview

This application combines advanced machine learning models with technical analysis to predict forex market movements. The system features multiple prediction models (LSTM, GRU, XGBoost, K-Means clustering) and includes comprehensive trade management tools including risk assessment, stop loss calculation, and position sizing.

## Features

### Prediction Capabilities
- **Multi-model predictions**: LSTM, GRU, and XGBoost models for price forecasting
- **Zone detection**: K-means clustering to identify potential support/resistance zones
- **Technical indicators**: Integration of common technical indicators (EMA, RSI, MACD, etc.)
- **Feature engineering**: Advanced feature creation for improved prediction accuracy

### Trade Management
- **Stop Loss Calculation Methods**:
  - ATR-based (volatility-adjusted)
  - Fixed percentage
  - Support/Resistance levels
- **Risk Management**:
  - Customizable risk percentage per trade (0.5% to 5%)
  - Position size calculator
  - Risk-to-reward ratio optimization

### Backtesting
- Multiple built-in strategies for historical performance analysis
- Detailed statistics including win rate, profit factor, drawdown
- Performance visualization with equity curves
- Strategy optimization tools

### User Interface
- Interactive charts with prediction visualization
- Trade parameter controls
- Exportable trade data (CSV format)
- Educational content on prediction methods and risk management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Forex-Prediction-App.git
cd Forex-Prediction-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage Guide

### Data Import
1. Upload historical forex data in CSV format
2. Select the desired timeframe and currency pair
3. Choose preprocessing options

### Model Selection
1. Navigate to the Prediction page
2. Select the desired prediction model (LSTM, GRU, XGBoost)
3. Adjust model parameters if needed

### Trade Planning
1. Review the prediction results
2. Select a stop loss calculation method
3. Adjust risk parameters
4. View suggested position size and take profit levels

### Backtesting
1. Navigate to the Backtest page
2. Select a trading strategy
3. Configure backtest parameters
4. Run the backtest and analyze results

## Technical Architecture

- **Frontend**: Streamlit (Python-based interactive UI)
- **Machine Learning**: TensorFlow/Keras (LSTM, GRU), XGBoost, scikit-learn (K-means)
- **Technical Analysis**: Custom implementations and TA-Lib
- **Backtesting Engine**: Custom implementation with performance metrics

## Model Information

### LSTM Model
- Optimized for capturing long-term dependencies in time series data
- Features multi-layer architecture with dropout for regularization
- Trained on normalized price data with technical indicators

### GRU Model
- Alternative RNN architecture with fewer parameters than LSTM
- Designed for computational efficiency without sacrificing performance

### XGBoost Model
- Gradient boosting implementation for classification tasks
- Used for trend direction prediction and zone classification

### Zone Detector (K-means)
- Unsupervised learning approach to identify market structure
- Detects potential support/resistance zones for trade planning

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- Streamlit
- Plotly
- TA-Lib

## Future Enhancements

- Integration with live market data
- Automated trading capabilities
- Expanded model selection
- Enhanced visualization options
- Additional risk management tools



## Acknowledgments

- Trading and machine learning community
- Open source libraries that made this project possible
