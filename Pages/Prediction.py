import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import AverageTrueRange
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
from keras.models import load_model
import joblib
from keras.metrics import MeanSquaredError
import xgboost
import numpy as np
import matplotlib.pyplot as plt

#================================================================================
# INITIALIZERS
#===============================================================================

def initializer():
    if "df" not in st.session_state: st.session_state.df=None
    if "zone_scale" not in st.session_state: st.session_state.zone_scale=None
    if "zone_model" not in st.session_state: st.session_state.zone_model=None
    if "lstm_model" not in st.session_state: st.session_state.lstm_model=None
    if "gru_model" not in st.session_state: st.session_state.gru_model=None
    if "lstm_f_scale" not in st.session_state: st.session_state.lstm_f_scale=None
    if "lstm_t_scale" not in st.session_state: st.session_state.lstm_t_scale=None
    if "gru_f_scale" not in st.session_state: st.session_state.gru_f_scale=None
    if "gru_t_scale" not in st.session_state: st.session_state.gru_t_scale=None
    if "zone_data" not in st.session_state: st.session_state.zone_data=None
    if "lstm_data" not in st.session_state: st.session_state.lstm_data=None
    if "gru_data" not in st.session_state: st.session_state.gru_data=None

initializer()

#================================================================================
# DAILOG BOX
#===============================================================================

@st.dialog("Overview")
def show_feature():
    st.header("😎 Key Models")
    feature={"Boosting":"Boosting techniques is used including XGBoost LightBoost",
             "RNN":"RNN techniques is used including LSTM(Long-Short term memory and GRU(Gated Recurrent Neural Network))",
             "Stacking":"Stacking Models",
             "ANN":"Artificial Neural Network"}
    for name,val in feature.items():
        st.markdown(f"<p><b>{name}</b> : {val}</p>",unsafe_allow_html=True)


#================================================================================
# DATA LOADING
#===============================================================================

def get_gbpusd_4h(limit=500, period="120d"):
    try:
        df_1h = yf.download("6B=F", interval="1h", period=period, progress=False)
        df_1h.columns = ["close", "high", "low", "open", "volume"]
        df_1h.index.name = "date"
        if df_1h.empty:
            st.error("⚠️ No data received from Yahoo Finance. Server might be busy.")
            return pd.DataFrame()

        df_4h = df_1h.resample("4H").agg({
            "open": "first",  # first value in 4h
            "high": "max",  # highest value in 4h
            "low": "min",  # lowest value in 4h
            "close": "last",  # last value in 4h
            "volume": "sum"  # total volume in 4h
        })
        df_4h.dropna(inplace=True)
        return df_4h

    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return pd.DataFrame()

#================================================================================
# FEATURE MAKING
#===============================================================================

def feature_engineering(data):
    data["ema_20"] = EMAIndicator(data["close"], window=20).ema_indicator()
    data["ema_50"] = EMAIndicator(data["close"], window=50).ema_indicator()
    data["ema_100"] = EMAIndicator(data["close"], window=100).ema_indicator()
    for ema, window in zip([20,20,50,50,50,100], [10,15,20,30,40,80]):
        mean = data[f"ema_{ema}"].rolling(window).mean()
        std = data[f"ema_{ema}"].rolling(window).std()
        data[f"zscore_{ema}_{window}"] = (data[f"ema_{ema}"] - mean) / std
    for i in (5,10, 30, 50):
        data[f"price_score_{i}"] = (data["close"] - data["close"].rolling(i).mean()) / data[
            "close"].rolling(i).std()
        data[f"pct_change_{i}"] = data["close"].pct_change(i)
    atr = AverageTrueRange(high=data["high"], low=data["low"], close=data["close"])
    data["atr"] = atr.average_true_range()
    data["candle_range"] = data["high"] - data["low"]
    data["ema"] = EMAIndicator(data["close"], window=21).ema_indicator()
    data["rsi"] = RSIIndicator(data["close"], window=13).rsi()
    bb = BollingerBands(data["close"])
    data["uperband"] = bb.bollinger_hband()
    data["lowerband"] = bb.bollinger_lband()
    data["bb_avg"] = bb.bollinger_mavg()
    data["mean_price"] = data["close"].rolling(13).mean()
    candle_mean = (data["high"] - data["low"]).mean()
    data["candle_strentgh"] = (data["high"] - data["low"]) / candle_mean

    return data

#================================================================================
# TREND PREDICTOR
#===============================================================================

def trend_predictor(data):
    # Local import ensures `st` name is bound early in the function
    import streamlit as st
    import numpy as np
    import pandas as pd

    # Defensive checks
    zs = st.session_state.get("zone_scale")
    zm = st.session_state.get("zone_model")
    if zs is None or zm is None:
        st.error("Missing session_state['zone_scale'] or session_state['zone_model']. "
                 "Load models/scalers before running trend_predictor.")
        return

    # Ensure data is DataFrame-like and has rows
    if data is None or (hasattr(data, "shape") and data.shape[0] == 0):
        st.warning("No zone data provided to trend_predictor.")
        return

    try:
        features = zs.transform(data)
    except Exception as e:
        st.error(f"zone_scale.transform failed: {e}")
        raise

    try:
        prediction = zm.predict(features)
    except Exception as e:
        st.error(f"zone_model.predict failed: {e}")
        raise

    pred = (
        pd.Series(prediction)
        .map({0: -1, 1: 11, 2: 11, 3: 0, 4: 1, 5: 0})
        .dropna()
        .astype(int)
    )

    if pred.empty:
        st.warning("No valid predictions after mapping.")
        return

    zones = np.array_split(pred, 4)
    label_map = {-1: "Downtrend 📉", 0: "Sideways ➡️", 1: "Uptrend 📈"}
    zone_results = []
    non_empty_zones = 0

    cols = st.columns(4)
    for i, (z, c) in enumerate(zip(zones, cols), start=1):
        counts = z.value_counts()
        total = int(counts.sum())
        if total == 0:
            sig = 0
            conf = 0.0
            votes = {}
        else:
            sig = counts.idxmax()        # -1 / 0 / 1
            conf = counts.max() / total
            votes = counts.to_dict()
            non_empty_zones += 1

        # Normalise numpy scalars to Python types
        if isinstance(sig, (np.generic,)):
            sig = sig.item()

        # Lookup label robustly
        lbl = label_map.get(sig)
        if lbl is None:
            lbl = label_map.get(str(sig).strip().lower(), "UNKNOWN")
            st.warning(f"Unexpected signal value: {repr(sig)} (type={type(sig)}). "
                       f"Known keys: {list(label_map.keys())[:10]}...")

        zone_results.append({
            "zone": i,
            "signal": sig,
            "confidence": conf,
            "label": lbl,
            "votes": votes
        })
        c.metric(f"Zone {i}", lbl, f"{conf*100:.1f}% agreement", help=f"Votes: {votes}")

    if non_empty_zones == 0:
        st.warning("All zones were empty.")
        return

    zone_signals = [r["signal"] for r in zone_results]
    zone_counts = pd.Series(zone_signals).value_counts()
    top_freq = zone_counts.max()
    top_signals = zone_counts[zone_counts == top_freq].index.tolist()

    if len(top_signals) == 1:
        final_signal = top_signals[0]
    else:
        best_sig = None
        best_conf = -1.0
        best_zone_idx = -1
        for idx, r in enumerate(zone_results):  # idx increases with recency
            if r["signal"] in top_signals:
                if (r["confidence"] > best_conf) or (r["confidence"] == best_conf and idx > best_zone_idx):
                    best_sig = r["signal"]
                    best_conf = r["confidence"]
                    best_zone_idx = idx
        final_signal = best_sig

    # Final label lookup (safe)
    final_label = label_map.get(final_signal, "UNKNOWN")
    final_support = zone_counts.get(final_signal, 0) / max(non_empty_zones, 1) * 100.0

    st.metric("🔥🔥Final Trend🔥🔥", final_label, f"{final_support:.0f}% of zones")


#================================================================================
# PRICE PREDICTOR
#===============================================================================

@st._cache_data
def price_predictor(lstm_data,gru_data):
    L=[x for x in range(1,len(lstm_data.columns))]
    G=[x for x in range(1,len(gru_data.columns))]

    features1=lstm_data.values
    features2=gru_data.values

    features1[:,0]=st.session_state.lstm_t_scale.transform(features1[:,0].reshape(-1,1)).ravel()
    features1[:,L]=st.session_state.lstm_f_scale.transform(features1[:,L])
    features2[:,0]=st.session_state.gru_t_scale.transform(features2[:,0].reshape(-1,1)).ravel()
    features2[:,G]=st.session_state.gru_f_scale.transform(features2[:,G])


    features1 = features1.reshape(1,100,len(L)+1)
    features2 = features2.reshape(1,100,len(G)+1)

    prediction_1 = st.session_state.lstm_model.predict(features1, verbose=0)
    prediction_2 = st.session_state.gru_model.predict(features2, verbose=0)

    prediction_1 = st.session_state.lstm_t_scale.inverse_transform(np.asarray(prediction_1).reshape(-1, 1)).ravel()
    prediction_2 = st.session_state.gru_t_scale.inverse_transform(np.asarray(prediction_2).reshape(-1, 1)).ravel()

    return prediction_1, prediction_2

with st.container():
    st.markdown("<h1 style='text-align: center; font-size:4rem'>Predication Zone</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; max-width: 800px; margin: auto;'>In this part"
        " you can try and test multiple backtesting strategies with your own data or we have some samples data as well.</p>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Top buttons ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c3:
        if st.button("🔍 Show Models"):
            show_feature()
    with c5:
        st.caption("Only for 4H GBP/USD")

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)


    st.markdown("<h2>📖 LSTM and GRU </h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Visualization Space")

    st.markdown("""
        **LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)** are types of 
        Recurrent Neural Networks (RNNs) designed to learn **patterns across time**.  

        ---
        ### 🔑 Key Concept: Timesteps
        - A **timestep** is how many past data points the model looks back on.
        - Example: Timestep = 60 → Uses last 60 prices to predict the next.
        - Large timesteps = more context, slower training.
        - Small timesteps = faster, but less historical learning.

        ---
        ### 🌟 Benefits in Stock/Forex Market
        - Capture **sequential trends** (price depends on history).
        - Learn **long-term dependencies** (weeks/months).
        - Handle **noisy data** better than classical models.
        - Useful for:
          - Price prediction
          - Volatility forecasting
          - Trend detection
    """)
    st.info("👉 Below are predictions of LSTM on Forex Data")
    st.caption("Black line is real data | Orange is prediction on training data | Green is prediction on Testing data")
    st.image("Asset/output.png")


    st.markdown("<h2>📖 XGBoost with KMeans for Zone Prediction</h2>", unsafe_allow_html=True)

    st.markdown("""
        ### ⚡ XGBoost (Extreme Gradient Boosting)
        - A **boosting algorithm** that combines many weak learners (decision trees).
        - Great for **tabular and structured financial data**.

        ---
        ### 🔑 Why use KMeans + XGBoost?
        1. **KMeans (Clustering)**  
           - Groups prices into **zones** (support/resistance).  
           - These zones become **labels** for supervised learning.
        2. **XGBoost (Prediction)**  
           - Learns to predict which **zone** new price data belongs to.  
           - Helps identify **trends & market regimes**.

        ---
        ### 🌟 Benefits
        - Zone Prediction → find support & resistance
        - Pattern Learning → from clusters to unseen data
        - Speed & Accuracy → scalable on large datasets
        - Hybrid Power → clustering + prediction
    """)
    st.info("👉 Below are zones made by KMeans and predictions fitted by XGBoost. [0 = Downtrend] [4 = Uptrend]")
    st.caption("Overall Theory is if in range Number of point having Downtrend is more then its down else up or sideways")
    st.image("Asset/zone.png")


    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💻 Model Code Space")

 


    st.markdown("<div class='info-container'></div>", unsafe_allow_html=True)

    st.markdown("<h2>👾 Predict Future Price</h2>", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.info("This Will only predict GBP/USD in 4h timeframe")
    if st.button("🔥Start Predict"):
        # Load the data
        st.session_state.df = get_gbpusd_4h(limit=500, period="120d")

        # Show data only if available
        if st.session_state.df is not None:
            with st.expander("⏰ View Data From Below"):
                st.dataframe(st.session_state.df.tail(10))


    if st.session_state.df is not None:
        st.session_state.df=feature_engineering(st.session_state.df)

        zone_name=['zscore_20_15', 'zscore_50_30', 'zscore_50_40', 'zscore_100_80',
       'price_score_10', 'pct_change_10', 'price_score_30', 'pct_change_30',
       'price_score_50', 'pct_change_50']
        lstm_name=['close', 'volume', 'ema_20', 'ema_50', 'zscore_20_10', 'zscore_50_20',
       'price_score_5', 'pct_change_5', 'price_score_10', 'pct_change_10',
       'atr', 'candle_range']
        gru_name=['close', 'volume', 'ema', 'rsi', 'uperband', 'lowerband', 'bb_avg',
       'mean_price', 'candle_strentgh']

        st.session_state.zone_data=st.session_state.df[zone_name]
        st.session_state.lstm_data=st.session_state.df[lstm_name]
        st.session_state.gru_data=st.session_state.df[gru_name]


        st.session_state.zone_scale=joblib.load("Models/zone_scaler_mm.pkl")
        st.session_state.zone_model=joblib.load("Models/zone_detector.pkl")
        
        # Import needed for custom objects
        import tensorflow as tf
        from tensorflow import keras
        import os

        # Define a more comprehensive set of custom objects
        class CustomInputLayer(keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                if "batch_shape" in config:
                    config["batch_input_shape"] = config.pop("batch_shape")
                return cls(**config)

        # Custom loader function with better error handling
        def load_keras_model_safely(model_path, model_type="LSTM"):
            try:
                # Option 1: Try with custom objects
                model = keras.models.load_model(
                    model_path,
                    custom_objects={
                        "InputLayer": CustomInputLayer,
                        # Add additional custom objects if needed
                        "DTypePolicy": tf.keras.mixed_precision.Policy
                    },
                    compile=False
                )
                return model
            except Exception as e:
                st.error(f"Error loading {model_type} model: {str(e)}")
                st.info(f"Please ensure you have tensorflow==2.13.0 and keras==2.13.1 installed")
                
                # Try one more approach - load with skip_mismatch
                try:
                    st.warning(f"Attempting alternative loading method...")
                    model = tf.keras.models.load_model(
                        model_path, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(
                            experimental_skip_checkpoint=True
                        )
                    )
                    return model
                except Exception as e2:
                    st.error(f"Alternative loading also failed: {str(e2)}")
                    return None

        # Load both models with the safe loader
        try:
            st.session_state.lstm_model = load_keras_model_safely("Models/lstm_model.h5", "LSTM")
            
            # Only compile if model loaded successfully
            if st.session_state.lstm_model is not None:
                st.session_state.lstm_model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])
            
            st.session_state.gru_model = load_keras_model_safely("Models/gru_model.h5", "GRU")
            
            # Only compile if model loaded successfully
            if st.session_state.gru_model is not None:
                st.session_state.gru_model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])
            
            # Check if models loaded before continuing
            if st.session_state.lstm_model is None or st.session_state.gru_model is None:
                st.error("❌ Model loading failed. Cannot continue with predictions.")
                st.stop()  # This stops execution of the rest of the script
                
        except Exception as e:
            st.error(f"❌ Critical error in model loading: {str(e)}")
            st.stop()

        st.session_state.lstm_f_scale=joblib.load("Models/lstm_features.pkl")
        st.session_state.lstm_t_scale=joblib.load("Models/lstm_target.pkl")
        st.session_state.gru_f_scale=joblib.load("Models/gru_features.pkl")
        st.session_state.gru_t_scale=joblib.load("Models/gru_target.pkl")

        st.write("👾 Below is the recent chart of GBPUSD")
        st.caption("This is downloaded from yfinance some delay may present")
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(st.session_state.df["close"], color="green", linewidth=2, label="4Hour")
        ax.legend()
        ax.set_title("GBP/USD Chart")
        st.pyplot(fig)


        #trend predictor
        trend_predictor(st.session_state.zone_data)

        #GRU and LSTM predictor
        d1,d2=price_predictor(st.session_state.lstm_data.tail(100),st.session_state.gru_data.tail(100))

        lstm_df = st.session_state.lstm_data.tail(100)
        gru_df = st.session_state.gru_data.tail(100)

        # Pick a price series (prefer common names, else last numeric col) — inline, no functions

        price_series= lstm_df["close"].astype(float)
        last_price = float(price_series.iloc[-1])

        lstm_pred = float(d1)
        gru_pred = float(d2)+0.012 #Bias Added
        avg_pred = (lstm_pred + gru_pred) / 2.0

        # Deltas vs last
        lstm_delta_pct = (lstm_pred - last_price) / last_price * 100.0
        gru_delta_pct = (gru_pred - last_price) / last_price * 100.0
        avg_delta_pct = (avg_pred - last_price) / last_price * 100.0

        agree = np.sign(lstm_delta_pct) == np.sign(gru_delta_pct)

        # --- Header ---
        st.markdown("""
        <h2 style="text-align:center;margin:0;">📈 Next-Candle Outlook</h2>
        <p style="text-align:center;margin-top:4px;opacity:.75;">LSTM vs GRU with recent market context</p>
        """, unsafe_allow_html=True)

        # --- Highlight badges (agreement + move range) ---
        colA, colB = st.columns([1, 1])
        with colA:
            if agree:
                st.success("✅ Models agree on direction")
            else:
                st.warning("⚠️ Models disagree — use caution")
        with colB:
            band_low, band_high = min(lstm_pred, gru_pred), max(lstm_pred, gru_pred)
            st.info(f"Uncertainty band: {band_low:.5f} → {band_high:.5f}")

        # --- Metrics row ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Last Price", f"{last_price:.5f}")
        with c2:
            st.metric("LSTM → Next", f"{lstm_pred:.5f}", f"{lstm_delta_pct:+.2f}%")
        with c3:
            st.metric("GRU → Next", f"{gru_pred:.5f}", f"{gru_delta_pct:+.2f}%")
        with c4:
            st.metric("Average (LSTM+GRU)", f"{avg_pred:.5f}", f"{avg_delta_pct:+.2f}%")
            
        # --- Stop Loss Section ---
        st.subheader("🛑 Stop Loss Calculation")
        
        # Calculate ATR for dynamic stop loss if available
        atr_value = None
        if 'atr' in lstm_df.columns:
            atr_value = lstm_df['atr'].iloc[-1]
        
        # Default to 1% if ATR not available
        atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.1)
        risk_percent = st.slider("Risk Percentage", 0.5, 5.0, 1.0, 0.1)
        
        # Calculate stop loss levels
        if avg_delta_pct > 0:  # Bullish prediction
            trade_type = "BUY"
            entry_price = last_price
            
            # ATR-based stop loss
            if atr_value:
                atr_stop = entry_price - (atr_value * atr_multiplier)
                atr_stop_pct = (atr_stop - entry_price) / entry_price * 100.0
            else:
                atr_stop = entry_price * (1 - 0.01 * atr_multiplier)
                atr_stop_pct = -1.0 * atr_multiplier
            
            # Percentage-based stop loss
            pct_stop = entry_price * (1 - risk_percent/100)
            pct_stop_pct = -risk_percent
            
            # Support-based stop loss (using recent lows)
            recent_low = lstm_df['close'].tail(20).min()
            support_stop = recent_low
            support_stop_pct = (support_stop - entry_price) / entry_price * 100.0
            
            # Take profit suggestion based on risk:reward of 1:2
            take_profit = entry_price * (1 + (risk_percent * 2)/100)
            take_profit_pct = risk_percent * 2
            
        else:  # Bearish prediction
            trade_type = "SELL"
            entry_price = last_price
            
            # ATR-based stop loss
            if atr_value:
                atr_stop = entry_price + (atr_value * atr_multiplier)
                atr_stop_pct = (atr_stop - entry_price) / entry_price * 100.0
            else:
                atr_stop = entry_price * (1 + 0.01 * atr_multiplier)
                atr_stop_pct = 1.0 * atr_multiplier
            
            # Percentage-based stop loss
            pct_stop = entry_price * (1 + risk_percent/100)
            pct_stop_pct = risk_percent
            
            # Resistance-based stop loss (using recent highs)
            recent_high = lstm_df['close'].tail(20).max()
            support_stop = recent_high
            support_stop_pct = (support_stop - entry_price) / entry_price * 100.0
            
            # Take profit suggestion based on risk:reward of 1:2
            take_profit = entry_price * (1 - (risk_percent * 2)/100)
            take_profit_pct = -risk_percent * 2
        
        # Create columns for displaying stop loss options
        sl_col1, sl_col2, sl_col3 = st.columns(3)
        
        with sl_col1:
            st.info(f"**{trade_type} Signal**")
            st.metric("Entry Price", f"{entry_price:.5f}")
            
        with sl_col2:
            sl_option = st.radio(
                "Stop Loss Method",
                ["ATR-Based", "Fixed Percentage", "Support/Resistance"],
                index=0
            )
            
        with sl_col3:
            if sl_option == "ATR-Based":
                st.metric("Stop Loss", f"{atr_stop:.5f}", f"{atr_stop_pct:.2f}%")
                selected_stop = atr_stop
            elif sl_option == "Fixed Percentage":
                st.metric("Stop Loss", f"{pct_stop:.5f}", f"{pct_stop_pct:.2f}%")
                selected_stop = pct_stop
            else:
                st.metric("Stop Loss", f"{support_stop:.5f}", f"{support_stop_pct:.2f}%")
                selected_stop = support_stop
            
            st.metric("Take Profit (2R)", f"{take_profit:.5f}", f"{take_profit_pct:.2f}%")
        
        # Risk calculation
        position_size_col1, position_size_col2 = st.columns(2)
        
        with position_size_col1:
            account_size = st.number_input("Account Size (USD)", min_value=100.0, value=10000.0, step=100.0)
            risk_amount = account_size * (risk_percent / 100)
            st.metric("Risk Amount", f"${risk_amount:.2f}")
        
        with position_size_col2:
            pip_value = 0.0001 if "USD" in "GBPUSD" else 0.01  # Adjust based on currency pair
            price_diff = abs(entry_price - selected_stop)
            pips_at_risk = price_diff / pip_value
            
        with position_size_col2:
            pip_value = 0.0001 if "USD" in "GBPUSD" else 0.01  # Adjust based on currency pair
            price_diff = abs(entry_price - selected_stop)
            pips_at_risk = price_diff / pip_value
            
            # Position size calculation
            if pips_at_risk > 0:
                position_size = risk_amount / pips_at_risk
                st.metric("Position Size", f"{position_size:.2f} units")
                st.caption(f"Based on {pips_at_risk:.1f} pips at risk")
            else:
                st.warning("Cannot calculate position size: stop loss too close to entry")
        
        with st.expander("📚 About Stop Loss Methods"):
            st.markdown("""
            ### Stop Loss Methods Explained
            
            #### 1. ATR-Based Stop Loss
            - Uses Average True Range to set dynamic stop loss based on market volatility
            - Multiplier adjusts how many ATR units away your stop is placed
            - Higher volatility = wider stop loss to avoid premature exits
            
            #### 2. Fixed Percentage Stop Loss
            - Simple method that sets stop loss at a fixed percentage from entry
            - Example: 1% risk means stop loss is 1% away from entry price
            - Consistent approach regardless of market conditions
            
            #### 3. Support/Resistance Stop Loss
            - Uses recent market structure (highs/lows) to place stop loss
            - For buy trades: Stop goes below recent support
            - For sell trades: Stop goes above recent resistance
            - Often considered more "natural" in terms of market structure
            
            #### Position Sizing
            - Position size is automatically calculated based on:
              - Your account size
              - Your risk percentage
              - Distance to stop loss in pips
            - This ensures consistent risk management across all trades
            """)

        st.divider()

        hist = price_series
        if isinstance(hist.index, pd.DatetimeIndex):
            if len(hist.index) > 1:
                step = hist.index[-1] - hist.index[-2]
            else:
                step = pd.Timedelta(hours=1)
            next_idx = hist.index[-1] + step
        else:
            next_idx = hist.index[-1] + 1 if hasattr(hist.index, "__add__") else len(hist)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History"))

        fig.add_trace(go.Scatter(x=[next_idx], y=[lstm_pred], mode="markers+text",
                                 name="LSTM", text=["LSTM"], textposition="top center"))
        fig.add_trace(go.Scatter(x=[next_idx], y=[gru_pred], mode="markers+text",
                                 name="GRU", text=["GRU"], textposition="bottom center"))
        fig.add_trace(go.Scatter(x=[next_idx], y=[avg_pred], mode="markers+text",
                                 name="Average", text=["AVG"], textposition="middle left"))

        fig.add_shape(type="rect",
                      x0=hist.index[-1], x1=next_idx,
                      y0=min(lstm_pred, gru_pred), y1=max(lstm_pred, gru_pred),
                      fillcolor="LightSkyBlue", opacity=0.22, line_width=0)
        
        # Add stop loss and take profit lines
        if sl_option == "ATR-Based":
            stop_level = atr_stop
        elif sl_option == "Fixed Percentage":
            stop_level = pct_stop
        else:
            stop_level = support_stop
        
        fig.add_shape(type="line",
                     x0=hist.index[-20], x1=next_idx,
                     y0=stop_level, y1=stop_level,
                     line=dict(color="Red", width=2, dash="dash"),
                     name="Stop Loss")
        
        fig.add_shape(type="line",
                     x0=hist.index[-20], x1=next_idx,
                     y0=take_profit, y1=take_profit,
                     line=dict(color="Green", width=2, dash="dash"),
                     name="Take Profit")
        
        # Add annotations for stop loss and take profit
        fig.add_annotation(
            x=hist.index[-5],
            y=stop_level,
            text=f"Stop Loss: {stop_level:.5f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40 if trade_type == "BUY" else 40,
            font=dict(color="Red")
        )
        
        fig.add_annotation(
            x=hist.index[-5],
            y=take_profit,
            text=f"Take Profit: {take_profit:.5f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40 if trade_type == "BUY" else -40,
            font=dict(color="Green")
        )

        fig.update_layout(
            title="Recent Price with Next-Candle Predictions",
            xaxis_title="Time" if isinstance(hist.index, pd.DatetimeIndex) else "Index",
            yaxis_title="Price",
            hovermode="x unified",
            height=460,
            margin=dict(l=30, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Compact summary table ---
        summary_df = pd.DataFrame({
            "Model": ["LSTM", "GRU", "Average", "Entry", "Stop Loss", "Take Profit"],
            "Prediction": [lstm_pred, gru_pred, avg_pred, entry_price, stop_level, take_profit],
            "Δ vs Last": [f"{lstm_pred - last_price:+.6f}", f"{gru_pred - last_price:+.6f}",
                          f"{avg_pred - last_price:+.6f}", "0.000000", 
                          f"{stop_level - entry_price:+.6f}", f"{take_profit - entry_price:+.6f}"],
            "Δ %": [f"{lstm_delta_pct:+.2f}%", f"{gru_delta_pct:+.2f}%", f"{avg_delta_pct:+.2f}%",
                   "0.00%", f"{(stop_level-entry_price)/entry_price*100:+.2f}%", 
                   f"{(take_profit-entry_price)/entry_price*100:+.2f}%"]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # --- Optional: Candlesticks if OHLC exists (kept inline, no functions) ---
        ohlc_cols = {c.lower(): c for c in lstm_df.columns}
        if all(k in ohlc_cols for k in ("open", "high", "low", "close")):
            o, h, l, c = ohlc_cols["open"], ohlc_cols["high"], ohlc_cols["low"], ohlc_cols["close"]
            ohlc = lstm_df[[o, h, l, c]].copy()
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                ohlc.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(ohlc), freq="H")
            fig2 = go.Figure(data=[go.Candlestick(
                x=ohlc.index, open=ohlc[o], high=ohlc[h], low=ohlc[l], close=ohlc[c], name="OHLC"
            )])
            fig2.add_hline(y=lstm_pred, line_dash="dot", annotation_text="LSTM next", opacity=0.5)
            fig2.add_hline(y=gru_pred, line_dash="dot", annotation_text="GRU next", opacity=0.5)
            fig2.add_hline(y=avg_pred, line_dash="dash", annotation_text="AVG next", opacity=0.5)
            fig2.update_layout(title="Candlestick (last 100) + Predicted Levels",
                               height=460, margin=dict(l=30, r=20, t=50, b=40))
            with st.expander("Candlestick view"):
                st.plotly_chart(fig2, use_container_width=True)

        # --- Export for logging/backtest ---
        export_df = pd.DataFrame({
            "last_price": [last_price],
            "lstm_pred": [lstm_pred],
            "gru_pred": [gru_pred],
            "avg_pred": [avg_pred],
            "lstm_delta_pct": [lstm_delta_pct],
            "gru_delta_pct": [gru_delta_pct],
            "avg_delta_pct": [avg_delta_pct],
            "agree": [agree],
            "trade_type": [trade_type],
            "entry_price": [entry_price],
            "stop_loss": [stop_level],
            "take_profit": [take_profit],
            "risk_percent": [risk_percent],
            "position_size": [position_size if 'position_size' in locals() else 0]
        })
        st.download_button("Download trading plan (CSV)", export_df.to_csv(index=False), "trading_plan.csv")






