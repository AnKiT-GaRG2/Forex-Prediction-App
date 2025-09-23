import streamlit as st

@st.dialog("✨ Key Features")
def show_features_form():
    st.header("✨ Key Features")
    features = {
        "🎯 Price Prediction": "Predict forex prices for major, minor, and exotic pairs.",
        "⚙️ Backtesting": "Test multiple FX strategies (trend-following, mean reversion, breakout).",
        "🌟 Simplicity": "Clean UI with one-click workflows.",
        "🧠 Pre-Trained Models": "Ready-to-use models trained on FX time series.",
        "🛠️ Model Tuning": "Fine-tuned for volatility, sessions, and regime shifts.",
        "📊 Backtest Evaluation": "Evaluate with FX-specific metrics (pips, hit-rate, Sharpe, max drawdown)."
    }
    for feature, desc in features.items():
        st.write(f"**{feature}:** {desc}")

# Main content
with st.container():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-weight: 800;'>Forex Price Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.3rem; max-width: 800px; margin: 0 auto 30px auto;'>Leverage advanced algorithms and robust backtesting to navigate the FX market—across London, New York, and Tokyo sessions—with data-driven confidence.</p>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✨ Explore Features", key="features_btn", use_container_width=True):
            show_features_form()

# --- ABOUT THE MODEL SECTION ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h2>About The Model</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.markdown(
            """
            <div class='info-card'>
            <h3>🧠 Core Architecture</h3>
            <p>We use an ensemble of time-series models tailored for high-frequency FX dynamics:</p>
            <p><b>➡️ Long Short-Term Memory (LSTM) Networks:</b> Capture temporal dependencies, regime shifts, and session-based patterns in currency pairs.</p>
            <p><b>➡️ Gated Recurrent Units (GRUs):</b> Efficient recurrent layers for sequential FX data, helping stabilize training under volatility.</p>
            <p><b>➡️ Feature Engineering:</b> Session flags (London/NY/Tokyo), rolling volatility (ATR), RSI, moving averages, and price action features in pips.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.image("Asset/lstm.png", width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image("Asset/stocks.png", width='stretch')
    with col2:
        st.markdown(
            """
            <div class='info-card highlight-card'>
            <h3>⚙️ Training and Evaluation</h3>
            <p>Models are continuously updated on multi-pair FX data to remain aligned with liquidity cycles and macro events:</p>
            <p><b>➡️ Fine-Tuning:</b> Regular updates with recent tick/1m/5m/1h candles to adapt to volatility regimes.</p>
            <p><b>➡️ Backtesting Engine:</b> Simulated trades on historical FX data with realistic spread, slippage, and session filters.</p>
            <p><b>➡️ Performance Metrics:</b> Pips gained, hit rate, Sharpe ratio, profit factor, and max drawdown.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOREX CONCEPTS EXPLAINED SECTION ---
with st.container():
    st.markdown('<div class="info-container">', unsafe_allow_html=True)
    st.markdown("<h2>Forex Concepts Explained</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown(
            """
            <div class='info-card'>
            <h3>💱 Currency Pairs, Lots, and Pips</h3>
            <p><b>Currency Pairs:</b> FX is quoted as pairs (e.g., EUR/USD, GBP/JPY). The first is the <i>base</i> currency; the second is the <i>quote</i> currency.</p>
            <p><b>Pips:</b> The standard unit of price movement in FX. Strategy returns are often measured in pips.</p>
            <p><b>Lot Sizes:</b> Standard (100k units), Mini (10k), and Micro (1k). Position sizing in lots determines risk per pip.</p>
            <p><b>Leverage & Margin:</b> Brokers allow leverage. Use carefully—pips scale both profit and loss.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.image("Asset/stock_option.png", width='stretch')

    st.markdown('<br>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.image("Asset/bid_ask.png", width='stretch')
    with c2:
        st.markdown(
            """
            <div class='info-card'>
            <h3>📊 Bid, Ask, and Spread</h3>
            <p>Every FX quote has two prices:</p>
            <p><b>Bid Price:</b> The price at which you can sell the base currency.</p>
            <p><b>Ask Price:</b> The price at which you can buy the base currency.</p>
            <p>The difference is the <b>spread</b>, typically measured in pips.</p>
            <p><b>Slippage:</b> The difference between expected and executed price.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-card highlight-card'>
        <h3>🔑 Why These Concepts Matter</h3>
        <p>Mastering FX basics enables robust strategy design and realistic performance expectations:</p>
        <ul style="padding-left: 20px;">
            <li><b>Currency Pairs & Sessions:</b> Liquidity varies by session.</li>
            <li><b>Pips & Position Sizing:</b> Measuring returns in pips keeps drawdowns in check.</li>
            <li><b>Backtesting:</b> Validates rules on historical FX data.</li>
            <li><b>Bid-Ask Spread:</b> A core transaction cost in FX.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)