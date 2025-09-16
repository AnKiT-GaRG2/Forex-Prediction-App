import streamlit as st
import time

#st.set_page_config(layout="wide", page_title="Forex Price Predictor")

st.markdown(f"""
<style>
    /* --- General Styling --- */
    .stApp {{
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); }}
        50% {{ transform: scale(1.02); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12); }}
        100% {{ transform: scale(1); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); }}
    }}

    /* --- Containers and Cards --- */
    .main-container, .info-container {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-in-out;
    }}

    .info-card {{
        background: #FFFFFF;
        border-radius: 15px;
        padding: 1.8rem;
        transition: all 0.3s ease;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        height: 100%;
    }}
    
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }}
    
    .highlight-card {{
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%);
        color: white !important;
    }}
    
    .highlight-card h3, .highlight-card p, .highlight-card b {{
        color: white !important;
    }}

    /* --- Typography --- */
    h1 {{
        background: linear-gradient(90deg, #3B82F6, #1E40AF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }}
    
    h2 {{
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 3.5rem; /* Increased from 2.5rem to 3.5rem to create more space */
        color: #1E3A8A !important;
        position: relative;
    }}
    
    h2:after {{
        content: "";
        position: absolute;
        bottom: -20px; /* Changed from -10px to -20px to position the line further down */
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6, #1E40AF);
        border-radius: 2px;
    }}
    
    h3 {{
        font-size: 1.6rem;
        font-weight: 600;
        color: #3B82F6 !important;
        margin-bottom: 1.2rem;
    }}

    p {{
        line-height: 1.7;
    }}

    /* --- Button Styling --- */
    div[data-testid="stButton"] > button {{
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%);
        color: white;
        transition: all 0.3s ease;
        padding: 12px 32px;
        font-weight: 600;
        display: block;
        margin: 0 auto;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }}
    
    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
    }}

    /* Dialog styling */
    [data-testid="stDialog"] {{
        color: white;
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2) !important;
    }}

    /* Accent elements */
    .accent-text {{
        color: #3B82F6;
        font-weight: 600;
    }}
    
    .feature-icon {{
        font-size: 1.8rem;
        margin-right: 10px;
        color: #3B82F6;
    }}
    
    /* Image styling */
    img {{
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    img:hover {{
        transform: scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    }}

</style>
""", unsafe_allow_html=True)


@st.dialog("Overview")
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
    st.markdown("<br>", unsafe_allow_html=True)
    for feature, desc in features.items():
        st.write(f"**{feature}:** {desc}")


with st.container():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-weight: 800;'>Forex Price Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.3rem; max-width: 800px; margin: 0 auto 30px auto; color: #4B5563;'>Leverage advanced algorithms and robust backtesting to navigate the FX market—across London, New York, and Tokyo sessions—with data-driven confidence.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✨ Explore Features"):
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
        st.image("Asset/lstm.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image("Asset/stocks.png", use_container_width=True)
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
            <p><b>Currency Pairs:</b> FX is quoted as pairs (e.g., EUR/USD, GBP/JPY). The first is the <i>base</i> currency; the second is the <i>quote</i> currency.
            There are many currency pairs available in the market but the majority of trades take place in 4 major currency pairs.</p>
            <p><b>Pips:</b> The standard unit of price movement in FX (typically the 4th decimal place, 2nd for JPY pairs). Strategy returns are often measured in pips.
            Each pip movement will earn around 10 points. Means if you chose 1 lot and price moves by 1 pip your price fluctuation will be 
            1*1*10 = 10 dollars.</p>
            <p><b>Lot Sizes:</b> Standard (100k units), Mini (10k), and Micro (1k). Position sizing in lots determines risk per pip.
            You can select lot in decimal places even.</p>
            <p><b>Leverage & Margin:</b> Brokers allow leverage (e.g., 1:30, 1:100). Use carefully—pips scale both profit and loss.
            It plays a crucial role in maximizing profits if handled carefully, otherwise the account will be blown up.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
        st.image("Asset/stock_option.png", use_container_width=True)

    st.markdown('<br>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
        st.image("Asset/bid_ask.png", use_container_width=True)
    with c2:
        st.markdown(
            """
            <div class='info-card'>
            <h3>📊 Bid, Ask, and Spread</h3>
            <p>Every FX quote has two prices:</p>
            <p><b>Bid Price:</b> The price at which you can sell the base currency.
            Like price of currency is 100 but you want to sell it at 102, that is the bid price.</p>
            <p><b>Ask Price:</b> The price at which you can buy the base currency.
            Like price of currency is 100 but you want to buy it at 98, that is ask price.</p>
            <p>The difference is the <b>spread</b>, typically measured in pips. Lower spreads (e.g., EUR/USD during London/NY overlap) imply better liquidity and lower trading costs.</p>
            <p><b>Slippage:</b> The difference between expected and executed price—more common during news or low-liquidity hours.
            So the difference is 102-98 = 4, this is our spread.</p>
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
            <li><b>Currency Pairs & Sessions:</b> Liquidity varies by session (Tokyo, London, New York). Your strategy should account for time-of-day effects.</li>
            <li><b>Pips & Position Sizing:</b> Measuring returns in pips and sizing positions by risk per trade keeps drawdowns in check.</li>
            <li><b>Backtesting:</b> Validates rules on historical FX data including spread and slippage—critical before going live.</li>
            <li><b>Bid-Ask Spread:</b> A core transaction cost in FX; tighter spreads can significantly improve strategy efficiency.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
