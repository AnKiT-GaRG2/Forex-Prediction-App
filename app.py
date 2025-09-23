import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    layout="wide", 
    page_title="Forex Price Predictor",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Unified Professional Color Theme
COLOR_THEME = {
    'light': {
        'primary': '#2563EB',      # Professional Blue
        'secondary': '#7C3AED',    # Purple Accent
        'accent': '#059669',       # Emerald Green
        'background': '#F8FAFC',   # Light Gray
        'card': '#FFFFFF',         # White
        'text': '#1E293B',         # Dark Gray
        'border': '#E2E8F0',       # Light Border
        'sidebar': '#FFFFFF'       # Sidebar background
    },
    'dark': {
        'primary': '#3B82F6',      # Bright Blue
        'secondary': '#8B5CF6',    # Light Purple
        'accent': '#10B981',       # Bright Green
        'background': '#0F172A',   # Dark Blue
        'card': '#1E293B',         # Dark Card
        'text': '#F1F5F9',         # Light Text
        'border': '#334155',       # Dark Border
        'sidebar': '#1E293B'       # Sidebar background
    }
}

# Get current colors based on dark mode
colors = COLOR_THEME['dark' if st.session_state.dark_mode else 'light']

# Modern CSS with Dark Mode
st.markdown(f"""
<style>
/* ===== FOREX PRICE PREDICTOR - PROFESSIONAL THEME ===== */
:root {{
    /* Color Variables */
    --primary-color: {colors['primary']};
    --secondary-color: {colors['secondary']};
    --accent-color: {colors['accent']};
    --bg-primary: {colors['background']};
    --bg-secondary: {colors['card']};
    --bg-sidebar: {colors['sidebar']};
    --text-primary: {colors['text']};
    --text-secondary: {colors['text']}99;
    --border-color: {colors['border']};
    --gradient-primary: linear-gradient(135deg, {colors['primary']} 0%, {colors['secondary']} 100%);
    --gradient-accent: linear-gradient(135deg, {colors['accent']} 0%, {colors['primary']} 100%);
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    --shadow-hover: 0 20px 40px rgba(0, 0, 0, 0.12);
}}

[data-theme="dark"] {{
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    --shadow-hover: 0 20px 40px rgba(0, 0, 0, 0.4);
}}

/* ===== GLOBAL STYLES ===== */
.stApp {{
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    transition: all 0.3s ease;
}}

/* ===== SIDEBAR STYLING ===== */
section[data-testid="stSidebar"] {{
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-color);
}}

section[data-testid="stSidebar"] .stButton button {{
    width: 100%;
    margin: 5px 0;
    border: 1px solid var(--border-color) !important;
}}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(30px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-50px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

/* ===== TYPOGRAPHY ===== */
h1 {{
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 1rem;
    animation: fadeInUp 0.8s ease-out;
}}

h2 {{
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    text-align: center;
    margin: 2rem 0 1.5rem 0;
    position: relative;
    animation: slideInLeft 0.6s ease-out;
}}

h2:after {{
    content: "";
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 4px;
    background: var(--gradient-primary);
    border-radius: 2px;
}}

h3 {{
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--primary-color) !important;
    margin-bottom: 1rem;
}}

p {{
    line-height: 1.7;
    color: var(--text-primary);
}}

/* ===== CONTAINERS & CARDS ===== */
.main-container, .info-container {{
    background: var(--bg-secondary);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow);
    animation: fadeInUp 0.6s ease-out;
    transition: all 0.3s ease;
}}

.main-container:hover, .info-container:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}}

.info-card {{
    background: var(--bg-secondary);
    border-radius: 16px;
    padding: 2rem;
    border-left: 4px solid var(--primary-color);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    height: 100%;
}}

.info-card:hover {{
    transform: translateY(-5px);
    border-left-color: var(--accent-color);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
}}

.highlight-card {{
    background: var(--gradient-primary);
    color: white !important;
}}

.highlight-card h3, 
.highlight-card p, 
.highlight-card b {{
    color: white !important;
}}

/* ===== BUTTONS ===== */
div[data-testid="stButton"] > button {{
    border-radius: 12px !important;
    border: none !important;
    background: var(--gradient-primary) !important;
    color: white !important;
    transition: all 0.3s ease !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--primary-color)40 !important;
}}

div[data-testid="stButton"] > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px var(--primary-color)60 !important;
}}

/* ===== FORM ELEMENTS ===== */
.stSelectbox > div {{
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-secondary) !important;
    transition: all 0.3s ease;
}}

.stSelectbox > div:hover {{
    border-color: var(--primary-color) !important;
}}

.stNumberInput input {{
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}}

.stNumberInput input:focus {{
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px var(--primary-color)20 !important;
}}

/* ===== METRICS & DATA DISPLAY ===== */
[data-testid="metric-container"] {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}}

.stDataFrame {{
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}}

/* ===== EXPANDER STYLING ===== */
.streamlit-expanderHeader {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    margin: 5px 0 !important;
}}

.streamlit-expanderContent {{
    background: var(--bg-secondary) !important;
    border-radius: 0 0 12px 12px !important;
}}

/* ===== TABS STYLING ===== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
}}

.stTabs [data-baseweb="tab"] {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px 12px 0 0 !important;
    margin: 0 2px !important;
}}

.stTabs [aria-selected="true"] {{
    background: var(--gradient-primary) !important;
    color: white !important;
}}

/* ===== DIALOG STYLING ===== */
[data-testid="stDialog"] {{
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    color: var(--text-primary) !important;
}}

/* ===== RESPONSIVE DESIGN ===== */
@media (max-width: 768px) {{
    h1 {{
        font-size: 2.5rem !important;
    }}
    h2 {{
        font-size: 2rem !important;
    }}
    .main-container, .info-container {{
        padding: 1.5rem;
        margin: 1rem 0;
    }}
}}

/* ===== CUSTOM SCROLLBAR ===== */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: var(--bg-primary);
}}

::-webkit-scrollbar-thumb {{
    background: var(--primary-color);
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: var(--secondary-color);
}}

</style>
""", unsafe_allow_html=True)

# Enhanced company tagline with dynamic CSS
st.markdown(f"""
<style>
.company-tagline-wrapper {{
    margin-top: -8px;
    padding-top: 0;
    z-index: 9999;
    position: relative;
}}

.company-tagline {{
    background: var(--gradient-primary);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin: 0 0 20px 0;
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
    animation: fadeInUp 0.8s ease-out;
}}

.tagline-text {{
    color: white;
    font-size: 2.2rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
}}

.tagline-subtitle {{
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin-top: 8px;
    font-weight: 400;
}}
</style>

<div class="company-tagline-wrapper">
  <div class="company-tagline">
      <h2 class="tagline-text">Driven By Integrity, Powered by Progress</h2>
      <p class="tagline-subtitle">Professional Forex Analysis & Prediction Solutions</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Dark Mode Toggle in Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Dark Mode Toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Navigation")

# Main about page
Main_page = st.Page(
    page="Pages/About.py",
    title="About",
    icon="😉",
    default=True
)

# Backtesting Strategy page
Backtest_page = st.Page(
    page="Pages/Backtest.py",
    title="Backtesting",
    icon="📊"
)

# Future Price prediction
Predication_page = st.Page(
    page="Pages/Prediction.py",
    title="Prediction",
    icon="📈"
)

# Navigation
pg = st.navigation({
    "Info": [Main_page],
    "Models": [Backtest_page, Predication_page]
})

# Logo and branding
st.logo("Asset/Sidebar.png")
st.sidebar.text("Analysis with Ankit!")

try:
    pg.run()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page or check the console for more details.")