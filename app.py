# ==============================================================================
# 🌌 QUANTUM CORE V10.5 - UI ULTIMATE EDITION
# Design : Cards, Loaders, Visual Feedback
# ==============================================================================

import streamlit as st
import pandas as pd
import ccxt
import yfinance as yf
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta
import json
import time

# --- 1. CONFIGURATION VISUELLE AVANCÉE ---
st.set_page_config(
    page_title="QUANTUM V10",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# CSS INJECTÉ (Pour le look "App Native")
st.markdown("""
<style>
    /* Fond global */
    .stApp {background-color: #F0F2F6;}
    
    /* Style des Cartes */
    .trade-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 5px solid #BDC3C7;
    }
    
    /* Couleurs sémantiques */
    .card-bullish {border-top-color: #2ECC71;}
    .card-bearish {border-top-color: #E74C3C;}
    .card-neutral {border-top-color: #F1C40F;}
    
    /* Typographie */
    h1 {color: #2C3E50; font-family: 'Helvetica Neue', sans-serif;}
    h2 {font-size: 1.2rem; font-weight: 700; margin-bottom: 5px;}
    .big-price {font-size: 1.8rem; font-weight: 800; color: #34495E;}
    .sub-text {font-size: 0.9rem; color: #7F8C8D;}
    
    /* Boutons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ & CONNEXION ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⛔ ERREUR CRITIQUE : Clé API manquante dans les Secrets.")
    st.stop()

# --- 3. MOTEUR D'ACQUISITION (AVEC FEEDBACK) ---
@st.cache_data(ttl=120, show_spinner=False)
def fetch_data_v10(grid_size):
    try:
        # 1. COINBASE (PRIX)
        exchange = ccxt.coinbase()
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=100)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        current_price = df.iloc[-1]['close']
        
        # 2. PIVOT CALCUL (4H Synthétique)
        df_4h = df.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
        last_ts = df_4h.iloc[-1]['ts']
        # Trouver le dernier Lundi 16h
        days_delta = last_ts.weekday()
        monday = (last_ts - timedelta(days=days_delta)).replace(hour=0, minute=0, second=0)
        pivot_target = monday + timedelta(hours=16)
        
        pivot_row = df_4h[df_4h['ts'] == pivot_target]
        pivot_price = pivot_row.iloc[0]['open'] if not pivot_row.empty else df_4h.iloc[-1]['open']
        
        # 3. MACRO & NEWS
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
            dxy_val = dxy['Close'].iloc[-1]
            dxy_trend = "HAUSSIER 🔴" if dxy_val > dxy['Close'].iloc[0] else "BAISSIER 🟢"
        except: dxy_trend = "N/A"
        
        news = []
        try:
            feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
            if feed.entries: news = [e.title for e in feed.entries[:2]]
        except: news = ["Pas de news"]

        # 4. GRILLE
        dist = (current_price - pivot_price) % grid_size
        if dist > (grid_size/2): dist -= grid_size
        
        return {
            "price": current_price,
            "pivot": pivot_price,
            "dist": dist,
            "dxy": dxy_trend,
            "news": news,
            "vol_spike": df.iloc[-1]['vol'] > df['vol'].mean()
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. CERVEAU IA ---
def ask_guardian(context, tf):
    prompt = f"""
    ROLE: Trader Algorithmique (Quantum V10).
    DATA: {context}
    TIMEFRAME: {tf}
    
    RÈGLES STRICTES:
    - Si TF=1H: Regarde Volatilité.
    - Si TF=4H: Regarde Position vs Pivot.
    - Si TF=DAILY: Regarde Cycle.
    
    OUTPUT JSON SEULEMENT:
    {{
        "signal": "ACHAT" | "VENTE" | "ATTENTE",
        "color": "card-bullish" | "card-bearish" | "card-neutral",
        "entry": "Prix ou 'Market'",
        "stop": "Prix",
        "target": "Prix",
        "reason": "Explication technique brève (max 10 mots)",
        "edu": "Vulgarisation simple (max 15 mots)"
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)
    except:
        return {"signal": "ERREUR", "color": "card-neutral", "reason": "IA indisponible"}

# --- 5. INTERFACE DASHBOARD ---

# Sidebar
with st.sidebar:
    st.title("⚙️ RÉGLAGES")
    grid = st.slider("Grille ($)", 5000, 7000, 5760)
    st.caption("Quantum Core V10.5 Active")
    if st.button("Rafraîchir Données"):
        st.cache_data.clear()
        st.rerun()

# Header
st.title("🛡️ QUANTUM COMMAND")
st.markdown("---")

# Chargement avec Spinner
with st.spinner("📡 SCANNING DES MARCHÉS EN COURS..."):
    data = fetch_data_v10(grid)

if "error" in data:
    st.error(f"ÉCHEC CONNEXION : {data['error']}")
else:
    # BANDEAU MACRO
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BITCOIN", f"${data['price']:,.0f}")
    m2.metric("PIVOT LUNDI", f"${data['pivot']:,.0f}", f"{data['dist']:,.0f} diff")
    m3.metric("DOLLAR (DXY)", data['dxy'])
    m4.caption(f"News: {data['news'][0][:30]}...")

    st.markdown("### ⚡ CARTES DE MISSION")
    
    c1, c2, c3 = st.columns(3)

    # --- 1H ---
    with c1:
        st.markdown("**SCALPING (1H)**")
        if st.button("ANALYSER 1H", key="btn1h", use_container_width=True):
            verdict = ask_guardian(str(data), "1H")
            st.session_state['v1h'] = verdict
        
        if 'v1h' in st.session_state:
            v = st.session_state['v1h']
            st.markdown(f"""
            <div class="trade-card {v['color']}">
                <h2>{v['signal']}</h2>
                <div class="big-price">@{v['entry']}</div>
                <hr>
                <p>🛑 <b>SL:</b> {v['stop']} | 🏁 <b>TP:</b> {v['target']}</p>
                <p class="sub-text">🤖 {v['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Comprendre"):
                st.info(v['edu'])
            if st.button("💾 Journal", key="j1"): st.toast("Trade Archivé !")

    # --- 4H ---
    with c2:
        st.markdown("**TACTIQUE (4H)**")
        if st.button("ANALYSER 4H", key="btn4h", use_container_width=True):
            verdict = ask_guardian(str(data), "4H")
            st.session_state['v4h'] = verdict
            
        if 'v4h' in st.session_state:
            v = st.session_state['v4h']
            st.markdown(f"""
            <div class="trade-card {v['color']}">
                <h2>{v['signal']}</h2>
                <div class="big-price">@{v['entry']}</div>
                <hr>
                <p>🛑 <b>SL:</b> {v['stop']} | 🏁 <b>TP:</b> {v['target']}</p>
                <p class="sub-text">🤖 {v['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Comprendre"):
                st.info(v['edu'])
            if st.button("💾 Journal", key="j2"): st.toast("Trade Archivé !")

    # --- DAILY ---
    with c3:
        st.markdown("**STRATÈGE (DAILY)**")
        st.markdown(f"""
        <div class="trade-card card-neutral">
            <h2>INFO CYCLE</h2>
            <div class="big-price">${data['price']+5760:,.0f}</div>
            <p class="sub-text">Prochaine Grille Majeure</p>
            <hr>
            <p>Cycle Macro : <b>HIVER (Contraction)</b></p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Voir Analyse Globale"):
            st.write("Le cycle de 4 ans impose une prudence vendeuse.")
