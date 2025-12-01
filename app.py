# ==============================================================================
# 🌌 QUANTUM CORE V10.4 - STABLE COMMAND CENTER (PATCH "TYPE CLEANING")
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import feedparser
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="QUANTUM CORE V10", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp {background-color: #F8F9FA;}
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #BDC3C7;
    }
    .bullish {border-left-color: #27AE60 !important;}
    .bearish {border-left-color: #C0392B !important;}
    .neutral {border-left-color: #F39C12 !important;}
    h1, h2, h3 {color: #2C3E50;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except:
    st.error("⚠️ CLÉS API MANQUANTES DANS LES SECRETS.")
    st.stop()

# --- 3. MOTEUR PHYSIQUE (NETTOYÉ) ---
@st.cache_data(ttl=300)
def fetch_market_physics(grid_size, atom_size):
    # A. MACRO
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        dxy_trend = "HAUSSIER (Risk Off)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "BAISSIER (Risk On)"
    except: dxy_trend = "Inconnu"

    # B. NEWS
    news_titles = []
    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        for entry in feed.entries[:3]: news_titles.append(entry.title)
    except: news_titles = ["Flux indisponible"]

    # C. CRYPTO
    exchange = ccxt.coinbase()
    try:
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=300)
        df_1h = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], unit='ms')
    except: return None

    # Calculs
    current_price = df_1h.iloc[-1]['close']
    
    # Pivot 4H
    df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first','close':'last'}).dropna().reset_index()
    last_ts = df_4h.iloc[-1]['ts']
    monday_date = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0,minute=0,second=0)
    pivot_row = df_4h[df_4h['ts'] == (monday_date + timedelta(hours=16))]
    pivot_price = pivot_row.iloc[0]['open'] if not pivot_row.empty else df_4h.iloc[-1]['open']
    
    # Grille
    dist = (current_price - pivot_price) % grid_size
    if dist > (grid_size/2): dist -= grid_size
    nearest_level = current_price - dist
    
    # Structure
    df_wk = df_1h.set_index('ts').resample('W-MON').agg({'close':'last'}).dropna()
    ma20_wk = df_wk['close'].rolling(20).mean().iloc[-1] if len(df_wk)>20 else current_price
    wk_trend = "BAISSIER" if current_price < ma20_wk else "HAUSSIER"

    # Logique Booléenne
    is_thu = (datetime.now().weekday() == 3) and (datetime.now().hour >= 16)
    vol_sp = df_1h.iloc[-1]['vol'] > (df_1h['vol'].rolling(20).mean().iloc[-1] * 1.5)

    # --- LE CORRECTIF JSON (CASTING EXPLICITE) ---
    return {
        "price": float(current_price),
        "pivot": float(pivot_price),
        "grid_dist": float(dist),
        "nearest": float(nearest_level),
        "wk_trend": str(wk_trend),
        "dxy": str(dxy_trend),
        "news": news_titles,
        "is_thursday": bool(is_thu), # Force Python Bool
        "vol_spike": bool(vol_sp)    # Force Python Bool
    }

# --- 4. CERVEAU ---
def get_ai_verdict(data, timeframe):
    system_prompt = f"""
    TU ES LE GARDIEN V10.4. Analyse ce marché ({timeframe}).
    INPUT: {json.dumps(data)}
    RÈGLES:
    - Weekly BAISSIER -> Pas d'achat swing.
    - Prix < Pivot -> Biais Vendeur.
    FORMAT JSON STRICT:
    {{
        "action": "ACHETER" | "VENDRE" | "ATTENDRE",
        "confiance": "ÉLEVÉE" | "MOYENNE",
        "stop_loss": 00000,
        "target": 00000,
        "raison_technique": "Phrase courte",
        "vulgarisation": "Explication simple"
    }}
    """
    # Utilisation modèle v2.0-flash (Gratuit et Rapide)
    model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system_prompt)
    try:
        resp = model.generate_content("ANALYSE", generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)
    except:
        return {"action": "ERREUR", "raison_technique": "IA Indisponible", "vulgarisation": "Réessayez"}

# --- 5. UI ---
with st.sidebar:
    st.header("⚙️ PARAMÈTRES")
    grid_setting = st.slider("Grille ($)", 5000, 7000, 5760)
    if st.button("Recharger Données"): st.cache_data.clear()

st.title("🛡️ QUANTUM COMMAND CENTER")
data = fetch_market_physics(grid_setting, 720)

if data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix BTC", f"${data['price']:,.0f}", f"{data['grid_dist']:,.0f} vs Grille")
    c2.metric("Tendance Hebdo", data['wk_trend'])
    c3.metric("Macro", data['dxy'])
    c4.metric("Pivot Lundi", f"${data['pivot']:,.0f}")
    st.markdown("---")

    col_1h, col_4h, col_d = st.columns(3)

    with col_1h:
        st.subheader("⚡ 1H • CINÉTIQUE")
        if st.button("SCAN 1H", use_container_width=True):
            verdict = get_ai_verdict(data, "1H")
            color = "bullish" if verdict['action']=="ACHETER" else "bearish" if verdict['action']=="VENDRE" else "neutral"
            st.markdown(f'<div class="metric-card {color}"><h2>{verdict["action"]}</h2></div>', unsafe_allow_html=True)
            with st.expander("DÉTAILS"):
                st.write(f"**STOP:** ${verdict['stop_loss']}")
                st.write(f"**CIBLE:** ${verdict['target']}")
                st.info(verdict['raison_technique'])

    with col_4h:
        st.subheader("⚔️ 4H • TACTIQUE")
        if st.button("SCAN 4H", use_container_width=True):
            verdict = get_ai_verdict(data, "4H")
            color = "bullish" if verdict['action']=="ACHETER" else "bearish" if verdict['action']=="VENDRE" else "neutral"
            st.markdown(f'<div class="metric-card {color}"><h2>{verdict["action"]}</h2></div>', unsafe_allow_html=True)
            with st.expander("PLAN"):
                st.write(f"**STOP:** ${verdict['stop_loss']}")
                st.write(f"**CIBLE:** ${verdict['target']}")
                st.info(verdict['raison_technique'])

    with col_d:
        st.subheader("🏛️ DAILY • STRATÈGE")
        st.markdown(f"**Prochaine Grille:** ${data['nearest']:,.0f}")
        st.progress(max(0, min(100, int(50 + (data['grid_dist']/grid_setting)*100))))
        with st.expander("INFOS"):
            st.write(f"News: {data['news'][0]}")

else: st.error("Erreur Connexion Marchés")
