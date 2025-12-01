# ==============================================================================
# 🌌 QUANTUM CORE V13.1 - STABLE PATCH (YFINANCE FIX)
# Correction : Aplatissement des MultiIndex Yahoo pour éviter l'erreur "Ambiguous"
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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="QUANTUM V13", layout="wide", initial_sidebar_state="expanded", page_icon="🛡️")

st.markdown("""
<style>
    .stApp {background-color: #0E1117; color: #E0E0E0;}
    .report-box {background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #3498DB; margin-bottom: 20px;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 300;}
    .stTabs [data-baseweb="tab"] {background-color: #262730; color: white;}
    .stTabs [aria-selected="true"] {background-color: #3498DB;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ CLÉ API MANQUANTE")
    st.stop()

# --- 3. MOTEUR PHYSIQUE ROBUSTE (PATCHÉ) ---
@st.cache_data(ttl=300)
def fetch_fractal_physics(grid_size):
    try:
        # A. STRUCTURE LONG TERME (Yahoo - Patch MultiIndex)
        df_macro = yf.download("BTC-USD", period="2y", interval="1wk", progress=False)
        
        # --- CORRECTIF V13.1 : APLATISSEMENT DES COLONNES ---
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro.columns = df_macro.columns.get_level_values(0)
        # ----------------------------------------------------

        if not df_macro.empty:
            df_macro['MA20'] = df_macro['Close'].rolling(window=20).mean()
            # Conversion explicite en float pour éviter l'erreur "Ambiguous"
            ma20_val = float(df_macro['MA20'].iloc[-1])
            last_close = float(df_macro['Close'].iloc[-1])
            cycle_status = "HIVER (BAISSIER)" if last_close < ma20_val else "ÉTÉ (HAUSSIER)"
        else:
            cycle_status = "INCONNU"
            ma20_val = 0.0

        # B. PRÉCISION LIVE (Coinbase)
        exchange = ccxt.coinbase()
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=500)
        df_1h = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], unit='ms')
        current_price = float(df_1h.iloc[-1]['close'])
        
        # C. PIVOT TACTIQUE
        df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
        last_ts = df_4h.iloc[-1]['ts']
        monday = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0, minute=0, second=0)
        pivot_target = monday + timedelta(hours=16)
        
        pivot_row = df_4h[df_4h['ts'] == pivot_target]
        if not pivot_row.empty:
            pivot_price = float(pivot_row.iloc[0]['open'])
            pivot_time = pivot_target.strftime("%A %H:00")
        else:
            pivot_price = float(df_4h.iloc[-1]['open'])
            pivot_time = "En attente"

        # D. GRILLE GANN
        dist = (current_price - pivot_price) % grid_size
        if dist > (grid_size/2): dist -= grid_size
        nearest_level = current_price - dist
        
        # E. MACRO DXY
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
            # Correctif DXY aussi au cas où
            if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)
            
            dxy_val_now = float(dxy['Close'].iloc[-1])
            dxy_val_prev = float(dxy['Close'].iloc[0])
            dxy_trend = "HAUSSIER (Risk Off)" if dxy_val_now > dxy_val_prev else "BAISSIER (Risk On)"
        except: dxy_trend = "N/A"

        return {
            "df_1h": df_1h,
            "price": current_price,
            "cycle": cycle_status,
            "ma20_wk": ma20_val,
            "pivot": pivot_price,
            "pivot_time": pivot_time,
            "grid_dist": float(dist),
            "nearest": float(nearest_level),
            "dxy": dxy_trend,
            "is_thursday": (datetime.now().weekday() == 3) and (datetime.now().hour >= 16)
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. CERVEAU GARDIEN ---
def ask_fractal_mind(data, target_timeframe):
    prompt = f"""
    TU ES LE GARDIEN DU NOYAU V13.1.
    OBJECTIF: Analyse FRACTALE pour : {target_timeframe}.
    
    DONNÉES:
    1. CYCLE HEBDO (MAJEUR) : {data['cycle']} (Niveau clé MM20: ${data['ma20_wk']:.0f})
    2. MACRO DXY : {data['dxy']}
    3. PIVOT HEBDO : ${data['pivot']:.0f}
    4. PRIX : ${data['price']:.0f}
    5. GANN : {data['grid_dist']:.0f}$ vs ${data['nearest']:.0f}
    
    RÈGLES IMPÉRATIVES:
    - Si CYCLE = "HIVER": INTERDIT DE SWING BUY. Seulement VENTE ou SCALP court.
    - Si PRIX < PIVOT (4H) : Biais Vendeur.
    
    SORTIE MARKDOWN:
    ### 🛡️ VERDICT ({target_timeframe})
    **1. ANALYSE CONTEXTE**
    [Analyse Macro/Weekly]
    
    **2. PLAN D'ACTION**
    * 🎯 **ORDRE :** [ACHAT/VENTE/ATTENTE]
    * 📍 **ZONE :** [Prix]
    * 🛑 **STOP :** [Prix]
    * 🏁 **CIBLE :** [Prix]
    
    **3. LOGIQUE**
    [Une phrase simple]
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except: return "⚠️ IA Indisponible."

# --- 5. VISUALISATION ---
def plot_chart(df, pivot, nearest, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="BTC"))
    fig.add_hline(y=pivot, line_width=2, line_dash="dash", line_color="orange", annotation_text="PIVOT")
    fig.add_hline(y=nearest, line_width=1, line_color="#3498DB", annotation_text="GANN")
    fig.update_layout(title=title, template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# --- 6. INTERFACE ---
with st.sidebar:
    st.header("🎛️ V13 CONTROL")
    grid_size = st.slider("Vibration ($)", 5000, 6000, 5760)
    if st.button("🔄 REBOOT"): st.cache_data.clear()

st.title("🛡️ QUANTUM CORE V13.1")

data = fetch_fractal_physics(grid_size)

if "error" not in data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRIX", f"${data['price']:,.0f}")
    
    # Couleur dynamique Cycle
    delta_col = "normal" if "ÉTÉ" in data['cycle'] else "inverse"
    c2.metric("CYCLE", data['cycle'], "MM20 Hebdo", delta_color=delta_col)
    
    c3.metric("PIVOT", f"${data['pivot']:,.0f}", f"{data['grid_dist']:.0f}")
    c4.metric("DXY", data['dxy'])

    t1, t2, t3 = st.tabs(["⚡ 1H", "⚔️ 4H", "🏛️ DAILY"])

    with t1:
        st.plotly_chart(plot_chart(data['df_1h'].tail(48), data['pivot'], data['nearest'], "1H View"), use_container_width=True)
        if st.button("SCAN 1H", type="primary"):
            with st.spinner("Analyse..."):
                st.markdown(ask_fractal_mind(data, "1H"))

    with t2:
        df_view = data['df_1h'].set_index('ts').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        st.plotly_chart(plot_chart(df_view.tail(60), data['pivot'], data['nearest'], "4H View"), use_container_width=True)
        if st.button("SCAN 4H"):
            with st.spinner("Analyse..."):
                st.markdown(ask_fractal_mind(data, "4H"))

    with t3:
        st.info(f"MM20 Hebdo (Seuil Hiver/Été) : ${data['ma20_wk']:,.0f}")
        if st.button("SCAN DAILY"):
            st.markdown(ask_fractal_mind(data, "DAILY"))

else:
    st.error(f"Erreur Données : {data['error']}")
