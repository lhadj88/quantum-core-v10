# ==============================================================================
# 🌌 QUANTUM CORE V12.0 - DEEP DATA EDITION
# Architecture : Yahoo (Macro History) + Coinbase (Live Precision) + Gemini Flash Lock
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
st.set_page_config(
    page_title="QUANTUM V12",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# CSS "App Native"
st.markdown("""
<style>
    .stApp {background-color: #0E1117; color: #FAFAFA;}
    .metric-container {background-color: #262730; padding: 15px; border-radius: 8px; border-left: 4px solid #3498DB;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 300;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {background-color: #1E1E1E; border-radius: 5px; color: white;}
    .stTabs [aria-selected="true"] {background-color: #2980B9;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ CLÉ API MANQUANTE (Voir Secrets)")
    st.stop()

# --- 3. MOTEUR PHYSIQUE HYBRIDE (DEEP DATA) ---
@st.cache_data(ttl=300)
def fetch_deep_physics(grid_size):
    try:
        # A. STRUCTURE LONG TERME (Yahoo Finance - 2 Ans d'historique)
        # On utilise Yahoo pour avoir assez de données pour la MM20 Hebdo
        start_date = (datetime.now() - timedelta(weeks=100)).strftime('%Y-%m-%d')
        df_wk = yf.download("BTC-USD", start=start_date, interval="1wk", progress=False)
        
        # Calcul MM20 Hebdo
        if not df_wk.empty:
            df_wk['MA20'] = df_wk['Close'].rolling(window=20).mean()
            ma20_val = df_wk['MA20'].iloc[-1]
            last_close_wk = df_wk['Close'].iloc[-1]
            # La règle d'Or : Prix < MM20 = HIVER
            cycle_status = "HIVER (BAISSIER)" if last_close_wk < ma20_val else "ÉTÉ (HAUSSIER)"
        else:
            cycle_status = "INCONNU"
            ma20_val = 0

        # B. PRÉCISION COURT TERME (Coinbase via CCXT)
        exchange = ccxt.coinbase()
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=500)
        df_1h = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], unit='ms')
        
        current_price = df_1h.iloc[-1]['close']
        
        # C. PIVOT TACTIQUE (4H)
        df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
        last_ts = df_4h.iloc[-1]['ts']
        monday = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0, minute=0, second=0)
        pivot_target = monday + timedelta(hours=16)
        
        pivot_row = df_4h[df_4h['ts'] == pivot_target]
        pivot_price = pivot_row.iloc[0]['open'] if not pivot_row.empty else df_4h.iloc[-1]['open']
        
        # D. GRILLE GANN
        dist = (current_price - pivot_price) % grid_size
        if dist > (grid_size/2): dist -= grid_size
        nearest_level = current_price - dist
        
        # E. MACRO DXY
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        dxy_trend = "HAUSSIER (Mauvais)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "BAISSIER (Bon)"

        # F. NEWS RSS
        news = []
        try:
            f = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
            news = [e.title for e in f.entries[:3]]
        except: news = ["Flux HS"]

        return {
            "df_1h": df_1h,
            "price": float(current_price),
            "cycle": cycle_status,
            "ma20_wk": float(ma20_val),
            "pivot": float(pivot_price),
            "grid_dist": float(dist),
            "nearest": float(nearest_level),
            "dxy": dxy_trend,
            "news": news,
            "is_thursday": (datetime.now().weekday() == 3) and (datetime.now().hour >= 16)
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. CERVEAU GARDIEN (CADENASSÉ) ---
def ask_guardian_v12(data, tf):
    # On injecte la vérité mathématique dans le prompt pour empêcher l'hallucination
    prompt = f"""
    TU ES LE GARDIEN V12.0.
    
    DONNÉES VÉRIFIÉES (NE PAS CONTESTER):
    - PRIX ACTUEL: ${data['price']}
    - CYCLE HEBDO (LOI SUPRÊME): {data['cycle']}
    - PIVOT LUNDI: ${data['pivot']}
    - DISTANCE GRILLE: {data['grid_dist']} $ (Si proche de 0 = Support/Résistance)
    
    RÈGLES D'ENGAGEMENT STRICTES:
    1. SI CYCLE = "HIVER (BAISSIER)":
       - INTERDICTION FORMELLE de proposer "ACHAT" en Swing (4H/Daily).
       - Seuls les scalps (1H) très courts sont tolérés.
       - La priorité est "VENTE SUR REBOND" ou "ATTENTE".
       
    2. SI TIMEFRAME = "4H":
       - Regarde le PIVOT. Si Prix < Pivot ET Hiver -> SHORT AGRESSIF.
       
    TA MISSION:
    Rédige un rapport Markdown ultra-clair.
    
    STRUCTURE:
    ### 🎯 DÉCISION: [ACHAT / VENTE / ATTENTE]
    **Confiance:** [0-100]%
    
    **Analyse du Gardien:**
    [Explique en 2 phrases pourquoi, en citant le Cycle Hebdo et le Pivot]
    
    **Paramètres de Tir:**
    - 🔵 Entrée: [Prix]
    - 🔴 Stop Loss: [Prix]
    - 🟢 Objectif: [Prix]
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except: return "⚠️ Erreur Cerveau (Quota ou Panne)"

# --- 5. VISUALISATION (PLOTLY) ---
def plot_chart(df, pivot, nearest, title):
    fig = go.Figure()
    # Chandeliers
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="BTC"))
    # Pivot Lundi
    fig.add_hline(y=pivot, line_width=2, line_dash="dash", line_color="orange", annotation_text="PIVOT")
    # Grille Gann
    fig.add_hline(y=nearest, line_width=1, line_color="#3498DB", annotation_text="GANN LEVEL")
    
    fig.update_layout(title=title, template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# --- 6. INTERFACE ---
with st.sidebar:
    st.header("🎛️ V12 CONTROL")
    grid_size = st.slider("Vibration ($)", 5000, 6000, 5760)
    if st.button("🔄 FORCE REFRESH"): st.cache_data.clear()

st.title("🛡️ QUANTUM CORE V12")
st.caption("Deep Data & Flash Lock Architecture")

data = fetch_deep_physics(grid_size)

if "error" not in data:
    # BANDEAU MACRO
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BITCOIN", f"${data['price']:,.0f}")
    
    # Couleur conditionnelle pour le cycle
    cycle_color = "normal"
    if "HIVER" in data['cycle']: cycle_color = "inverse" # Affiche en rouge/gris selon theme
    k2.metric("CYCLE HEBDO", data['cycle'], "MM20 Trend")
    
    k3.metric("PIVOT", f"${data['pivot']:,.0f}", f"{data['grid_dist']:.0f} vs Gann")
    k4.metric("DXY", data['dxy'])

    # ONGLETS
    t1, t2, t3 = st.tabs(["⚡ 1H (SCALP)", "⚔️ 4H (SWING)", "🏛️ DAILY (MACRO)"])

    # 1H
    with t1:
        st.plotly_chart(plot_chart(data['df_1h'].tail(48), data['pivot'], data['nearest'], "Cinétique 1H"), use_container_width=True)
        if st.button("LANCER ANALYSE 1H", type="primary"):
            with st.spinner("Calcul Inertie..."):
                st.markdown(ask_guardian_v12(data, "1H"))

    # 4H
    with t2:
        # Resample pour vue large
        df_view = data['df_1h'].set_index('ts').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        st.plotly_chart(plot_chart(df_view.tail(60), data['pivot'], data['nearest'], "Structure 4H"), use_container_width=True)
        if st.button("LANCER ANALYSE 4H"):
            with st.spinner("Vérification Pivot & Cycle..."):
                st.markdown(ask_guardian_v12(data, "4H"))

    # DAILY
    with t3:
        st.info(f"Moyenne Mobile 20 Semaines (Ligne de Vie) : ${data['ma20_wk']:,.0f}")
        st.write("NEWS FEED:")
        for n in data['news']: st.write(f"- {n}")
        if st.button("STRATÉGIE LONG TERME"):
            st.markdown(ask_guardian_v12(data, "DAILY"))

else:
    st.error(f"Erreur Données : {data['error']}")
            
