# ==============================================================================
# 🌌 QUANTUM CORE V11.0 - TITAN EDITION (PRO VISUALIZATION)
# Architecture : Streamlit + Plotly Interactive Charts + Deep AI Report
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

# --- 1. CONFIGURATION PRO ---
st.set_page_config(
    page_title="QUANTUM TITAN",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

# Style Minimaliste & Pro (Typographie Apple/Bloomberg)
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 300;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px; color: white;}
    .stTabs [aria-selected="true"] {background-color: #2980B9; color: white;}
    div[data-testid="metric-container"] {background-color: #262730; padding: 15px; border-radius: 10px; border-left: 4px solid #3498DB;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.warning("⚠️ Mode Démo : Clés API manquantes.")

# --- 3. MOTEUR GRAPHIQUE (PLOTLY) ---
def plot_quantum_chart(df, pivot, grid_lines, title):
    """Génère un graphique financier Pro interactif"""
    fig = go.Figure()

    # Bougies (Candlestick)
    fig.add_trace(go.Candlestick(
        x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='BTC', increasing_line_color='#2ECC71', decreasing_line_color='#E74C3C'
    ))

    # Pivot Lundi (Ligne Jaune)
    fig.add_hline(y=pivot, line_width=2, line_dash="dash", line_color="#F1C40F", annotation_text="PIVOT LUNDI")

    # Grille Gann (Lignes Bleues)
    for level in grid_lines:
        fig.add_hline(y=level, line_width=1, line_color="#3498DB", opacity=0.5, annotation_text=f"GRID {level:.0f}")

    # Layout Sombre Pro
    fig.update_layout(
        title=title,
        height=500,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# --- 4. MOTEUR D'ACQUISITION V11 ---
@st.cache_data(ttl=300)
def fetch_deep_data(grid_size):
    # Données étendues pour le graphique
    exchange = ccxt.coinbase()
    try:
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=500)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    except: return None

    current_price = df.iloc[-1]['close']
    
    # Calcul Pivot Lundi (Précis)
    df_4h = df.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
    last_ts = df_4h.iloc[-1]['ts']
    monday = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0, minute=0, second=0)
    pivot_row = df_4h[df_4h['ts'] == monday + timedelta(hours=16)]
    pivot_price = pivot_row.iloc[0]['open'] if not pivot_row.empty else df_4h.iloc[-1]['open']

    # Calcul des lignes de Grille à afficher (+/- 2 niveaux)
    grid_levels = []
    base_grid = pivot_price
    for i in range(-2, 3):
        grid_levels.append(base_grid + (i * grid_size))

    # Macro & News
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
        dxy_trend = "HAUSSIER (Risk Off)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "BAISSIER (Risk On)"
    except: dxy_trend = "N/A"

    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        news = [e.title for e in feed.entries[:3]]
    except: news = []

    return {
        "df": df,
        "price": current_price,
        "pivot": pivot_price,
        "grid_levels": grid_levels,
        "dxy": dxy_trend,
        "news": news,
        "grid_dist": (current_price - pivot_price) % grid_size
    }

# --- 5. ANALYSTE IA AVANCÉ ---
def ask_titan_brain(context, timeframe):
    prompt = f"""
    ROLE: Stratège Trading Crypto Institutionnel (Quantum V11).
    CONTEXTE: {context}
    TIMEFRAME: {timeframe}
    
    TA MISSION : Rédiger un Rapport Tactique Complet.
    Ne donne pas de JSON. Rédige en MARKDOWN propre et aéré.
    
    STRUCTURE DE TA RÉPONSE :
    ### 🎯 VERDICT : [ACHAT / VENTE / ATTENTE] (En Gras)
    
    **1. ANALYSE TECHNIQUE**
    Explique la position par rapport au Pivot et à la Grille Gann. Parle du Volume.
    
    **2. LE PLAN DE BATAILLE**
    - **Zone d'Entrée Idéale :** [Prix]
    - **Stop Loss (Invalidation) :** [Prix]
    - **Cibles (TP1 / TP2) :** [Prix]
    
    **3. EXPLICATION (Vulgarisation)**
    Une phrase simple avec une métaphore (ex: ressort, plafond de verre) pour que l'utilisateur comprenne la physique du mouvement.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"⚠️ Erreur IA : {str(e)}"

# --- 6. INTERFACE UTILISATEUR V11 ---

# Sidebar
with st.sidebar:
    st.header("🎛️ CONTRÔLE")
    grid_size = st.number_input("Vibration Gann ($)", value=5760, step=10)
    if st.button("🔄 Rafraîchir Système"): st.cache_data.clear()
    st.markdown("---")
    st.caption("Quantum Core V11.0 Titan")

# En-tête Dashboard
st.title("🛡️ QUANTUM TITAN")
data = fetch_deep_data(grid_size)

if data:
    # BANDEAU HAUT (MÉTRIQUES CLÉS)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BITCOIN (Live)", f"${data['price']:,.0f}")
    k2.metric("PIVOT HEBDO", f"${data['pivot']:,.0f}", delta_color="off")
    k3.metric("DIST. GRILLE", f"{data['grid_dist']:.0f} $", "Prochain niveau")
    k4.metric("MACRO DXY", data['dxy'])

    # ONGLETS DE NAVIGATION (Le Secret de la lisibilité)
    tab1, tab2, tab3 = st.tabs(["⚡ 1H (SCALP)", "⚔️ 4H (SWING)", "🏛️ DAILY (MACRO)"])

    # --- ONGLET 1H ---
    with tab1:
        st.subheader("Analyse Cinétique & Inertie")
        col_graph, col_ana = st.columns([2, 1]) # Graphique prend 2/3 de la place
        
        with col_graph:
            # On affiche les 48 dernières heures seulement pour le zoom 1H
            st.plotly_chart(plot_quantum_chart(data['df'].tail(48), data['pivot'], data['grid_levels'], "BTC/USD - 1H View"), use_container_width=True)
        
        with col_ana:
            st.info("💡 Cliquez ci-dessous pour l'analyse IA")
            if st.button("LANCER LE SCAN 1H", key="btn_1h", type="primary"):
                with st.spinner("Le Gardien analyse l'inertie..."):
                    report = ask_titan_brain(f"Prix: {data['price']}, Pivot: {data['pivot']}", "1H")
                    st.markdown(report)
                    st.button("💾 Enregistrer Trade", key="save1")

    # --- ONGLET 4H ---
    with tab2:
        st.subheader("Stratégie Hebdomadaire (Le Juge de Paix)")
        col_graph4, col_ana4 = st.columns([2, 1])
        
        with col_graph4:
            # Vue plus large (14 jours)
            # Resample local pour l'affichage
            df_4h_view = data['df'].set_index('ts').resample('4h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna().reset_index()
            st.plotly_chart(plot_quantum_chart(df_4h_view.tail(84), data['pivot'], data['grid_levels'], "BTC/USD - 4H Structure"), use_container_width=True)
            
        with col_ana4:
            st.warning("⚠️ Attention à la règle du Jeudi 16h")
            if st.button("LANCER LE SCAN 4H", key="btn_4h", type="primary"):
                with st.spinner("Le Gardien vérifie la structure..."):
                    report = ask_titan_brain(f"Prix: {data['price']}, Pivot: {data['pivot']}", "4H")
                    st.markdown(report)
                    st.button("💾 Enregistrer Trade", key="save2")

    # --- ONGLET DAILY ---
    with tab3:
        st.subheader("Vision Long Terme (Saisons)")
        st.markdown(f"**NEWS EN TEMPS RÉEL :**")
        for n in data['news']:
            st.write(f"📰 {n}")
        
        st.markdown("---")
        if st.button("ANALYSE MACRO COMPLÈTE", key="btn_daily"):
             with st.spinner("Lecture des cycles planétaires..."):
                    report = ask_titan_brain(f"Prix: {data['price']}, Tendance DXY: {data['dxy']}", "DAILY")
                    st.markdown(report)

else:
    st.error("Erreur de connexion aux marchés. Vérifiez l'API.")
    
