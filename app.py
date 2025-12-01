# ==============================================================================
# 🌌 QUANTUM CORE V14.0 - SINGULARITY EDITION
# Architecture : Async Parallel Fetching | Socratic AI | Deep Space UI
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
from concurrent.futures import ThreadPoolExecutor

# --- 1. CONFIGURATION SYSTÈME ---
st.set_page_config(
    page_title="QUANTUM V14",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

# DESIGN SYSTEM "DEEP SPACE" (CSS V14)
st.markdown("""
<style>
    /* Fond & Structure */
    .stApp {background-color: #0B0E11; color: #E0E0E0;}
    
    /* Indicateurs HUD (Head-Up Display) */
    .hud-box {
        background-color: #15191F; 
        padding: 15px; 
        border-radius: 8px; 
        border-top: 3px solid #34495E;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .hud-val {font-size: 1.5rem; font-weight: 700; color: #FFFFFF;}
    .hud-label {font-size: 0.8rem; text-transform: uppercase; color: #7F8C8D; letter-spacing: 1px;}
    
    /* Couleurs Sémantiques */
    .bull {border-top-color: #00FFA3 !important; color: #00FFA3;}
    .bear {border-top-color: #FF0055 !important; color: #FF0055;}
    .neutral {border-top-color: #3498DB !important; color: #3498DB;}
    
    /* Onglets & Boutons */
    .stTabs [data-baseweb="tab-list"] {background-color: #15191F; border-radius: 10px; padding: 5px;}
    .stTabs [data-baseweb="tab"] {color: #95A5A6;}
    .stTabs [aria-selected="true"] {background-color: #34495E; color: white; border-radius: 5px;}
    
    /* Loader */
    .stSpinner > div {border-top-color: #00FFA3 !important;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⛔ SÉCURITÉ : Clé API manquante.")
    st.stop()

# --- 3. MOTEUR D'ACQUISITION PARALLÈLE (VITESSE X3) ---

def get_crypto_data():
    """Récupère les prix Coinbase (1H)"""
    try:
        ex = ccxt.coinbase()
        bars = ex.fetch_ohlcv("BTC/USD", timeframe="1h", limit=500)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return pd.DataFrame()

def get_macro_data():
    """Récupère DXY et Historique Weekly (Yahoo)"""
    try:
        # DXY (Court terme)
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        dxy_trend = "RISK_OFF (Hausse $)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "RISK_ON (Baisse $)"
        
        # BTC Weekly (Long terme pour MM20)
        btc_wk = yf.download("BTC-USD", period="2y", interval="1wk", progress=False)
        if isinstance(btc_wk.columns, pd.MultiIndex): btc_wk.columns = btc_wk.columns.get_level_values(0)
        
        # Calcul MM20
        btc_wk['MA20'] = btc_wk['Close'].rolling(20).mean()
        last_ma = float(btc_wk['MA20'].iloc[-1])
        last_price = float(btc_wk['Close'].iloc[-1])
        cycle = "HIVER" if last_price < last_ma else "ÉTÉ"
        
        return {"dxy": dxy_trend, "cycle": cycle, "ma20": last_ma}
    except: return {"dxy": "N/A", "cycle": "INCONNU", "ma20": 0}

def get_news_data():
    """Récupère le sentiment RSS"""
    try:
        f = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        return [e.title for e in f.entries[:3]]
    except: return []

@st.cache_data(ttl=120, show_spinner=False)
def fetch_quantum_data(grid_size):
    # Exécution Parallèle (Threading)
    with ThreadPoolExecutor() as executor:
        f_crypto = executor.submit(get_crypto_data)
        f_macro = executor.submit(get_macro_data)
        f_news = executor.submit(get_news_data)
        
        df_1h = f_crypto.result()
        macro = f_macro.result()
        news = f_news.result()

    if df_1h.empty: return {"error": "Connexion Coinbase échouée"}

    # --- CALCULS PHYSIQUES V14 ---
    current_price = float(df_1h.iloc[-1]['close'])
    
    # Pivot Hebdo (Algorithme Temporel)
    df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
    last_ts = df_4h.iloc[-1]['ts']
    monday = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0, minute=0, second=0)
    pivot_row = df_4h[df_4h['ts'] == monday + timedelta(hours=16)]
    pivot_val = float(pivot_row.iloc[0]['open']) if not pivot_row.empty else float(df_4h.iloc[-1]['open'])
    
    # Grille Gann
    dist = (current_price - pivot_val) % grid_size
    if dist > (grid_size/2): dist -= grid_size
    nearest = current_price - dist
    
    # Vérité Jeudi
    is_thursday = (datetime.now().weekday() == 3) and (datetime.now().hour >= 16)

    return {
        "df": df_1h,
        "price": current_price,
        "pivot": pivot_val,
        "cycle": macro['cycle'],
        "ma20": macro['ma20'],
        "dxy": macro['dxy'],
        "news": news,
        "grid_dist": float(dist),
        "nearest": float(nearest),
        "is_thursday": is_thursday
    }

# --- 4. CERVEAU SOCRATIQUE (LOGIQUE FORCÉE) ---
def ask_guardian_v14(data, timeframe):
    # Prompt Socratique : Oblige l'IA à valider les étapes avant de conclure
    prompt = f"""
    TU ES LE GARDIEN V14.
    Analyses le marché pour : {timeframe}.
    
    DONNÉES VÉRIFIÉES :
    1. CYCLE HEBDO : {data['cycle']} (Seuil Hiver/Été : ${data['ma20']:.0f})
    2. PRIX : ${data['price']:.0f}
    3. PIVOT LUNDI : ${data['pivot']:.0f}
    4. GANN : {data['grid_dist']:.0f}$ vs niveau ${data['nearest']:.0f}
    
    MÉTHODE SOCRATIQUE (Réponds mentalement) :
    Q1: Le cycle hebdo est-il BAISSIER ? (Si oui, achat interdit sauf scalp).
    Q2: Le prix est-il sous le Pivot ? (Si oui, pression vendeuse).
    Q3: Sommes-nous sur une ligne Gann ? (Si oui, réaction probable).
    
    FORMAT DE SORTIE (MARKDOWN STRICT) :
    
    ### 🛡️ VERDICT : [ACHAT / VENTE / ATTENTE]
    
    **1. LOGIQUE DÉDUCTIVE**
    [Explique ton raisonnement en citant Q1 et Q2. Sois froid et direct.]
    
    **2. ORDRE DE MISSION ({timeframe})**
    * 🎯 **TYPE :** [LIMIT / MARKET]
    * 📍 **ZONE D'ENTRÉE :** [Prix précis]
    * 🛑 **STOP LOSS :** [Prix précis]
    * 🏁 **CIBLE (TP) :** [Prix précis]
    
    **3. PROBABILITÉ :** [FAIBLE / MOYENNE / ÉLEVÉE]
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except: return "⚠️ ERREUR CERVEAU : IA Indisponible."
    # --- 5. VISUALISATION AVANCÉE (PLOTLY DARK) ---
def render_chart(df, pivot, nearest, title):
    fig = go.Figure()
    
    # Bougies
    fig.add_trace(go.Candlestick(
        x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="BTC", increasing_line_color='#00FFA3', decreasing_line_color='#FF0055'
    ))
    
    # Pivot (Ligne Jaune pointillée)
    fig.add_hline(y=pivot, line_dash="dash", line_color="#F1C40F", annotation_text="PIVOT LUNDI", annotation_position="top right")
    
    # Gann (Ligne Bleue continue)
    fig.add_hline(y=nearest, line_color="#3498DB", annotation_text="GANN LEVEL", annotation_position="bottom right")
    
    # Layout "Deep Space"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False
    )
    return fig

# --- 6. TABLEAU DE BORD (MAIN LOOP) ---

# Sidebar
with st.sidebar:
    st.header("🎛️ SYSTÈME V14")
    grid_size = st.slider("Vibration ($)", 5000, 7000, 5760)
    st.markdown("---")
    if st.button("🚀 RELANCER MOTEUR"): st.cache_data.clear()
    st.caption("Quantum Singularity Active")

# Titre
st.title("QUANTUM SINGULARITY")
st.markdown("Réseau Neuronal Financier • Architecture V14")

# Chargement
with st.spinner("🔄 Synchronisation des flux parallèles..."):
    data = fetch_quantum_data(grid_size)

if "error" in data:
    st.error(f"❌ PANNE SYSTÈME : {data['error']}")
else:
    # --- HUD (HEAD-UP DISPLAY) ---
    # Un bandeau d'information permanent et stylisé
    col1, col2, col3, col4 = st.columns(4)
    
    # Couleur dynamique du Cycle
    cyc_col = "bear" if "HIVER" in data['cycle'] else "bull"
    
    with col1:
        st.markdown(f"""<div class="hud-box"><div class="hud-val">${data['price']:,.0f}</div><div class="hud-label">BITCOIN LIVE</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="hud-box {cyc_col}"><div class="hud-val">{data['cycle']}</div><div class="hud-label">CYCLE HEBDO</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="hud-box"><div class="hud-val">${data['pivot']:,.0f}</div><div class="hud-label">PIVOT TACTIQUE</div></div>""", unsafe_allow_html=True)
    with col4:
        dist_col = "bull" if abs(data['grid_dist']) < 150 else "neutral"
        st.markdown(f"""<div class="hud-box {dist_col}"><div class="hud-val">{data['grid_dist']:.0f} $</div><div class="hud-label">DIST. GANN</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- ZONE DE COMMANDEMENT (ONGLETS) ---
    t1, t2, t3 = st.tabs(["⚡ 1H (SCALP)", "⚔️ 4H (SWING)", "🏛️ DAILY (MACRO)"])

    # ONGLET 1H
    with t1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.plotly_chart(render_chart(data['df'].tail(48), data['pivot'], data['nearest'], "Cinétique 1H"), use_container_width=True)
        with c_right:
            st.markdown("### 🤖 ANALYSE IMMÉDIATE")
            if st.button("SCANNERS 1H ACTIVÉS", type="primary", use_container_width=True):
                with st.spinner("Calcul des probabilités..."):
                    report = ask_guardian_v14(data, "1H (SCALP)")
                    st.markdown(report)

    # ONGLET 4H
    with t2:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            # Resample 4H
            df_4h = data['df'].set_index('ts').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
            st.plotly_chart(render_chart(df_4h.tail(60), data['pivot'], data['nearest'], "Structure 4H"), use_container_width=True)
        with c_right:
            st.markdown("### 🧬 LOGIQUE HEBDOMADAIRE")
            st.info("Règle : Si Prix < Pivot et Hiver -> Short Prioritaire")
            if st.button("SCANNERS 4H ACTIVÉS", use_container_width=True):
                with st.spinner("Consultation du Gardien..."):
                    report = ask_guardian_v14(data, "4H (SWING)")
                    st.markdown(report)

    # ONGLET DAILY
    with t3:
        st.subheader("Données de Fond")
        st.write(f"**MM20 Hebdomadaire (Seuil Hiver/Été) :** ${data['ma20']:,.0f}")
        st.write(f"**Contexte DXY :** {data['dxy']}")
        
        st.markdown("#### 📰 FLUX INTELLIGENCE")
        for news in data['news']:
            st.code(news, language="text")
            
        if st.button("ANALYSE STRATÉGIQUE GLOBALE", use_container_width=True):
            st.markdown(ask_guardian_v14(data, "DAILY (INVEST)"))
