# ==============================================================================
# 🌌 QUANTUM CORE V13.0 - FRACTAL MIND EDITION
# "Todo List Level 5" pour Gemini Flash : Analyse Top-Down Forcée
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
    .bullish {border-left-color: #2ECC71;}
    .bearish {border-left-color: #E74C3C;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 300;}
    .stTabs [data-baseweb="tab"] {background-color: #262730; color: white;}
    .stTabs [aria-selected="true"] {background-color: #3498DB;}
</style>
""", unsafe_allow_html=True)

# --- 2. SÉCURITÉ ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("⚠️ CLÉ API MANQUANTE (Voir Secrets)")
    st.stop()

# --- 3. MOTEUR PHYSIQUE HYBRIDE (DATA V13) ---
@st.cache_data(ttl=300)
def fetch_fractal_physics(grid_size):
    try:
        # A. STRUCTURE LONG TERME (Yahoo - 2 Ans)
        # On a besoin de l'historique profond pour le contexte Monthly/Weekly
        df_macro = yf.download("BTC-USD", period="2y", interval="1wk", progress=False)
        
        # Calcul MM20 Hebdo (La Ligne de Vie)
        if not df_macro.empty:
            df_macro['MA20'] = df_macro['Close'].rolling(window=20).mean()
            ma20_val = df_macro['MA20'].iloc[-1]
            last_close = df_macro['Close'].iloc[-1]
            cycle_status = "HIVER (BAISSIER)" if last_close < ma20_val else "ÉTÉ (HAUSSIER)"
        else:
            cycle_status = "INCONNU"
            ma20_val = 0

        # B. PRÉCISION LIVE (Coinbase 1H)
        exchange = ccxt.coinbase()
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=500)
        df_1h = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], unit='ms')
        current_price = df_1h.iloc[-1]['close']
        
        # C. PIVOT TACTIQUE (Lundi 16h)
        # Reconstruction 4H pour trouver le pivot
        df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first', 'close':'last'}).dropna().reset_index()
        last_ts = df_4h.iloc[-1]['ts']
        monday = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0, minute=0, second=0)
        pivot_target = monday + timedelta(hours=16)
        
        pivot_row = df_4h[df_4h['ts'] == pivot_target]
        if not pivot_row.empty:
            pivot_price = pivot_row.iloc[0]['open']
            pivot_time = pivot_target.strftime("%A %H:00")
        else:
            pivot_price = df_4h.iloc[-1]['open'] # Fallback
            pivot_time = "En attente"

        # D. GRILLE GANN
        dist = (current_price - pivot_price) % grid_size
        if dist > (grid_size/2): dist -= grid_size
        nearest_level = current_price - dist
        
        # E. MACRO DXY
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
            dxy_trend = "HAUSSIER (Risk Off)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "BAISSIER (Risk On)"
        except: dxy_trend = "N/A"

        return {
            "df_1h": df_1h,
            "price": float(current_price),
            "cycle": cycle_status,
            "ma20_wk": float(ma20_val),
            "pivot": float(pivot_price),
            "pivot_time": pivot_time,
            "grid_dist": float(dist),
            "nearest": float(nearest_level),
            "dxy": dxy_trend,
            "is_thursday": (datetime.now().weekday() == 3) and (datetime.now().hour >= 16)
        }
    except Exception as e:
        return {"error": str(e)}

# --- 4. CERVEAU GARDIEN (TODO LIST LEVEL 5) ---
def ask_fractal_mind(data, target_timeframe):
    # C'est ici que la magie opère : Le Prompt Structuré
    prompt = f"""
    TU ES LE GARDIEN DU NOYAU V13.0.
    Ton objectif est de fournir une analyse FRACTALE DESCENDANTE pour le timeframe demandé : {target_timeframe}.
    
    DONNÉES BRUTES (VÉRITÉ MATHÉMATIQUE) :
    1. [MACRO] Cycle Hebdo (MM20) : {data['cycle']} (Niveau clé: ${data['ma20_wk']:.0f})
    2. [MACRO] DXY (Dollar) : {data['dxy']}
    3. [TACTIQUE] Pivot Hebdo : ${data['pivot']:.0f} (Date: {data['pivot_time']})
    4. [TACTIQUE] Prix Actuel : ${data['price']:.0f}
    5. [STRUCTURE] Position Grille Gann : {data['grid_dist']:.0f}$ vs niveau ${data['nearest']:.0f}
    
    TA TODO-LIST OBLIGATOIRE (Étape par Étape) :
    
    ETAPE 1 : CHECK MACRO (Monthly/Weekly)
    - Regarde le Cycle Hebdo. Si "HIVER", le biais de fond est VENDEUR. Si "ÉTÉ", le biais est ACHETEUR.
    - Note le conflit éventuel avec le DXY.
    
    ETAPE 2 : CHECK TACTIQUE (4H/Daily)
    - Compare Prix vs Pivot Hebdo.
    - Si Prix < Pivot : Les vendeurs dominent la semaine.
    - Si Prix > Pivot : Les acheteurs dominent.
    
    ETAPE 3 : CHECK CINETIQUE (1H - Seulement si demandé)
    - Si le timeframe cible est 1H, regarde la distance à la grille.
    
    ETAPE 4 : SYNTHÈSE OPÉRATIONNELLE (Le Verdict)
    - Combine les étapes 1, 2 et 3 pour donner un trade précis.
    
    FORMAT DE SORTIE (MARKDOWN STRICT) :
    
    ### 🛡️ ANALYSE FRACTALE ({target_timeframe})
    
    **1. LA VISION HÉLICOPTÈRE (Macro & Structure)**
    [Décris ici la tendance de fond (Monthly/Weekly) et le Cycle. Dis si le terrain est favorable ou hostile.]
    
    **2. LA BATAILLE ACTUELLE (Tactique 4H)**
    [Analyse la position par rapport au Pivot Hebdo ({data['pivot']}). Qui gagne ? Acheteurs ou Vendeurs ?]
    
    **3. OPPORTUNITÉ DE TRADE ({target_timeframe})**
    *C'est ici que tu dois être précis et simple.*
    
    * 🎯 **TYPE D'ORDRE :** [ACHAT LIMIT / VENTE LIMIT / ATTENTE]
    * 📍 **PRIX D'ENTRÉE :** [Donne un prix précis proche d'un niveau clé]
    * 🛑 **STOP LOSS :** [Niveau d'invalidation logique]
    * 🏁 **OBJECTIF (TP) :** [Prochain niveau Gann ou Pivot]
    * ⏳ **TIMING :** [Quand entrer ? Maintenant ou sur repli ?]
    
    **4. EXPLICATION SIMPLE**
    [Une phrase sans jargon pour expliquer pourquoi on prend ce risque.]
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except: return "⚠️ Cerveau IA hors ligne (Quota dépassé ou Erreur)."

# --- 5. VISUALISATION ---
def plot_chart(df, pivot, nearest, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['ts'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="BTC"))
    fig.add_hline(y=pivot, line_width=2, line_dash="dash", line_color="orange", annotation_text="PIVOT HEBDO")
    fig.add_hline(y=nearest, line_width=1, line_color="#3498DB", annotation_text="GANN MAGNET")
    fig.update_layout(title=title, template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# --- 6. INTERFACE V13 ---
with st.sidebar:
    st.header("🎛️ V13 CONTROL")
    grid_size = st.slider("Grille ($)", 5000, 6000, 5760)
    if st.button("🔄 FORCER ANALYSE"): st.cache_data.clear()

st.title("🛡️ QUANTUM CORE V13")
st.caption("Fractal Mind Analysis • Top-Down Logic")

data = fetch_fractal_physics(grid_size)

if "error" not in data:
    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BITCOIN", f"${data['price']:,.0f}")
    k2.metric("CYCLE HEBDO", data['cycle'])
    k3.metric("PIVOT", f"${data['pivot']:,.0f}", f"{data['grid_dist']:.0f} $ Gann")
    k4.metric("DXY", data['dxy'])

    # ONGLETS
    t1, t2, t3 = st.tabs(["⚡ 1H (SCALP)", "⚔️ 4H (SWING)", "🏛️ DAILY (INVEST)"])

    # 1H
    with t1:
        st.plotly_chart(plot_chart(data['df_1h'].tail(48), data['pivot'], data['nearest'], "Micro-Structure 1H"), use_container_width=True)
        if st.button("LANCER ANALYSE COMPLÈTE 1H", type="primary"):
            with st.spinner("Application de la logique fractale..."):
                report = ask_fractal_mind(data, "1H (SCALP)")
                st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)

    # 4H
    with t2:
        df_view = data['df_1h'].set_index('ts').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        st.plotly_chart(plot_chart(df_view.tail(60), data['pivot'], data['nearest'], "Structure Hebdo 4H"), use_container_width=True)
        if st.button("LANCER ANALYSE COMPLÈTE 4H"):
            with st.spinner("Analyse du Pivot et Cycle..."):
                report = ask_fractal_mind(data, "4H (SWING)")
                st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)

    # DAILY
    with t3:
        st.info(f"Moyenne Mobile 20 Semaines : ${data['ma20_wk']:,.0f}")
        if st.button("STRATÉGIE LONG TERME"):
            report = ask_fractal_mind(data, "DAILY (MACRO)")
            st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)

else:
    st.error(f"Erreur Données : {data['error']}")
        
