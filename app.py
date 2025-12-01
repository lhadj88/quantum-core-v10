<!-- end list -->
# ==============================================================================
# 🌌 QUANTUM CORE V10.3 - COMMAND CENTER (STREAMLIT EDITION)
# Architecture : Mobile-First | Fullstack | Neuro-Symbolic
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import feedparser
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time

# --- 1. CONFIGURATION DE L'INTERFACE (DESIGN MODERNE) ---
st.set_page_config(
    page_title="QUANTUM CORE V10",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Personnalisé (Look "Modern Professional")
st.markdown("""
<style>
    .stApp {background-color: #F8F9FA;} /* Fond clair pro */
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

# --- 2. GESTION DES SECRETS (SÉCURITÉ) ---
# Le code va chercher les clés dans st.secrets (configuré plus tard sur Streamlit Cloud)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_SHEETS_CREDENTIALS = st.secrets["gcp_service_account"]
    genai.configure(api_key=API_KEY)
except:
    st.error("⚠️ CLÉS API MANQUANTES. Configurez les 'Secrets' dans Streamlit Cloud.")
    st.stop()

# --- 3. MOTEUR PHYSIQUE V10 (CACHÉ POUR PERFORMANCE) ---
@st.cache_data(ttl=300) # Mise à jour toutes les 5 min pour économiser les requêtes
def fetch_market_physics(grid_size, atom_size):
    # A. MACRO (Yahoo)
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        dxy_trend = "HAUSSIER (Risk Off)" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[0] else "BAISSIER (Risk On)"
    except: dxy_trend = "Inconnu"

    # B. NEWS (RSS CoinDesk)
    news_titles = []
    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        for entry in feed.entries[:3]: news_titles.append(entry.title)
    except: news_titles = ["Flux indisponible"]

    # C. CRYPTO (Coinbase via CCXT)
    exchange = ccxt.coinbase()
    try:
        bars = exchange.fetch_ohlcv("BTC/USD", timeframe="1h", limit=300)
        df_1h = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df_1h['ts'] = pd.to_datetime(df_1h['ts'], unit='ms')
    except:
        return None # Gestion erreur

    # Calculs Fractals
    current_price = df_1h.iloc[-1]['close']
    
    # 4H Synthétique pour Pivot
    df_4h = df_1h.set_index('ts').resample('4h').agg({'open':'first','close':'last'}).dropna().reset_index()
    last_ts = df_4h.iloc[-1]['ts']
    monday_date = (last_ts - timedelta(days=last_ts.weekday())).replace(hour=0,minute=0,second=0)
    
    # Recherche Pivot Lundi 16h
    pivot_row = df_4h[df_4h['ts'] == (monday_date + timedelta(hours=16))]
    pivot_price = pivot_row.iloc[0]['open'] if not pivot_row.empty else df_4h.iloc[-1]['open']
    
    # Grille
    dist = (current_price - pivot_price) % grid_size
    if dist > (grid_size/2): dist -= grid_size
    nearest_level = current_price - dist
    
    # Structure Hebdo (Approx MM20 sur 1H resample)
    df_wk = df_1h.set_index('ts').resample('W-MON').agg({'close':'last'}).dropna()
    ma20_wk = df_wk['close'].rolling(20).mean().iloc[-1] if len(df_wk)>20 else current_price
    wk_trend = "BAISSIER" if current_price < ma20_wk else "HAUSSIER"

    return {
        "price": current_price,
        "pivot": pivot_price,
        "grid_dist": dist,
        "nearest": nearest_level,
        "wk_trend": wk_trend,
        "dxy": dxy_trend,
        "news": news_titles,
        "is_thursday": (datetime.now().weekday() == 3) and (datetime.now().hour >= 16),
        "vol_spike": df_1h.iloc[-1]['vol'] > (df_1h['vol'].rolling(20).mean().iloc[-1] * 1.5)
    }

# --- 4. CERVEAU GARDIEN (GEMINI 2.0) ---
def get_ai_verdict(data, timeframe):
    system_prompt = f"""
    TU ES LE GARDIEN V10.3. Analyse ce marché pour le timeframe {timeframe}.
    INPUT: {json.dumps(data)}
    RÈGLES:
    - Si Weekly BEARISH -> Interdit d'acheter (sauf scalp 1H).
    - Si Prix < Pivot -> Biais Vendeur.
    FORMAT JSON STRICT:
    {{
        "action": "ACHETER" | "VENDRE" | "ATTENDRE",
        "confiance": "ÉLEVÉE" | "MOYENNE",
        "stop_loss": 00000,
        "target": 00000,
        "raison_technique": "Phrase courte",
        "vulgarisation": "Explication simple pour débutant"
    }}
    """
    model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system_prompt)
    try:
        resp = model.generate_content("ANALYSE", generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)
    except:
        return {"action": "ERREUR", "raison_technique": "IA Indisponible"}

# --- 5. INTERFACE UTILISATEUR (UI) ---

# Sidebar (Laboratoire)
with st.sidebar:
    st.header("⚙️ PARAMÈTRES V10")
    grid_setting = st.slider("Grille Vibratoire ($)", 5000, 7000, 5760)
    atom_setting = st.slider("Atome Cinétique ($)", 500, 1000, 720)
    if st.button("Reset Usine"): st.rerun()
    st.info(f"Système Connecté • {datetime.now().strftime('%H:%M UTC')}")

# En-tête
st.title("🛡️ QUANTUM COMMAND CENTER")
data = fetch_market_physics(grid_setting, atom_setting)

if data:
    # Métriques Macro
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix Bitcoin", f"${data['price']:,.0f}", f"{data['price']-data['pivot']:,.0f} vs Pivot")
    c2.metric("Tendance Hebdo", data['wk_trend'], "Structure")
    c3.metric("Météo Macro (DXY)", data['dxy'])
    c4.metric("Pivot Lundi", f"${data['pivot']:,.0f}")

    st.markdown("---")

    # CARTE DES OPÉRATIONS (3 Colonnes)
    col_1h, col_4h, col_d = st.columns(3)

    # --- TUILE 1H (SCALP) ---
    with col_1h:
        st.subheader("⚡ 1H • CINÉTIQUE")
        if st.button("SCAN 1H", use_container_width=True):
            verdict = get_ai_verdict(data, "1H")
            color_class = "bullish" if verdict['action'] == "ACHETER" else "bearish" if verdict['action'] == "VENDRE" else "neutral"
            
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h2>{verdict['action']}</h2>
                <p>Confiance: {verdict['confiance']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("DÉTAILS OPÉRATIONNELS"):
                st.write(f"**STOP:** ${verdict['stop_loss']}")
                st.write(f"**CIBLE:** ${verdict['target']}")
                st.info(f"🤖 **Tech:** {verdict['raison_technique']}")
                st.success(f"🎓 **Simple:** {verdict['vulgarisation']}")
                
                # Bouton Journal
                if st.button("💾 SAUVEGARDER CE TRADE", key="save_1h"):
                    # Code d'enregistrement GSheet ici (simulé pour l'instant)
                    st.toast("Trade archivé dans le Journal de Guerre !")

    # --- TUILE 4H (TACTIQUE) ---
    with col_4h:
        st.subheader("⚔️ 4H • TACTIQUE")
        if st.button("SCAN 4H", use_container_width=True):
            verdict = get_ai_verdict(data, "4H")
            color_class = "bullish" if verdict['action'] == "ACHETER" else "bearish" if verdict['action'] == "VENDRE" else "neutral"
            
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h2>{verdict['action']}</h2>
                <p>Zone Pivot: {data['pivot']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("RAPPORT D'ÉTAT-MAJOR"):
                st.write(f"**STOP:** ${verdict['stop_loss']}")
                st.write(f"**CIBLE:** ${verdict['target']}")
                st.info(f"🤖 {verdict['raison_technique']}")
                st.success(f"🎓 {verdict['vulgarisation']}")
                if st.button("💾 SAUVEGARDER", key="save_4h"):
                    st.toast("Archivé !")

    # --- TUILE DAILY (STRUCTURE) ---
    with col_d:
        st.subheader("🏛️ DAILY • STRATÈGE")
        st.markdown(f"**Prochaine Grille:** ${data['nearest']:,.0f}")
        st.progress(max(0, min(100, int(50 + (data['grid_dist']/grid_setting)*100))))
        st.caption("Position relative à la Vibration")
        
        with st.expander("ANALYSE DE FOND"):
            st.write(f"**News:** {data['news'][0]}")
            st.write(f"**Cycle:** {'HIVER' if data['wk_trend']=='BAISSIER' else 'ÉTÉ'}")

else:
    st.error("Erreur de connexion aux marchés.")
      
