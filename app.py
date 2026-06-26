
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ── Make sure imports work regardless of cwd on Streamlit Cloud ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from predict import (
    load_models,
    predict_match,
    load_live_results,
    save_result,
    compute_live_elo,
    MODELS_FILE,
)
from data_prep import get_teams_df
from models import train_all


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
.stApp { background-color: #ffffff !important; color: #1a1a2e; }
section[data-testid="stSidebar"] {
    background-color: #f4f6f9 !important;
    border-right: 1px solid #e0e4ea;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background-color: #f4f6f9;
    border: 1px solid #dde1e9;
    border-radius: 10px;
    padding: 14px 18px;
}

/* ── Probability bars ── */
.prob-wrap {
    background-color: #eef0f4;
    border-radius: 8px;
    height: 30px;
    width: 100%;
    margin: 4px 0 14px 0;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding-left: 12px;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
}

/* ── Score badge ── */
.score-badge {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    font-size: 56px;
    font-weight: 900;
    letter-spacing: 6px;
    margin: 14px 0;
}

/* ── Section header ── */
.sec-header {
    background: linear-gradient(90deg, #1a1a2e 0%, #0f3460 100%);
    color: #ffffff;
    padding: 14px 22px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 18px;
}

/* ── Scoreline row ── */
.score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 14px;
    border-radius: 6px;
    margin: 3px 0;
    font-size: 14px;
    background-color: #f4f6f9;
}
.score-row:hover { background-color: #e8ecf2; }

/* ── Live badge ── */
.live-badge {
    display: inline-block;
    background-color: #e74c3c;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 8px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* ── Info card ── */
.info-card {
    background-color: #f0f7ff;
    border: 1px solid #b8d4f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a1a2e;
    border-bottom: 3px solid #1a1a2e;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_bundle():
    """
    Load models from models.joblib, re-training automatically if:
      • the file does not exist (fresh Streamlit Cloud deploy), or
      • the file is from a different joblib version (rare but possible), or
      • the file is corrupt / incomplete.

    Re-training writes a fresh models.joblib next to app.py so subsequent
    cache misses within the same session are fast.
    """
    joblib_path = MODELS_FILE  # absolute path resolved in predict.py

    # ── Try loading an existing file first ────────────────────────────────
    if os.path.exists(joblib_path):
        try:
            bundle = load_models(joblib_path)
            # Sanity-check the bundle has everything we need
            required = {'xgb_model', 'home_poisson', 'away_poisson',
                        'label_encoder', 'train_marginals',
                        'xgb_accuracy', 'xgb_cv_accuracy', 'poisson_mae'}
            if required.issubset(bundle.keys()):
                return bundle
            # Missing keys → fall through to retrain
        except Exception:
            pass  # Unpickling failed (version mismatch, corruption, etc.)

    # ── Train from scratch ────────────────────────────────────────────────
    with st.spinner("🏋️ First-time setup: training models from historical data (~30s)…"):
        try:
            bundle = train_all(save_path=joblib_path)
            return bundle
        except FileNotFoundError as exc:
            st.error(
                f"**Cannot find the historical CSV.**\n\n"
                f"{exc}\n\n"
                "Make sure `data/master_historical_features (1).csv` is committed "
                "to your GitHub repo."
            )
            st.stop()
        except Exception as exc:
            st.error(f"**Training failed:** {exc}")
            st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_static_teams():
    return get_teams_df()


def prob_bar(label, prob, color):
    """Render a coloured probability bar."""
    pct = prob * 100
    w   = max(pct, 4)
    st.markdown(f"""
    <div style="margin-bottom:4px;font-size:13px;font-weight:600;color:#1a1a2e;">{label}</div>
    <div class="prob-wrap">
        <div class="prob-fill" style="width:{w:.0f}%;background:{color};">{pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)


def verdict(win_p, draw_p, loss_p):
    if win_p > 0.55:   return "🟢 Strong home favourite"
    if win_p > 0.42:   return "🟡 Slight home advantage"
    if draw_p > 0.38:  return "🟠 Very even matchup"
    if loss_p > 0.55:  return "🔴 Away team favoured"
    return "⚪ Too close to call"


STAGE_OPTIONS = {
    "Group Stage":   (0, 0),
    "Round of 32":   (1, 1),
    "Round of 16":   (2, 1),
    "Quarter-Final": (3, 1),
    "Semi-Final":    (4, 1),
    "Final":         (5, 1),
}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ WC 2026 Predictor")
    st.markdown("*Powered by XGBoost + Poisson Regression*")
    st.markdown("---")

    # load_bundle() handles all error cases internally — it either returns a
    # valid bundle or calls st.stop(), so no None check is needed here.
    bundle = load_bundle()

    st.markdown("### 📊 Model Performance")
    st.metric("XGBoost 2022 WC Accuracy", f"{bundle['xgb_accuracy']:.1%}")
    st.metric("XGBoost CV Accuracy",       f"{bundle['xgb_cv_accuracy']:.1%}")
    st.metric("Poisson Goal MAE",          f"{bundle['poisson_mae']:.3f} goals")
    st.markdown("---")

    results = load_live_results()
    n_results = len(results)
    st.markdown("### 🔄 Live Feedback Loop")
    if n_results == 0:
        st.info("No results entered yet.\nAdd results in the **⚽ Enter Results** tab.")
    else:
        st.success(f"**{n_results}** match results logged.\nELO + form updating automatically.")

    st.markdown("---")
    st.markdown("""
**How accuracy works:**
- Football is chaotic — 50% is excellent
- 2022: Argentina lost to Saudi Arabia!
- Model improves as 2026 results come in

**Two models together:**
- 🏆 XGBoost → Win / Draw / Loss %
- ⚽ Poisson → Exact scoreline
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

teams_df   = get_static_teams()
team_names = sorted(teams_df['team_name'].tolist())
teams_dict = teams_df.set_index('team_name').to_dict('index')

results_df = load_live_results()
live_elo   = compute_live_elo(results_df)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Match Predictor",
    "⚽ Enter Results",
    "📅 All Fixtures",
    "🌍 Team Stats",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="sec-header">🎯 Predict Any Match</div>', unsafe_allow_html=True)

    if len(results_df) > 0:
        st.markdown(
            f'Using live ELO + form from **{len(results_df)}** completed 2026 matches'
            '<span class="live-badge">LIVE</span>', unsafe_allow_html=True
        )
    else:
        st.caption("Using pre-tournament ELO ratings. Add results in ⚽ Enter Results to enable live updates.")

    st.markdown("")

    col_h, col_vs, col_a = st.columns([5, 1, 5])
    with col_h:
        st.markdown("##### 🏠 Home Team")
        home_name = st.selectbox("Home", team_names,
                                 index=team_names.index("Brazil") if "Brazil" in team_names else 0,
                                 label_visibility="collapsed")
    with col_vs:
        st.markdown("<br><br><div style='text-align:center;font-size:22px;font-weight:900'>VS</div>",
                    unsafe_allow_html=True)
    with col_a:
        st.markdown("##### ✈️ Away Team")
        default_away = team_names.index("Argentina") if "Argentina" in team_names else 1
        away_name = st.selectbox("Away", team_names, index=default_away, label_visibility="collapsed")

    col_stage, col_btn = st.columns([3, 1])
    with col_stage:
        stage_label = st.selectbox("Match Stage", list(STAGE_OPTIONS.keys()))
        stage_code, is_knockout = STAGE_OPTIONS[stage_label]
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("⚽  Predict", type="primary", use_container_width=True)

    if home_name == away_name:
        st.warning("Please choose two different teams.")
    else:
        result = predict_match(home_name, away_name, bundle, stage_code, is_knockout)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"🏠 {home_name} ELO", f"{result['home_elo']:.0f}")
        c2.metric(f"✈️ {away_name} ELO", f"{result['away_elo']:.0f}")
        c3.metric("ELO Difference",      f"{result['home_elo'] - result['away_elo']:+.0f}")
        c4.metric("Stage",               stage_label)

        st.markdown("---")

        left, right = st.columns([3, 2])

        with left:
            st.markdown("### 📊 Outcome Probabilities")
            st.markdown(f"*{verdict(result['win_prob'], result['draw_prob'], result['loss_prob'])}*")
            prob_bar(f"🏠 {home_name} Win", result['win_prob'],  "#27ae60")
            prob_bar("🤝 Draw",              result['draw_prob'], "#e67e22")
            prob_bar(f"✈️ {away_name} Win",  result['loss_prob'], "#e74c3c")

            mp_h = result['home_matches_played']
            mp_a = result['away_matches_played']
            if mp_h > 0 or mp_a > 0:
                st.markdown('<div class="info-card">📈 <strong>Live Form Used</strong><br>'
                    f'{home_name}: {mp_h} match(es) — '
                    f'avg {result["home_form"]["form_gf"]:.1f} GF / {result["home_form"]["form_ga"]:.1f} GA<br>'
                    f'{away_name}: {mp_a} match(es) — '
                    f'avg {result["away_form"]["form_gf"]:.1f} GF / {result["away_form"]["form_ga"]:.1f} GA'
                    '</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🎯 Expected Scoreline (Poisson Model)")

            parts = result['most_likely_score'].split("–")
            st.markdown(f'<div class="score-badge">{parts[0]}  –  {parts[1]}</div>',
                        unsafe_allow_html=True)

            xg1, xg2 = st.columns(2)
            xg1.metric(f"⚽ {home_name} xG", f"{result['lambda_home']:.2f}")
            xg2.metric(f"⚽ {away_name} xG", f"{result['lambda_away']:.2f}")
            st.caption("xG = expected goals (Poisson model's predicted rate)")

        with right:
            st.markdown("### 🏆 Top Scorelines")
            rows_html = ""
            for score, prob in result['top_scores']:
                h, a = score.split("–")
                icon = "🟢" if int(h) > int(a) else ("🟡" if int(h) == int(a) else "🔴")
                rows_html += (f'<div class="score-row">'
                              f'<span style="font-weight:700">{icon} {score}</span>'
                              f'<span style="color:#6c757d">{prob*100:.1f}%</span></div>')
            st.markdown(rows_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Poisson Cross-Check")
            st.caption("Win probabilities derived directly from the scoreline matrix:")
            p1, p2, p3 = st.columns(3)
            p1.metric("Home Win", f"{result['poisson_win']:.1%}")
            p2.metric("Draw",     f"{result['poisson_draw']:.1%}")
            p3.metric("Away Win", f"{result['poisson_loss']:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENTER RESULTS (THE FEEDBACK LOOP TAB)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="sec-header">⚽ Enter 2026 Match Results</div>', unsafe_allow_html=True)
    st.markdown("""
Every result you enter here **automatically updates** ELO ratings and rolling form
for both teams. The next prediction will use these updated values — no retraining needed.
    """)

    col_form, col_hist = st.columns([2, 1])

    with col_form:
        st.markdown("#### Add a Result")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            r_home = st.selectbox("Home Team", team_names, key="r_home")
            r_home_score = st.number_input("Home Goals", min_value=0, max_value=20,
                                           value=0, step=1, key="r_hscore")
        with r_col2:
            r_away_options = [t for t in team_names if t != r_home]
            r_away = st.selectbox("Away Team", r_away_options, key="r_away")
            r_away_score = st.number_input("Away Goals", min_value=0, max_value=20,
                                           value=0, step=1, key="r_ascore")

        r_date = st.date_input("Match Date", value=date.today(), key="r_date")

        if st.button("✅  Save Result", type="primary"):
            if r_home == r_away:
                st.error("Home and away team must be different.")
            else:
                match_id = f"{r_home[:3].upper()}{r_away[:3].upper()}{str(r_date).replace('-','')}"
                save_result(
                    match_id   = match_id,
                    home_team  = r_home,
                    away_team  = r_away,
                    home_score = r_home_score,
                    away_score = r_away_score,
                    date_str   = str(r_date),
                )
                st.success(f"✅  Result saved: **{r_home} {r_home_score}–{r_away_score} {r_away}**")
                st.info("🔄 ELO and form have been updated. Go to Match Predictor to see the effect.")
                st.cache_data.clear()

    with col_hist:
        st.markdown("#### Results Log")
        results_now = load_live_results()
        if results_now.empty:
            st.info("No results entered yet.")
        else:
            display = results_now[['date','home_team','home_score','away_score','away_team']].copy()
            display.columns = ['Date','Home','GH','GA','Away']
            display['Date'] = display['Date'].dt.strftime('%d %b')
            st.dataframe(display, use_container_width=True, height=400)

            if st.button("🗑️  Clear All Results"):
                from predict import RESULTS_FILE as _RF
                if os.path.exists(_RF):
                    os.remove(_RF)
                st.success("All results cleared.")
                st.cache_data.clear()

    results_now = load_live_results()
    if not results_now.empty:
        st.markdown("---")
        st.markdown("#### 📈 ELO Ratings After Live Results")
        live_elo_now = compute_live_elo(results_now)
        elo_df = pd.DataFrame([
            {'Team': k, 'ELO': v} for k, v in live_elo_now.items()
        ]).sort_values('ELO', ascending=False).head(20)
        st.bar_chart(elo_df.set_index('Team')['ELO'], height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALL FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="sec-header">📅 All WC 2026 Fixtures</div>', unsafe_allow_html=True)

    FIXTURES = [
        # ── Group A ──
        {'home': 'Mexico',          'away': 'South Africa',       'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-11'},
        {'home': 'South Korea',     'away': 'Czech Republic',     'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-12'},
        {'home': 'Mexico',          'away': 'South Korea',        'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'Czech Republic',  'away': 'South Africa',       'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Mexico',          'away': 'Czech Republic',     'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'South Africa',    'away': 'South Korea',        'group': 'A', 'stage': 'Group Stage', 'date': '2026-06-22'},
        # ── Group B ──
        {'home': 'Canada',          'away': 'Bosnia Herzegovina',  'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-12'},
        {'home': 'Switzerland',     'away': 'Qatar',               'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-12'},
        {'home': 'Canada',          'away': 'Switzerland',         'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Qatar',           'away': 'Bosnia Herzegovina',  'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Canada',          'away': 'Qatar',               'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'Bosnia Herzegovina','away': 'Switzerland',       'group': 'B', 'stage': 'Group Stage', 'date': '2026-06-22'},
        # ── Group C ──
        {'home': 'Brazil',          'away': 'Haiti',               'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-13'},
        {'home': 'Morocco',         'away': 'Scotland',            'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-13'},
        {'home': 'Brazil',          'away': 'Morocco',             'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-18'},
        {'home': 'Scotland',        'away': 'Haiti',               'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-18'},
        {'home': 'Brazil',          'away': 'Scotland',            'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-23'},
        {'home': 'Haiti',           'away': 'Morocco',             'group': 'C', 'stage': 'Group Stage', 'date': '2026-06-23'},
        # ── Group D ──
        {'home': 'United States',   'away': 'Paraguay',            'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-12'},
        {'home': 'Australia',       'away': 'Turkey',              'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-13'},
        {'home': 'United States',   'away': 'Australia',           'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-18'},
        {'home': 'Turkey',          'away': 'Paraguay',            'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-18'},
        {'home': 'United States',   'away': 'Turkey',              'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-23'},
        {'home': 'Paraguay',        'away': 'Australia',           'group': 'D', 'stage': 'Group Stage', 'date': '2026-06-23'},
        # ── Group E ──
        {'home': 'Germany',         'away': 'Curacao',             'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-14'},
        {'home': 'Ecuador',         'away': 'Ivory Coast',         'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-14'},
        {'home': 'Germany',         'away': 'Ecuador',             'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-19'},
        {'home': 'Ivory Coast',     'away': 'Curacao',             'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-19'},
        {'home': 'Germany',         'away': 'Ivory Coast',         'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-24'},
        {'home': 'Curacao',         'away': 'Ecuador',             'group': 'E', 'stage': 'Group Stage', 'date': '2026-06-24'},
        # ── Group F ──
        {'home': 'Netherlands',     'away': 'Tunisia',             'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-14'},
        {'home': 'Japan',           'away': 'Sweden',              'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-15'},
        {'home': 'Netherlands',     'away': 'Japan',               'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-19'},
        {'home': 'Sweden',          'away': 'Tunisia',             'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-20'},
        {'home': 'Netherlands',     'away': 'Sweden',              'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-25'},
        {'home': 'Tunisia',         'away': 'Japan',               'group': 'F', 'stage': 'Group Stage', 'date': '2026-06-25'},
        # ── Group G ──
        {'home': 'Belgium',         'away': 'Egypt',               'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-15'},
        {'home': 'Iran',            'away': 'New Zealand',         'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-15'},
        {'home': 'Belgium',         'away': 'Iran',                'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-20'},
        {'home': 'New Zealand',     'away': 'Egypt',               'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-20'},
        {'home': 'Belgium',         'away': 'New Zealand',         'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-25'},
        {'home': 'Egypt',           'away': 'Iran',                'group': 'G', 'stage': 'Group Stage', 'date': '2026-06-25'},
        # ── Group H ──
        {'home': 'Spain',           'away': 'Saudi Arabia',        'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-15'},
        {'home': 'Uruguay',         'away': 'Cape Verde',          'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'Spain',           'away': 'Uruguay',             'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-20'},
        {'home': 'Cape Verde',      'away': 'Saudi Arabia',        'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-21'},
        {'home': 'Spain',           'away': 'Cape Verde',          'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-26'},
        {'home': 'Saudi Arabia',    'away': 'Uruguay',             'group': 'H', 'stage': 'Group Stage', 'date': '2026-06-26'},
        # ── Group I ──
        {'home': 'France',          'away': 'DR Congo',            'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'Norway',          'away': 'Senegal',             'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'France',          'away': 'Norway',              'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-21'},
        {'home': 'Senegal',         'away': 'DR Congo',            'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-21'},
        {'home': 'France',          'away': 'Senegal',             'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-26'},
        {'home': 'DR Congo',        'away': 'Norway',              'group': 'I', 'stage': 'Group Stage', 'date': '2026-06-26'},
        # ── Group J ──
        {'home': 'Argentina',       'away': 'Algeria',             'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'Austria',         'away': 'Jordan',              'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Argentina',       'away': 'Austria',             'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-21'},
        {'home': 'Jordan',          'away': 'Algeria',             'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'Argentina',       'away': 'Jordan',              'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-26'},
        {'home': 'Algeria',         'away': 'Austria',             'group': 'J', 'stage': 'Group Stage', 'date': '2026-06-27'},
        # ── Group K ──
        {'home': 'Portugal',        'away': 'IC Playoff Winner',   'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Colombia',        'away': 'Uzbekistan',          'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'Portugal',        'away': 'Colombia',            'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'Uzbekistan',      'away': 'IC Playoff Winner',   'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'Portugal',        'away': 'Uzbekistan',          'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-27'},
        {'home': 'IC Playoff Winner','away': 'Colombia',           'group': 'K', 'stage': 'Group Stage', 'date': '2026-06-27'},
        # ── Group L ──
        {'home': 'England',         'away': 'Ghana',               'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-16'},
        {'home': 'Croatia',         'away': 'Panama',              'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-17'},
        {'home': 'England',         'away': 'Croatia',             'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-21'},
        {'home': 'Panama',          'away': 'Ghana',               'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-22'},
        {'home': 'England',         'away': 'Panama',              'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-26'},
        {'home': 'Ghana',           'away': 'Croatia',             'group': 'L', 'stage': 'Group Stage', 'date': '2026-06-26'},
    ]

    fixtures_df_local = pd.DataFrame(FIXTURES)

    f1, f2 = st.columns(2)
    with f1:
        all_groups = ['All Groups'] + list('ABCDEFGHIJKL')
        grp_filter = st.selectbox("Filter by Group", all_groups)
    with f2:
        status_filter = st.selectbox("Status", ["All", "Upcoming", "Completed"])

    fdf = fixtures_df_local.copy()
    if grp_filter != 'All Groups':
        fdf = fdf[fdf['group'] == grp_filter]

    st.markdown(f"*Showing {len(fdf)} fixtures*")
    st.markdown("---")

    for _, fix in fdf.iterrows():
        h, a = fix['home'], fix['away']
        can_predict = h in teams_dict and a in teams_dict

        label = f"**{h}** vs **{a}**  ·  Group {fix['group']}  ·  {fix['date']}"

        with st.expander(label, expanded=False):
            if can_predict:
                sc, isk = STAGE_OPTIONS["Group Stage"]
                res = predict_match(h, a, bundle, sc, isk)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"🏠 {h} Win", f"{res['win_prob']:.1%}")
                c2.metric("🤝 Draw",      f"{res['draw_prob']:.1%}")
                c3.metric(f"✈️ {a} Win",  f"{res['loss_prob']:.1%}")
                c4.metric("Top Score",    res['most_likely_score'])
                st.caption(f"xG: {h} {res['lambda_home']:.2f} — {res['lambda_away']:.2f} {a}")
            else:
                st.info("One or both teams TBD — prediction not available yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEAM STATS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="sec-header">🌍 All 48 Teams — ELO & FIFA Rankings</div>',
                unsafe_allow_html=True)

    sort_by = st.radio("Sort by",
        ["ELO Rating ↓ (strongest first)", "FIFA Rank ↑ (best first)", "Group A→L"],
        horizontal=True)

    display_df = teams_df.copy()

    if live_elo:
        display_df['live_elo']  = display_df['team_name'].map(live_elo)
        display_df['elo_change'] = (display_df['live_elo'] - display_df['current_elo']).round(1)
        display_df['current_elo'] = display_df['live_elo'].fillna(display_df['current_elo'])

    if "ELO" in sort_by:
        display_df = display_df.sort_values('current_elo', ascending=False)
    elif "FIFA" in sort_by:
        display_df = display_df.sort_values('fifa_rank_2026')
    else:
        display_df = display_df.sort_values(['group', 'current_elo'], ascending=[True, False])

    display_df = display_df.reset_index(drop=True)
    display_df.index += 1

    def elo_tier(elo):
        if elo >= 1850: return "⭐ Elite"
        if elo >= 1700: return "🔵 Strong"
        if elo >= 1550: return "🟡 Competitive"
        if elo >= 1450: return "🟠 Mid-tier"
        return "⚪ Underdog"

    display_df['Tier'] = display_df['current_elo'].apply(elo_tier)

    cols_to_show = ['team_name', 'group', 'current_elo', 'fifa_rank_2026', 'Tier']
    rename_map = {
        'team_name':      'Team',
        'group':          'Group',
        'current_elo':    'ELO Rating',
        'fifa_rank_2026': 'FIFA Rank',
    }

    if 'elo_change' in display_df.columns and len(results_df) > 0:
        cols_to_show.insert(3, 'elo_change')
        rename_map['elo_change'] = 'ELO Δ (2026)'

    st.dataframe(
        display_df[cols_to_show].rename(columns=rename_map),
        use_container_width=True,
        height=620,
    )

    st.markdown("---")
    st.markdown("### ELO Distribution — All 48 Teams")
    chart_df = display_df.sort_values('current_elo', ascending=False).set_index('team_name')
    st.bar_chart(chart_df['current_elo'], height=320)