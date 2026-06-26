import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson

from data_prep import XGBOOST_FEATURES, POISSON_FEATURES, WC2026_TEAMS

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_HERE, 'results_2026.csv')
MODELS_FILE  = os.path.join(_HERE, 'models.joblib')  

ELO_K        = 32
DEFAULT_ELO  = 1500
DEFAULT_FORM = {'form_gf': 1.2, 'form_ga': 1.0, 'form_pts': 1.2}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models(joblib_path=None):
    """Load the trained models bundle from disk using joblib."""
    if joblib_path is None:
        joblib_path = MODELS_FILE
    if not os.path.isabs(joblib_path):
        candidate = os.path.join(_HERE, joblib_path)
        if os.path.exists(candidate):
            joblib_path = candidate
    return joblib.load(joblib_path)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE RESULTS + FEEDBACK LOOP
# ─────────────────────────────────────────────────────────────────────────────

def load_live_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=[
            'match_id', 'home_team', 'away_team',
            'home_score', 'away_score', 'date'
        ])
    df = pd.read_csv(RESULTS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def save_result(match_id, home_team, away_team, home_score, away_score, date_str):
    results = load_live_results()
    results = results[results['match_id'] != match_id]
    new_row = pd.DataFrame([{
        'match_id':   match_id,
        'home_team':  home_team,
        'away_team':  away_team,
        'home_score': int(home_score),
        'away_score': int(away_score),
        'date':       date_str,
    }])
    results = pd.concat([results, new_row], ignore_index=True)
    results = results.sort_values('date').reset_index(drop=True)
    results.to_csv(RESULTS_FILE, index=False)


def compute_live_elo(results_df):
    elo = {t['team_name']: t['current_elo'] for t in WC2026_TEAMS}
    if results_df.empty:
        return elo
    for _, row in results_df.iterrows():
        h, a = row['home_team'], row['away_team']
        elo.setdefault(h, DEFAULT_ELO)
        elo.setdefault(a, DEFAULT_ELO)
        he, ae  = elo[h], elo[a]
        exp_h   = 1 / (1 + 10 ** ((ae - he) / 400))
        exp_a   = 1 - exp_h
        if   row['home_score'] > row['away_score']: act_h, act_a = 1.0, 0.0
        elif row['home_score'] < row['away_score']: act_h, act_a = 0.0, 1.0
        else:                                        act_h, act_a = 0.5, 0.5
        elo[h] = round(he + ELO_K * (act_h - exp_h), 2)
        elo[a] = round(ae + ELO_K * (act_a - exp_a), 2)
    return elo


def compute_live_form(team_name, results_df, n=5):
    if results_df.empty:
        return DEFAULT_FORM.copy()
    records = []
    for _, r in results_df[results_df['home_team'] == team_name].iterrows():
        gf, ga = r['home_score'], r['away_score']
        records.append({'gf': gf, 'ga': ga, 'pts': 3 if gf>ga else (1 if gf==ga else 0), 'date': r['date']})
    for _, r in results_df[results_df['away_team'] == team_name].iterrows():
        gf, ga = r['away_score'], r['home_score']
        records.append({'gf': gf, 'ga': ga, 'pts': 3 if gf>ga else (1 if gf==ga else 0), 'date': r['date']})
    if not records:
        return DEFAULT_FORM.copy()
    records = sorted(records, key=lambda x: x['date'])[-n:]
    return {
        'form_gf':  round(np.mean([r['gf']  for r in records]), 3),
        'form_ga':  round(np.mean([r['ga']  for r in records]), 3),
        'form_pts': round(np.mean([r['pts'] for r in records]), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(home, away, stage_code=0, is_knockout=0):
    h_elo, a_elo   = home.get('current_elo', DEFAULT_ELO), away.get('current_elo', DEFAULT_ELO)
    h_rank, a_rank = home.get('fifa_rank_2026', 50),       away.get('fifa_rank_2026', 50)
    row = {
        'elo_diff':        h_elo - a_elo,
        'home_elo_before': h_elo,
        'away_elo_before': a_elo,
        'home_fifa_rank':  h_rank,
        'away_fifa_rank':  a_rank,
        'fifa_rank_diff':  h_rank - a_rank,
        'has_fifa_rank':   1,
        'home_form_gf':    home.get('form_gf',  DEFAULT_FORM['form_gf']),
        'home_form_ga':    home.get('form_ga',  DEFAULT_FORM['form_ga']),
        'home_form_pts':   home.get('form_pts', DEFAULT_FORM['form_pts']),
        'away_form_gf':    away.get('form_gf',  DEFAULT_FORM['form_gf']),
        'away_form_ga':    away.get('form_ga',  DEFAULT_FORM['form_ga']),
        'away_form_pts':   away.get('form_pts', DEFAULT_FORM['form_pts']),
        'stage_code':      stage_code,
        'is_knockout':     is_knockout,
    }
    return pd.DataFrame([row])[XGBOOST_FEATURES]


def build_poisson_feature_rows(home, away, stage_code=0, is_knockout=0):
    h_elo, a_elo   = home.get('current_elo', DEFAULT_ELO), away.get('current_elo', DEFAULT_ELO)
    h_rank, a_rank = home.get('fifa_rank_2026', 50),       away.get('fifa_rank_2026', 50)
    base = {'is_knockout': is_knockout, 'stage_code': stage_code}
    home_row = pd.DataFrame([{**base,
        'elo_diff': h_elo-a_elo, 'team_elo': h_elo, 'opp_elo': a_elo,
        'team_form_gf': home.get('form_gf', DEFAULT_FORM['form_gf']),
        'team_form_ga': home.get('form_ga', DEFAULT_FORM['form_ga']),
        'team_form_pts':home.get('form_pts',DEFAULT_FORM['form_pts']),
        'opp_form_gf':  away.get('form_gf', DEFAULT_FORM['form_gf']),
        'opp_form_ga':  away.get('form_ga', DEFAULT_FORM['form_ga']),
        'opp_form_pts': away.get('form_pts',DEFAULT_FORM['form_pts']),
        'team_fifa_rank': h_rank, 'opp_fifa_rank': a_rank, 'is_home': 1,
    }])[POISSON_FEATURES]
    away_row = pd.DataFrame([{**base,
        'elo_diff': a_elo-h_elo, 'team_elo': a_elo, 'opp_elo': h_elo,
        'team_form_gf': away.get('form_gf', DEFAULT_FORM['form_gf']),
        'team_form_ga': away.get('form_ga', DEFAULT_FORM['form_ga']),
        'team_form_pts':away.get('form_pts',DEFAULT_FORM['form_pts']),
        'opp_form_gf':  home.get('form_gf', DEFAULT_FORM['form_gf']),
        'opp_form_ga':  home.get('form_ga', DEFAULT_FORM['form_ga']),
        'opp_form_pts': home.get('form_pts',DEFAULT_FORM['form_pts']),
        'team_fifa_rank': a_rank, 'opp_fifa_rank': h_rank, 'is_home': 0,
    }])[POISSON_FEATURES]
    return home_row, away_row


# ─────────────────────────────────────────────────────────────────────────────
# SCORELINE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def scoreline_matrix(lambda_home, lambda_away, max_goals=6):
    goals      = np.arange(0, max_goals + 1)
    home_probs = poisson.pmf(goals, lambda_home)
    away_probs = poisson.pmf(goals, lambda_away)
    return np.outer(home_probs, away_probs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(home_name, away_name, bundle, stage_code=0, is_knockout=0):
    results  = load_live_results()
    live_elo = compute_live_elo(results)

    teams_static = {t['team_name']: t for t in WC2026_TEAMS}
    home_static  = teams_static.get(home_name, {})
    away_static  = teams_static.get(away_name, {})

    home_form = compute_live_form(home_name, results)
    away_form = compute_live_form(away_name, results)

    home = {
        'current_elo':    live_elo.get(home_name, home_static.get('current_elo', DEFAULT_ELO)),
        'fifa_rank_2026': home_static.get('fifa_rank_2026', 50),
        **home_form,
    }
    away = {
        'current_elo':    live_elo.get(away_name, away_static.get('current_elo', DEFAULT_ELO)),
        'fifa_rank_2026': away_static.get('fifa_rank_2026', 50),
        **away_form,
    }

    home_results = results[(results['home_team']==home_name)|(results['away_team']==home_name)]
    away_results = results[(results['home_team']==away_name)|(results['away_team']==away_name)]

    # XGBoost
    X          = build_feature_row(home, away, stage_code, is_knockout)
    raw_proba  = bundle['xgb_model'].predict_proba(X)[0]
    train_marg = bundle.get('train_marginals', np.array([0.242, 0.188, 0.570]))
    model_marg = np.array([0.319, 0.196, 0.485])
    corrected  = raw_proba * train_marg / model_marg
    corrected /= corrected.sum()

    classes   = list(bundle['label_encoder'].classes_)
    win_prob  = float(corrected[classes.index('home_win')])
    draw_prob = float(corrected[classes.index('draw')])
    loss_prob = float(corrected[classes.index('away_win')])

    # Poisson
    home_pf, away_pf = build_poisson_feature_rows(home, away, stage_code, is_knockout)
    lambda_home = max(0.3, min(float(bundle['home_poisson'].predict(home_pf)[0]), 6.0))
    lambda_away = max(0.3, min(float(bundle['away_poisson'].predict(away_pf)[0]), 6.0))

    matrix       = scoreline_matrix(lambda_home, lambda_away)
    poisson_win  = float(np.sum(np.tril(matrix, -1)))
    poisson_draw = float(np.sum(np.diag(matrix)))
    poisson_loss = float(np.sum(np.triu(matrix, 1)))

    max_g  = matrix.shape[0]
    scores = sorted(
        [(f"{i}–{j}", float(matrix[i][j])) for i in range(max_g) for j in range(max_g)],
        key=lambda x: x[1], reverse=True
    )

    return {
        'win_prob': win_prob, 'draw_prob': draw_prob, 'loss_prob': loss_prob,
        'lambda_home': lambda_home, 'lambda_away': lambda_away,
        'most_likely_score': scores[0][0], 'top_scores': scores[:8],
        'poisson_win': poisson_win, 'poisson_draw': poisson_draw, 'poisson_loss': poisson_loss,
        'home_elo': home['current_elo'], 'away_elo': away['current_elo'],
        'home_form': home_form, 'away_form': away_form,
        'home_matches_played': len(home_results),
        'away_matches_played': len(away_results),
    }