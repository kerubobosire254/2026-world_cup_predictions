import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LISTS  
# ─────────────────────────────────────────────────────────────────────────────

# XGBoost reads these 15 columns and predicts Win / Draw / Loss
XGBOOST_FEATURES = [
    'elo_diff',           # home ELO minus away ELO. Positive = home is stronger.
    'home_elo_before',    # raw home team ELO (global strength signal)
    'away_elo_before',    # raw away team ELO
    'home_fifa_rank',     # FIFA rank home (1 = best in world, lower number = stronger)
    'away_fifa_rank',     # FIFA rank away
    'fifa_rank_diff',     # home rank minus away rank. Negative = home is better ranked.
    'has_fifa_rank',      # 1 if match year >= 1998, 0 if pre-1998 (no rankings existed)
    'home_form_gf',       # home team's avg goals scored in last 5 WC matches
    'home_form_ga',       # home team's avg goals conceded
    'home_form_pts',      # home team's avg points (W=3, D=1, L=0)
    'away_form_gf',
    'away_form_ga',
    'away_form_pts',
    'stage_code',         # 0=group, 1=R16/R32, 2=QF, 3=SF/3rd place, 4=Final
    'is_knockout',        # binary: 0 = group stage, 1 = any knockout match
]

# Poisson reads these 14 columns and predicts how many goals ONE TEAM scores.
# We call it twice per match — once from home perspective, once from away.
POISSON_FEATURES = [
    'elo_diff',           # from scoring team's perspective (flipped for away)
    'team_elo',           # scoring team's ELO
    'opp_elo',            # opponent's ELO
    'team_form_gf',       # scoring team's avg goals in last 5 WC matches
    'team_form_ga',       # scoring team's avg goals conceded
    'team_form_pts',
    'opp_form_gf',
    'opp_form_ga',
    'opp_form_pts',
    'team_fifa_rank',
    'opp_fifa_rank',
    'is_knockout',
    'stage_code',
    'is_home',            # 1 = home team, 0 = away. Home teams score ~30% more goals.
]


# ─────────────────────────────────────────────────────────────────────────────
# ALL 48 WC 2026 TEAMS
# ─────────────────────────────────────────────────────────────────────────────

WC2026_TEAMS = [
    # ── GROUP A ──────────────────────────────────────────────────────────────
    {'team_name': 'Mexico',           'group': 'A', 'fifa_rank_2026': 15, 'current_elo': 1620},
    {'team_name': 'South Africa',     'group': 'A', 'fifa_rank_2026': 59, 'current_elo': 1430},
    {'team_name': 'South Korea',      'group': 'A', 'fifa_rank_2026': 23, 'current_elo': 1540},
    {'team_name': 'Czech Republic',   'group': 'A', 'fifa_rank_2026': 38, 'current_elo': 1510},
    # ── GROUP B ──────────────────────────────────────────────────────────────
    {'team_name': 'Canada',           'group': 'B', 'fifa_rank_2026': 27, 'current_elo': 1555},
    {'team_name': 'Switzerland',      'group': 'B', 'fifa_rank_2026': 19, 'current_elo': 1610},
    {'team_name': 'Qatar',            'group': 'B', 'fifa_rank_2026': 42, 'current_elo': 1440},
    {'team_name': 'Bosnia Herzegovina', 'group': 'B', 'fifa_rank_2026': 65, 'current_elo': 1490},
    # ── GROUP C ──────────────────────────────────────────────────────────────
    {'team_name': 'Brazil',           'group': 'C', 'fifa_rank_2026': 5,  'current_elo': 1820},
    {'team_name': 'Morocco',          'group': 'C', 'fifa_rank_2026': 9,  'current_elo': 1690},
    {'team_name': 'Scotland',         'group': 'C', 'fifa_rank_2026': 39, 'current_elo': 1530},
    {'team_name': 'Haiti',            'group': 'C', 'fifa_rank_2026': 78, 'current_elo': 1380},
    # ── GROUP D ──────────────────────────────────────────────────────────────
    {'team_name': 'United States',    'group': 'D', 'fifa_rank_2026': 11, 'current_elo': 1640},
    {'team_name': 'Paraguay',         'group': 'D', 'fifa_rank_2026': 43, 'current_elo': 1490},
    {'team_name': 'Australia',        'group': 'D', 'fifa_rank_2026': 25, 'current_elo': 1545},
    {'team_name': 'Turkey',           'group': 'D', 'fifa_rank_2026': 32, 'current_elo': 1540},
    # ── GROUP E ──────────────────────────────────────────────────────────────
    {'team_name': 'Germany',          'group': 'E', 'fifa_rank_2026': 12, 'current_elo': 1740},
    {'team_name': 'Ecuador',          'group': 'E', 'fifa_rank_2026': 21, 'current_elo': 1565},
    {'team_name': 'Ivory Coast',      'group': 'E', 'fifa_rank_2026': 53, 'current_elo': 1470},
    {'team_name': 'Curacao',          'group': 'E', 'fifa_rank_2026': 82, 'current_elo': 1350},
    # ── GROUP F ──────────────────────────────────────────────────────────────
    {'team_name': 'Netherlands',      'group': 'F', 'fifa_rank_2026': 7,  'current_elo': 1760},
    {'team_name': 'Japan',            'group': 'F', 'fifa_rank_2026': 14, 'current_elo': 1660},
    {'team_name': 'Tunisia',          'group': 'F', 'fifa_rank_2026': 29, 'current_elo': 1510},
    {'team_name': 'Sweden',           'group': 'F', 'fifa_rank_2026': 20, 'current_elo': 1570},
    # ── GROUP G ──────────────────────────────────────────────────────────────
    {'team_name': 'Belgium',          'group': 'G', 'fifa_rank_2026': 4,  'current_elo': 1780},
    {'team_name': 'Iran',             'group': 'G', 'fifa_rank_2026': 24, 'current_elo': 1510},
    {'team_name': 'New Zealand',      'group': 'G', 'fifa_rank_2026': 96, 'current_elo': 1390},
    {'team_name': 'Egypt',            'group': 'G', 'fifa_rank_2026': 34, 'current_elo': 1490},
    # ── GROUP H ──────────────────────────────────────────────────────────────
    {'team_name': 'Spain',            'group': 'H', 'fifa_rank_2026': 8,  'current_elo': 1860},
    {'team_name': 'Uruguay',          'group': 'H', 'fifa_rank_2026': 18, 'current_elo': 1660},
    {'team_name': 'Saudi Arabia',     'group': 'H', 'fifa_rank_2026': 47, 'current_elo': 1450},
    {'team_name': 'Cape Verde',       'group': 'H', 'fifa_rank_2026': 72, 'current_elo': 1390},
    # ── GROUP I ──────────────────────────────────────────────────────────────
    {'team_name': 'France',           'group': 'I', 'fifa_rank_2026': 2,  'current_elo': 1880},
    {'team_name': 'Norway',           'group': 'I', 'fifa_rank_2026': 28, 'current_elo': 1570},
    {'team_name': 'Senegal',          'group': 'I', 'fifa_rank_2026': 17, 'current_elo': 1600},
    {'team_name': 'DR Congo',         'group': 'I', 'fifa_rank_2026': 60, 'current_elo': 1400},
    # ── GROUP J ──────────────────────────────────────────────────────────────
    {'team_name': 'Argentina',        'group': 'J', 'fifa_rank_2026': 1,  'current_elo': 1980},
    {'team_name': 'Austria',          'group': 'J', 'fifa_rank_2026': 22, 'current_elo': 1580},
    {'team_name': 'Algeria',          'group': 'J', 'fifa_rank_2026': 41, 'current_elo': 1510},
    {'team_name': 'Jordan',           'group': 'J', 'fifa_rank_2026': 68, 'current_elo': 1380},
    # ── GROUP K ──────────────────────────────────────────────────────────────
    {'team_name': 'Portugal',         'group': 'K', 'fifa_rank_2026': 6,  'current_elo': 1820},
    {'team_name': 'Colombia',         'group': 'K', 'fifa_rank_2026': 16, 'current_elo': 1630},
    {'team_name': 'Uzbekistan',       'group': 'K', 'fifa_rank_2026': 74, 'current_elo': 1370},
    {'team_name': 'IC Playoff Winner','group': 'K', 'fifa_rank_2026': 70, 'current_elo': 1400},
    # ── GROUP L ──────────────────────────────────────────────────────────────
    {'team_name': 'England',          'group': 'L', 'fifa_rank_2026': 3,  'current_elo': 1840},
    {'team_name': 'Croatia',          'group': 'L', 'fifa_rank_2026': 13, 'current_elo': 1670},
    {'team_name': 'Panama',           'group': 'L', 'fifa_rank_2026': 51, 'current_elo': 1440},
    {'team_name': 'Ghana',            'group': 'L', 'fifa_rank_2026': 46, 'current_elo': 1480},
]


def get_teams_df():
    """Return the 48-team dataframe."""
    return pd.DataFrame(WC2026_TEAMS)


# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION  — works locally AND on Streamlit Cloud
# ─────────────────────────────────────────────────────────────────────────────

def _find_csv(csv_path: str) -> str:

    # 1. As given
    if os.path.exists(csv_path):
        return csv_path

    # 2. Relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, csv_path)
    if os.path.exists(candidate):
        return candidate

    # 3. Try stripping directory components — look for the filename anywhere
    #    under the repo root (one level above this file)
    filename = os.path.basename(csv_path)
    repo_root = os.path.dirname(here)
    for root, _dirs, files in os.walk(repo_root):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"Cannot find '{csv_path}'.\n"
        f"Searched: (1) cwd, (2) {here}, (3) recursive under {repo_root}.\n"
        f"Make sure the CSV is committed to your repo inside a 'data/' folder."
    )


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA LOADING + CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path='data/master_historical_features (1).csv'):
    """
    Load the historical dataset, fill missing values, encode the outcome label.

    WHY TIME SPLIT (not random):
      We split train/test by date — everything before 2022 trains the model,
      the 2022 WC is the test set. This mirrors real life: you always predict
      the future using the past. A random split would let the model train on
      2022 data and test on 1998 data — that's dishonest.

    Returns: df, train, test, label_encoder
    """
    resolved = _find_csv(csv_path)
    df = pd.read_csv(resolved)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Fill NaN FIFA ranks with 50 (neutral).
    df['home_fifa_rank'] = df['home_fifa_rank'].fillna(50)
    df['away_fifa_rank'] = df['away_fifa_rank'].fillna(50)
    df['fifa_rank_diff'] = df['fifa_rank_diff'].fillna(0)

    form_cols = [
        'home_form_gf', 'home_form_ga', 'home_form_pts',
        'away_form_gf', 'away_form_ga', 'away_form_pts',
    ]
    df[form_cols] = df[form_cols].fillna(0)

    le = LabelEncoder()
    df['outcome_encoded'] = le.fit_transform(df['outcome'])

    train = df[df['match_date'] < '2022-01-01'].copy()
    test  = df[df['match_date'] >= '2022-01-01'].copy()

    return df, train, test, le


# ─────────────────────────────────────────────────────────────────────────────
# POISSON ROW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_poisson_rows(df):
    """
    Convert one-row-per-match → two-rows-per-match for Poisson training.

    WHY TWO ROWS:
      Poisson predicts "how many goals does THIS team score against THAT opponent?"
      One model handles both home and away teams — the 'is_home' feature (1 or 0)
      teaches it that home teams score roughly 30% more goals on average.
    """
    rows = []
    for _, r in df.iterrows():
        base = {'is_knockout': r['is_knockout'], 'stage_code': r['stage_code']}

        # ── Home team row ──
        rows.append({**base,
            'elo_diff':       r['elo_diff'],
            'team_elo':       r['home_elo_before'],
            'opp_elo':        r['away_elo_before'],
            'team_form_gf':   r['home_form_gf'],
            'team_form_ga':   r['home_form_ga'],
            'team_form_pts':  r['home_form_pts'],
            'opp_form_gf':    r['away_form_gf'],
            'opp_form_ga':    r['away_form_ga'],
            'opp_form_pts':   r['away_form_pts'],
            'team_fifa_rank': r['home_fifa_rank'],
            'opp_fifa_rank':  r['away_fifa_rank'],
            'is_home':        1,
            'goals':          r['home_team_score'],
        })

        # ── Away team row ──
        rows.append({**base,
            'elo_diff':       -r['elo_diff'],    # flip! now from away's perspective
            'team_elo':       r['away_elo_before'],
            'opp_elo':        r['home_elo_before'],
            'team_form_gf':   r['away_form_gf'],
            'team_form_ga':   r['away_form_ga'],
            'team_form_pts':  r['away_form_pts'],
            'opp_form_gf':    r['home_form_gf'],
            'opp_form_ga':    r['home_form_ga'],
            'opp_form_pts':   r['home_form_pts'],
            'team_fifa_rank': r['away_fifa_rank'],
            'opp_fifa_rank':  r['home_fifa_rank'],
            'is_home':        0,
            'goals':          r['away_team_score'],
        })

    return pd.DataFrame(rows)