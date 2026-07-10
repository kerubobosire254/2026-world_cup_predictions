# ⚽ World Cup 2026 Match Predictor

*Because watching football is more fun when you've already argued about the result before kickoff.*

---
## Live Demo : https://2026-worldcuppredictions-msqvpuk2xsy4w8zsuo7taz.streamlit.app/
## App snippet

<img width="1351" height="572" alt="image" src="https://github.com/user-attachments/assets/03c75173-c225-4cec-a565-58bbc361cd08" />

<div align="center">

# ⚽ World Cup 2026 Match Predictor

*Because watching football is more fun when you've already argued about the result before kickoff.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-calibrated-EB5B25)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-live%20demo-FF4B4B?logo=streamlit&logoColor=white)](https://2026-worldcuppredictions-msqvpuk2xsy4w8zsuo7taz.streamlit.app/)
[![License](https://img.shields.io/badge/license-MIT-blue)]()

**[Live Demo →](https://2026-worldcuppredictions-msqvpuk2xsy4w8zsuo7taz.streamlit.app/)**

</div>

## App Snippet

[*(keep your existing screenshot embed here)*]

## The Problem

Every four years, millions of people fill out tournament brackets with complete confidence. And every four years, Argentina loses to Saudi Arabia and ruins everything.

The issue isn't that upsets happen, it's that most prediction tools either pretend they don't, or hide behind "football is unpredictable" as an excuse not to try. Neither is honest.

I wanted something in the middle: a system that gives real, calibrated probabilities (not fake certainty, not a shrug), and one that gets smarter as the tournament plays out rather than repeating matchday-1 predictions on matchday 3.

## What It Does

Drop in any two of the 48 WC 2026 teams, pick the match stage, and the app gives you:

* **Win / Draw / Loss probabilities** from a calibrated XGBoost model trained on 964 World Cup matches going back to 1930
* **Expected goals (xG)** for each team: an actual predicted rate like 1.78 vs 0.40, not just "Brazil will probably score more"
* **A full scoreline probability table**, every result from 0-0 to 6-6, ranked by likelihood. You can see a Brazil 2-0 is more likely than a 1-1 before you put money on it
* **A live feedback loop**: enter real results as the tournament plays and the system recalculates ELO ratings and form stats on the fly, so the next prediction uses what a team actually did, not what we guessed

There are also tabs to browse all 72 scheduled fixtures with predictions attached, and a full team stats board ranked by ELO, FIFA rank, or group.

## Under the Hood

Two Teams ⚽ → Feature Engineering → ┬→ XGBoost (Win/Draw/Loss)
└→ Poisson (Scoreline Grid)
↓
Combined Prediction
↓
Live Result Entered by User
↓
ELO + Form Recalculated From Scratch
↓
Next Prediction Updates

Two models run in parallel for every prediction, working independently so you can compare their outputs, which is half the fun.

**XGBoost** handles win/draw/loss. It learns from 15 features per match: ELO ratings, the gap between them, FIFA rankings, recent form (goals scored, goals conceded, points per game over the last 5 WC appearances), match stage, and whether it's a knockout game. Trained on 1930–2018 data, tested blind on the 2022 World Cup.

The trickiest part was class imbalance. Home wins make up 57% of historical matches, so a lazy model learns to shout "home win!" every time and gets 57% accuracy while being completely useless. Sample weighting hurt confidence on genuine home favourites, so the cleaner fix was prior correction at inference time: let the model learn the real signal during training, then rescale the output probabilities to match actual base rates. Now it genuinely predicts all three outcomes.

**Poisson Regression** handles the scoreline. Goals are rare, discrete, non-negative events, basically the textbook definition of what Poisson distributions were built for. Given expected goals λ, `P(k goals) = e^(-λ) × λ^k / k!`. I trained two separate models, one for home goals and one for away, rather than a single combined model, because home and away dynamics are structurally different enough that specialising helps. The scoreline matrix multiplies every home probability by every away probability for the full picture.

## How the Numbers Look

| What | Score | The honest take |
|---|---|---|
| XGBoost cross-validation (5-fold) | **58%** | Solid. Unseen historical data, all three outcomes predicted |
| XGBoost on 2022 World Cup | **45.3%** | Honest. Japan beat Germany AND Spain. Argentina lost to Saudi Arabia. No model called those |
| Poisson home goals MAE | **1.21 goals** | Within a goal and a bit on average |
| Poisson away goals MAE | **0.86 goals** | Away teams score less, so this is easier to predict |

For context: FiveThirtyEight, with a much larger team and more data, got similar numbers on tournament football. The 58% cross-val is the figure I'd lean on; 45% on 64 matches swings heavily on a handful of upsets, and the 2022 WC had more than its share.

## The Bit That Actually Interested Me

The feedback loop was the most interesting engineering problem. After a team plays a match, two things update: their ELO (a rolling rating based on result and opponent strength) and their form stats (goals scored/conceded/points over the last 5 WC appearances). Both feed directly into the next prediction.

The naive approach (update values in a CSV and reload) breaks as soon as ordering issues or missing results show up. Instead, `compute_live_elo()` recalculates every team's ELO from scratch using the full match history each time a new result comes in. Slower, but always correct. Same principle for form. After entering Brazil 3-0 in their opener, the Brazil vs Morocco prediction immediately uses Brazil's updated ELO and new form average, not their pre-tournament estimate.

## Challenges

**The class imbalance was genuinely annoying.** Getting from "the model just predicts home win every time" to "the model predicts all three outcomes meaningfully" took three approaches before prior correction clicked. Not glamorous, but explainable, and that matters.

**64 test matches is a small number.** The 2022 WC result is honest but noisy. A slightly weaker model that got lucky on three key upsets would have scored higher. Cross-validation is the more trustworthy signal.

**The feedback loop relies on humans entering results.** In production this would hit a live football API. Right now, forgetting to enter a result lets the form data quietly drift from reality.

**Pre-1998 data has no FIFA rankings.** That's 45% of the training set. Handled with a neutral fill and a flag column, but those features are effectively decorative for older matches. ELO and form do the heavy lifting for the 1930–1994 era.

## What I'd Do Next

* Pull live scores from a football API so the feedback loop runs itself
* Monte Carlo simulation of the full bracket: run 10,000 tournaments and report winner probabilities for every team, not just individual match odds
* A penalty shootout model for knockout games that end level; right now a draw is a draw and the model doesn't know what happens next
* Better form data using qualifying matches and friendlies, not just World Cup appearances, for richer recent context on newer teams

## Stack

Python · XGBoost · scikit-learn · SciPy · Pandas · Streamlit

## Run It

```bash
pip install -r requirements.txt
python models.py       # trains both models, saves to data/models.pkl (~30 seconds)
streamlit run app.py   # launches the app
```

<div align="center">

*96 years of data. Two models. One scoreline table. It still won't predict the next Saudi Arabia upset, but at least it'll tell you the odds.*

**Kerubo Bosire**
Actuarial Science · Risk Analytics · Machine Learning
[GitHub](https://github.com/kerubobosire254) · [LinkedIn](https://linkedin.com/in/kerubo-bosire-364523283)

</div>
