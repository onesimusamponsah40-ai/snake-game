#!/usr/bin/env python3
"""
Ledgerline ML: a beginner's stock predictor
===========================================

The question we train a computer to answer:

    "Will this stock beat the overall market over the next 5 trading days?"

We compare against the market (the S&P 500, ticker SPY) instead of asking
"will it go up?", because most stocks rise when the whole market rises.
Beating the market is the part that takes skill.

The 6 steps of almost every machine learning project:

    1. Collect data      -> daily prices for ~40 big US companies
    2. Make features     -> turn prices into clues (momentum, trend, RSI...)
    3. Make labels       -> the right answer for each example (beat SPY? 1/0)
    4. Split the data    -> train on the past, test on the most recent period
    5. Train and test    -> fit models, measure them against a dumb baseline
    6. Predict           -> score every stock for next week

Run it:
    python3 stock_predictor.py          # real prices (needs network access)
    python3 stock_predictor.py --demo   # fake random prices, works offline

Needs: pip install pandas numpy scikit-learn
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ~40 large companies from different industries, plus SPY (the market).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM", "ADBE",  # tech
    "JPM", "BAC", "V", "MA", "GS",                                                   # finance
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",                                       # health
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "HD",                            # consumer
    "XOM", "CVX",                                                                    # energy
    "CAT", "GE", "HON", "UPS", "BA",                                                 # industrial
    "TSLA", "DIS", "NFLX", "T",                                                      # other
]
MARKET = "SPY"
HORIZON = 5          # predict 5 trading days (one week) ahead
FEATURES = [
    "ret_5", "ret_20", "ret_60",        # momentum: % change over 1 week, 1 month, 3 months
    "rel_20",                           # 1-month return minus the market's
    "dist_sma50", "dist_sma200",        # how far price is above/below its trend lines
    "rsi14",                            # overbought (>70) / oversold (<30) meter
    "vol_20",                           # how jumpy the stock has been
    "volume_ratio",                     # trading volume vs. its normal level
    "mkt_ret_20",                       # how the whole market did last month
]


def say(title, text=""):
    """Print a step heading so the output reads like a lesson."""
    print(f"\n=== {title} ===")
    if text:
        print(text)


# --------------------------------------------------------------------------- 1. data

def fetch_yahoo(ticker):
    """Download 5 years of daily prices from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5y&interval=1d"
    cafile = "/root/.ccr/ca-bundle.crt" if os.path.exists("/root/.ccr/ca-bundle.crt") else None
    ctx = ssl.create_default_context(cafile=cafile)
    req = urllib.request.Request(url, headers={"User-Agent": "Ledgerline ML learning project"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": adj,              # adjusted close: accounts for splits and dividends
        "volume": q["volume"],
    }).dropna()
    return df.drop_duplicates("date").set_index("date")


def load_prices(ticker, demo, refresh):
    """Real prices (cached to data/<TICKER>.csv) or fake ones in demo mode."""
    if demo:
        return fake_prices(ticker)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path) and not refresh and time.time() - os.path.getmtime(path) < 12 * 3600:
        return pd.read_csv(path, index_col="date", parse_dates=True)
    df = fetch_yahoo(ticker)
    df.to_csv(path)
    time.sleep(0.2)
    return df


def fake_prices(ticker, days=1260):
    """A 'random walk': each day's move is pure chance. There is NO pattern to find.

    Useful for learning: a good test setup should show the model can't beat a
    coin flip here. If it did, our test would be broken (it would be cheating)."""
    rng = np.random.default_rng(sum(ord(ch) * 31 ** i for i, ch in enumerate(ticker)) % 2**32)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    daily = rng.normal(0.0004, 0.017, days)                  # ~10%/yr drift, normal volatility
    close = 100 * np.exp(np.cumsum(daily))
    volume = rng.lognormal(15, 0.3, days)
    return pd.DataFrame({"close": close, "volume": volume}, index=dates).rename_axis("date")


# --------------------------------------------------------------------------- 2. features

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def make_features(df, market):
    """Turn a price history into clues the model can learn from.

    Every feature only uses data up to that day. Using future data here
    ("look-ahead") is the #1 way beginners fool themselves."""
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    m = market["close"].reindex(df.index).ffill()
    f["ret_5"] = c.pct_change(5)
    f["ret_20"] = c.pct_change(20)
    f["ret_60"] = c.pct_change(60)
    f["mkt_ret_20"] = m.pct_change(20)
    f["rel_20"] = f["ret_20"] - f["mkt_ret_20"]
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    f["rsi14"] = rsi(c)
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # ---------------------------------------------------------------- 3. labels
    # The answer we want to predict: did the stock beat SPY over the NEXT 5 days?
    fwd = c.shift(-HORIZON) / c - 1
    fwd_mkt = m.shift(-HORIZON) / m - 1
    f["fwd_excess"] = fwd - fwd_mkt                 # kept for the backtest
    f["label"] = (f["fwd_excess"] > 0).astype(float)
    f.loc[f["fwd_excess"].isna(), "label"] = np.nan  # the last 5 days have no answer yet
    return f


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Train a model to pick stocks that beat the market next week.")
    ap.add_argument("--demo", action="store_true", help="use fake random-walk prices (works offline)")
    ap.add_argument("--refresh", action="store_true", help="re-download prices even if cached")
    ap.add_argument("--out", default=os.path.join(HERE, "predictions.json"))
    args = ap.parse_args()

    say("STEP 1: Collect data",
        "Fake random prices (demo mode). Nothing real to learn here, on purpose." if args.demo
        else "Downloading 5 years of daily prices from Yahoo Finance...")
    try:
        market = load_prices(MARKET, args.demo, args.refresh)
    except Exception as e:
        sys.exit(f"Could not download prices ({e}).\n"
                 "Allow query1.finance.yahoo.com in the environment's network settings, "
                 "or run with --demo to practice on fake data.")
    frames, failed = [], []
    for t in UNIVERSE:
        try:
            feats = make_features(load_prices(t, args.demo, args.refresh), market)
            feats["ticker"] = t
            frames.append(feats)
        except Exception as e:
            failed.append(f"{t} ({e})")
    if failed:
        print("Skipped:", ", ".join(failed))
    data = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    print(f"Got {len(frames)} stocks, {data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}.")

    say("STEP 2-3: Features and labels",
        f"Each example = one stock on one day. Clues: {', '.join(FEATURES)}.\n"
        f"Answer: 1 if it beat {MARKET} over the next {HORIZON} days, else 0.")
    usable = data.dropna(subset=FEATURES)
    labeled = usable.dropna(subset=["label"])
    # Keep one example per stock per week, so examples don't overlap and repeat each other.
    weeks = sorted(labeled["date"].unique())[::HORIZON]
    labeled = labeled[labeled["date"].isin(weeks)]
    print(f"{len(labeled):,} labeled examples. {labeled['label'].mean():.1%} of them beat the market.")

    say("STEP 4: Split by time",
        "Train on the older 75%, test on the newest 25%. Never shuffle stock data randomly:\n"
        "that lets the model peek at the future and makes results look far better than reality.")
    dates = sorted(labeled["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    gap = pd.Timedelta(days=10)                      # buffer so 5-day answers can't leak across
    train = labeled[labeled["date"] < cut - gap]
    test = labeled[labeled["date"] >= cut]
    print(f"Train: {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d} ({len(train):,} examples)")
    print(f"Test:  {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d} ({len(test):,} examples)")

    say("STEP 5: Train and test",
        "We compare 2 real models against a 'dumb' baseline. A model is only useful if it beats the baseline.")
    Xtr, ytr, Xte, yte = train[FEATURES], train["label"], test[FEATURES], test["label"]
    baseline_guess = float(ytr.mean() >= 0.5)
    models = {
        "Logistic regression (a weighted sum of the clues)":
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random forest (300 small decision trees voting)":
            RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, random_state=0, n_jobs=-1),
    }
    print(f"{'Baseline (always guess the most common answer)':52s} accuracy {np.mean(yte == baseline_guess):.1%}")
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        acc = np.mean((prob >= 0.5) == yte)
        results[name] = (model, prob)
        print(f"{name:52s} accuracy {acc:.1%}")

    # Accuracy isn't everything. What matters for an investor: if each week we bought
    # the 5 stocks the model liked most, did they beat the market?
    say("Backtest: buy the model's top 5 picks every week (test period only)")
    t = test.copy()
    rows = []
    for name, (_, prob) in results.items():
        t["prob"] = prob
        top = t.sort_values("prob", ascending=False).groupby("date").head(5)
        bottom = t.sort_values("prob").groupby("date").head(5)
        top_wk, bot_wk = top.groupby("date")["fwd_excess"].mean(), bottom.groupby("date")["fwd_excess"].mean()
        rows.append((name, top_wk.mean() * 100, (top_wk > 0).mean() * 100, bot_wk.mean() * 100, len(top_wk)))
    all_wk = t.groupby("date")["fwd_excess"].mean()
    print(f"{'':52s} {'top 5 vs SPY':>13s} {'weeks won':>10s} {'bottom 5':>9s}")
    for name, top_avg, won, bot_avg, n in rows:
        print(f"{name[:52]:52s} {top_avg:+12.2f}% {won:9.0f}% {bot_avg:+8.2f}%")
    print(f"{'Buying all 40 stocks equally':52s} {all_wk.mean() * 100:+12.2f}%")
    print(f"(average per week over {rows[0][4]} test weeks; 'weeks won' = top 5 beat SPY that week)")

    best_name = max(rows, key=lambda r: r[1])[0]
    rf = results["Random forest (300 small decision trees voting)"][0]
    say("What the random forest found most useful",
        "\n".join(f"  {f:14s} {'#' * int(w * 100)} {w:.0%}"
                  for f, w in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])))

    say("STEP 6: Predict next week",
        f"Retrain the best model ({best_name.split(' (')[0]}) on ALL labeled history, then score today's data.")
    final = models[best_name]
    final.fit(labeled[FEATURES], labeled["label"])
    latest = usable[usable["date"] == usable["date"].max()].copy()
    latest["prob"] = final.predict_proba(latest[FEATURES])[:, 1]
    latest = latest.sort_values("prob", ascending=False)
    print(f"Chance each stock beats {MARKET} over the next {HORIZON} trading days (as of {latest['date'].max():%Y-%m-%d}):")
    for _, r in latest.iterrows():
        print(f"  {r['ticker']:6s} {r['prob']:.0%}  {'#' * int(r['prob'] * 40)}")

    out = {
        "demo": args.demo,
        "asOf": f"{latest['date'].max():%Y-%m-%d}",
        "model": best_name,
        "horizonDays": HORIZON,
        "testAccuracy": round(float(np.mean((results[best_name][1] >= 0.5) == yte)), 4),
        "baselineAccuracy": round(float(np.mean(yte == baseline_guess)), 4),
        "predictions": [{"ticker": r["ticker"], "probBeatMarket": round(float(r["prob"]), 4)} for _, r in latest.iterrows()],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nSaved to {args.out}")

    say("How to read this",
        "- Stock markets are close to random week to week. 51-53% accuracy is normal, even for good models.\n"
        "- If a model scores much higher than that, suspect a bug (usually look-ahead) before celebrating.\n"
        "- A small edge only matters if it holds up over many weeks and beats trading costs.\n"
        "- Treat these predictions as one more clue next to fundamentals and news, never on their own.")


if __name__ == "__main__":
    main()
