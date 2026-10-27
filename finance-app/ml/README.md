# Ledgerline ML: your first stock prediction model

`stock_predictor.py` trains a computer to answer one question:

> Will this stock beat the market (SPY) over the next 5 trading days?

## Run it

```bash
pip install pandas numpy scikit-learn
python3 stock_predictor.py --demo   # practice on fake random prices (works offline)
python3 stock_predictor.py          # real prices for ~40 big US companies (needs network access to query1.finance.yahoo.com)
```

The script prints each step as it goes, like a lesson, and saves next week's predictions to `predictions.json`.

## The 6 steps (the same in almost every ML project)

| Step | What happens here | Why it matters |
|---|---|---|
| 1. Collect data | 5 years of daily prices per stock | No data, no learning |
| 2. Features | Turn prices into clues: momentum, distance from 50/200-day averages, RSI, volatility, volume, how the market did | The model only sees what you give it |
| 3. Labels | 1 if the stock beat SPY over the next 5 days, else 0 | The "right answers" the model learns from |
| 4. Split by time | Train on the older 75%, test on the newest 25% | Testing on data the model never saw is the only honest test |
| 5. Train and test | Logistic regression and a random forest vs. a "dumb" baseline | A model is only useful if it beats the baseline |
| 6. Predict | Retrain on everything, score this week's data | The actual output |

## Things to try (this is how you learn)

1. Run `--demo` first. The prices are pure chance, so the models should NOT beat the baseline. If they did, the test would be broken.
2. Run on real data and compare. Is the random forest above 50%? Does its top 5 beat SPY in more than half the weeks?
3. Change `HORIZON` from 5 to 20 (one month). Does it get easier or harder?
4. Delete a feature from `FEATURES` and re-run. Did accuracy drop? Then that clue mattered.
5. Add your own feature in `make_features`, for example `c.pct_change(120)` (6-month momentum).
6. Break it on purpose: change `make_features` to use `c.shift(-1)` (tomorrow's price) as a feature. Accuracy jumps to an impossible level. That's "look-ahead" cheating, the most common beginner mistake.

## Honest expectations

Week-to-week stock moves are mostly noise. Real-world models that are right 51 to 53% of the time are considered good. Treat the predictions as one clue next to fundamentals and news, never as a reason to buy on their own.
