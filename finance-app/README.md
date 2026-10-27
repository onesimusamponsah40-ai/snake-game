# Ledgerline

An AI stock market analyst in one HTML file: it finds the best stocks to buy, flags the ones to sell, and answers your questions.

## Features

- **Stocks**: enter your holdings or watchlist with key numbers (price, P/E, EPS growth, revenue growth, net margin, debt/equity, dividend yield, 52-week range). Each stock gets a score and a verdict: *Buy candidate*, *Hold / watch*, or *Avoid for now*. A breakdown shows how each factor counted. Gain/loss, allocation, and concentration warnings are included.
- **Daily picks**: each weekday, scheduled Claude research runs after the market opens (9:35am New York) and after it closes (4:10pm). The open run finds 3–5 stocks that score as buy candidates under the rules below, and posts them with a market summary, reasons, risks, and sources. The close run also reviews how the morning picks did and suggests ideas for the next day. Both runs also check the stocks you own and mark each one Sell, Trim, Watch or Hold with a reason; sell and trim alerts show as a banner on the Overview and lead the phone notification. Both list popular names to avoid. Between them, a quick check runs every hour (10:30am–3:30pm New York) on your stocks and the day's picks, and shows in a "Latest check" panel. Add any pick to your watchlist with one click.
- **Ask AI**: chat about your portfolio and budget. The assistant sees your holdings, scores, trends, and the latest research picks and sell signals. It works when the page is opened as a Claude artifact.

## Data collection

`collector.py` gathers the numbers the research runs decide with, straight from the source:

- **Prices** (Yahoo Finance chart API): price, day and 1-month change, 52-week range, 50- and 200-day moving averages, 14-day RSI, trailing dividend yield.
- **Fundamentals** (SEC EDGAR XBRL filings): trailing-twelve-month revenue, net income and EPS with year-over-year growth, net margin, P/E, debt/equity.

It scores each ticker with the rules below. The runs save the results to the app's `quotes` collection, and the Stocks tab uses them in place of typed-in numbers, so you only enter the ticker, shares and average cost. News, catalysts and analyst views still come from web search.

```
python3 collector.py MSFT KO TSLA
```

It needs network access to `query1.finance.yahoo.com`, `www.sec.gov` and `data.sec.gov`.

## Scoring

| Factor | Points |
| --- | --- |
| Valuation: dividend-adjusted PEG = P/E ÷ (EPS growth + yield) | −2 to +2 |
| EPS growth | −2 to +2 |
| Revenue growth | −1 to +1 |
| Net margin | −2 to +2 |
| Debt / equity | −2 to +1 |
| Dividend yield (2–8% is good, above 8% may be a trap) | −1 to +1 |

At least 50% of max points = Buy candidate; 0 or more = Hold / watch; net negative = Avoid. Blank fields are skipped.

## Running it

Open `index.html` in a browser. Opened as a local file, your stock list is saved in your browser's localStorage and the AI chat is turned off. Opened as a Claude artifact, data is saved privately to your account and the AI chat works.

This is a learning and organizing tool, not financial advice. Numbers you enter can go stale, so check current prices before you act.

## Machine learning

`ml/stock_predictor.py` is a hands-on learning project: it trains models to predict which stocks will beat the market over the next week, tests them honestly on the most recent data, and backtests a "buy the top 5" strategy. See [ml/README.md](ml/README.md).
