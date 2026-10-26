# Ledgerline

A personal finance app in one HTML file: stock scoring, budgeting, and an AI assistant.

## Features

- **Stocks**: enter your holdings or watchlist with key numbers (price, P/E, EPS growth, revenue growth, net margin, debt/equity, dividend yield, 52-week range). Each stock gets a score and a verdict: *Buy candidate*, *Hold / watch*, or *Avoid for now*. A breakdown shows how each factor counted. Gain/loss, allocation, and concentration warnings are included.
- **Budget**: set your monthly take-home pay and emergency fund, create categories (need / want / saving) with limits, and log transactions. The app shows spending against each limit, a 50/30/20 check, your savings rate, and how many months your emergency fund covers.
- **Daily picks**: each weekday, scheduled Claude research runs after the market opens (9:35am New York) and after it closes (4:10pm). The open run finds 3–5 stocks that score as buy candidates under the rules below, and posts them with a market summary, reasons, risks, and sources. The close run also reviews how the morning picks did and suggests ideas for the next day. Both runs also check the stocks you own and mark each one Sell, Trim, Watch or Hold with a reason; sell and trim alerts show as a banner on the Overview and lead the phone notification. Both list popular names to avoid. Add any pick to your watchlist with one click.
- **Ask AI**: chat about your portfolio and budget. The assistant sees your numbers and scores. It works when the page is opened as a Claude artifact.

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

Open `index.html` in a browser. Opened as a local file, data is saved in your browser's localStorage and the AI chat is turned off. Opened as a Claude artifact, data is saved privately to your account and the AI chat works.

This is a learning and organizing tool, not financial advice. Numbers you enter can go stale, so check current prices before you act.
