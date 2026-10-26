#!/usr/bin/env python3
"""Ledgerline data collector.

Gathers the numbers the research runs need for each ticker, straight from
the source instead of from search snippets:

  * Prices (Yahoo Finance chart API): current price, day / 1-month change,
    52-week range, 50- and 200-day moving averages, 14-day RSI, dividends.
  * Fundamentals (SEC EDGAR XBRL company facts): trailing-twelve-month
    revenue, net income and diluted EPS with year-over-year growth, net
    margin, P/E, and debt / equity.

It also scores each ticker with the same rules as the app's Stocks tab.

Usage:
    python3 collector.py MSFT KO TSLA            # JSON to stdout
    python3 collector.py --out quotes.json MSFT  # JSON to a file

Standard library only. Network hosts used: query1.finance.yahoo.com,
www.sec.gov and data.sec.gov.
"""
import argparse
import datetime as dt
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

UA = "Ledgerline personal research (contact: ledgerline-research@users.noreply.github.com)"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=1y&interval=1d&events=div"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenuesNetOfInterestExpense",
]
NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]
EQUITY_TAGS = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
# Debt parts are summed; "LongTermDebt" already includes its current portion.
DEBT_TOTAL_TAGS = ["LongTermDebt", "DebtLongtermAndShorttermCombinedAmount"]
DEBT_NONCURRENT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
DEBT_CURRENT_TAGS = ["LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
SHORT_BORROW_TAGS = ["ShortTermBorrowings", "CommercialPaper"]


# ---------------------------------------------------------------- http

def _ssl_context():
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if not cafile and os.path.exists("/root/.ccr/ca-bundle.crt"):
        cafile = "/root/.ccr/ca-bundle.crt"
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


_CTX = _ssl_context()


def get_json(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} for {url}"
            if e.code in (404,):
                break
        except Exception as e:  # network errors, bad JSON
            last = f"{type(e).__name__}: {e} for {url}"
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)


# ---------------------------------------------------------------- prices

def sma(values, n):
    return sum(values[-n:]) / n if len(values) >= n else None


def rsi(closes, n=14):
    if len(closes) <= n:
        return None
    gains, losses = [], []
    for a, b in zip(closes[:-1], closes[1:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):  # Wilder smoothing
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def pct(a, b):
    return None if a is None or b in (None, 0) else (a - b) / abs(b) * 100


def price_data(chart):
    res = chart["chart"]["result"][0]
    meta = res["meta"]
    ts = res.get("timestamp") or []
    q = (res.get("indicators") or {}).get("quote", [{}])[0]
    rows = [(t, c, h, l) for t, c, h, l in zip(ts, q.get("close", []), q.get("high", []), q.get("low", [])) if c is not None]
    closes = [r[1] for r in rows]
    price = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
    mkt_time = meta.get("regularMarketTime")
    # Previous close: the last daily close before the current session.
    prev = None
    if rows and mkt_time:
        today = dt.datetime.utcfromtimestamp(mkt_time + (meta.get("gmtoffset") or 0)).date()
        earlier = [c for t, c, _, _ in rows if dt.datetime.utcfromtimestamp(t + (meta.get("gmtoffset") or 0)).date() < today]
        prev = earlier[-1] if earlier else None
    if prev is None and len(closes) >= 2:
        prev = closes[-2]
    hi = meta.get("fiftyTwoWeekHigh") or (max(r[2] for r in rows if r[2] is not None) if rows else None)
    lo = meta.get("fiftyTwoWeekLow") or (min(r[3] for r in rows if r[3] is not None) if rows else None)
    one_year_ago = (mkt_time or time.time()) - 365 * 86400
    divs = (res.get("events") or {}).get("dividends") or {}
    div_ttm = sum(d.get("amount", 0) for d in divs.values() if d.get("date", 0) >= one_year_ago)
    month_ago = closes[-22] if len(closes) >= 22 else None
    s50, s200 = sma(closes, 50), sma(closes, 200)
    return {
        "name": meta.get("longName") or meta.get("shortName"),
        "currency": meta.get("currency"),
        "price": price,
        "prevClose": prev,
        "dayChangePct": pct(price, prev),
        "monthChangePct": pct(price, month_ago),
        "low52": lo,
        "high52": hi,
        "fromHigh52Pct": pct(price, hi),
        "sma50": s50,
        "sma200": s200,
        "aboveSma50": None if s50 is None or price is None else price > s50,
        "aboveSma200": None if s200 is None or price is None else price > s200,
        "rsi14": rsi(closes),
        "divYield": (div_ttm / price * 100) if price else None,
        "priceTime": dt.datetime.utcfromtimestamp(mkt_time).isoformat() + "Z" if mkt_time else None,
    }


# ---------------------------------------------------------------- fundamentals

def _d(s):
    return dt.date.fromisoformat(s)


def _facts(gaap, tags, unit_pred):
    """First tag with data -> list of facts, deduped by period keeping the latest filing."""
    for tag in tags:
        units = (gaap.get(tag) or {}).get("units") or {}
        for unit, facts in units.items():
            if not unit_pred(unit):
                continue
            best = {}
            for f in facts:
                if f.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
                    continue
                key = (f.get("start"), f["end"])
                if key not in best or f.get("filed", "") > best[key].get("filed", ""):
                    best[key] = f
            if best:
                return list(best.values())
    return []


def _near(a, b, days=12):
    return abs((_d(a) - _d(b)).days) <= days


def ttm(facts, years_back=0):
    """Trailing-twelve-month value: latest annual + current YTD - prior-year YTD.

    years_back=1 gives the same measure one year earlier (for growth)."""
    dur = [f for f in facts if f.get("start")]
    annual = sorted((f for f in dur if 350 <= (_d(f["end"]) - _d(f["start"])).days <= 380), key=lambda f: f["end"])
    if not annual:
        return None, None
    if years_back:
        # Shift the "now" point back a year.
        latest_end = max(f["end"] for f in dur)
        cutoff = _d(latest_end) - dt.timedelta(days=360)
        annual = [f for f in annual if _d(f["end"]) <= cutoff]
        dur = [f for f in dur if _d(f["end"]) <= cutoff]
        if not annual:
            return None, None
    a = annual[-1]
    ytd = [f for f in dur if _d(f["start"]) > _d(a["end"]) and _near(f["start"], (_d(a["end"]) + dt.timedelta(days=1)).isoformat())
           and (_d(f["end"]) - _d(f["start"])).days < 350]
    if not ytd:
        return a["val"], a["end"]
    cur = max(ytd, key=lambda f: f["end"])
    cur_len = (_d(cur["end"]) - _d(cur["start"])).days
    prior = [f for f in dur if _near(f["start"], a["start"]) and abs((_d(f["end"]) - _d(f["start"])).days - cur_len) <= 12]
    if not prior:
        return a["val"], a["end"]
    return a["val"] + cur["val"] - prior[0]["val"], cur["end"]


def latest_instant(facts):
    inst = [f for f in facts if not f.get("start")]
    if not inst:
        return None, None
    f = max(inst, key=lambda f: f["end"])
    return f["val"], f["end"]


def instant_at(facts, end):
    for f in facts:
        if not f.get("start") and f["end"] == end:
            return f["val"]
    return None


def fundamentals(facts_json):
    gaap = (facts_json.get("facts") or {}).get("us-gaap") or {}
    usd = lambda u: u == "USD"
    eps_unit = lambda u: u.startswith("USD/") or u == "USD/shares"
    rev_f = _facts(gaap, REVENUE_TAGS, usd)
    ni_f = _facts(gaap, NET_INCOME_TAGS, usd)
    eps_f = _facts(gaap, EPS_TAGS, eps_unit)
    rev, rev_end = ttm(rev_f)
    rev_prev, _ = ttm(rev_f, 1)
    ni, _ = ttm(ni_f)
    eps, eps_end = ttm(eps_f)
    eps_prev, _ = ttm(eps_f, 1)
    equity, eq_end = latest_instant(_facts(gaap, EQUITY_TAGS, usd))
    debt = None
    if eq_end:
        total = next((v for v in (instant_at(_facts(gaap, [t], usd), eq_end) for t in DEBT_TOTAL_TAGS) if v is not None), None)
        if total is None:
            nc = next((v for v in (instant_at(_facts(gaap, [t], usd), eq_end) for t in DEBT_NONCURRENT_TAGS) if v is not None), None)
            cur = next((v for v in (instant_at(_facts(gaap, [t], usd), eq_end) for t in DEBT_CURRENT_TAGS) if v is not None), None)
            total = (nc or 0) + (cur or 0) if (nc is not None or cur is not None) else None
        short = sum(v for v in (instant_at(_facts(gaap, [t], usd), eq_end) for t in SHORT_BORROW_TAGS) if v is not None)
        debt = None if total is None and not short else (total or 0) + short
    return {
        "revenueTTM": rev,
        "netIncomeTTM": ni,
        "epsTTM": eps,
        "revGrowth": pct(rev, rev_prev),
        "epsGrowth": pct(eps, eps_prev) if eps_prev and eps_prev > 0 else None,
        "margin": (ni / rev * 100) if ni is not None and rev else None,
        "debtEq": (debt / equity) if debt is not None and equity and equity > 0 else None,
        "fundamentalsAsOf": eps_end or rev_end,
    }


_CIKS = None


def cik_for(ticker):
    global _CIKS
    if _CIKS is None:
        data = get_json(SEC_TICKERS)
        _CIKS = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    return _CIKS.get(ticker.upper().replace(".", "-"))


# ---------------------------------------------------------------- scoring (same rules as the app)

def score(m):
    pe, eg, rg, mg, de, dy = (m.get(k) for k in ("pe", "epsGrowth", "revGrowth", "margin", "debtEq", "divYield"))
    pts, mx = 0, 0
    if pe is not None:
        mx += 2
        if pe <= 0:
            pts -= 2
        elif eg is not None and eg > 0:
            peg = pe / (eg + (dy or 0))
            pts += 2 if peg < 1 else 1 if peg <= 2 else 0 if peg <= 3 else -2
        else:
            pts += 0 if pe < 15 else -2
    if eg is not None:
        mx += 2
        pts += 2 if eg >= 20 else 1 if eg >= 8 else 0 if eg >= 0 else -2
    if rg is not None:
        mx += 1
        pts += 1 if rg >= 15 else 0 if rg >= 0 else -1
    if mg is not None:
        mx += 2
        pts += 2 if mg >= 20 else 1 if mg >= 10 else 0 if mg >= 0 else -2
    if de is not None:
        mx += 1
        pts += 1 if de <= 0.5 else 0 if de <= 1.5 else -1 if de <= 3 else -2
    if dy is not None:
        mx += 1
        pts += 1 if 2 <= dy <= 8 else -1 if dy > 8 else 0
    ratio = pts / mx if mx else 0
    verdict = "Need data" if not mx else "Buy candidate" if ratio >= 0.5 else "Hold / watch" if ratio >= 0 else "Avoid for now"
    return {"score": pts, "scoreMax": mx, "verdict": verdict}


def trend_label(m):
    a50, a200, r = m.get("aboveSma50"), m.get("aboveSma200"), m.get("rsi14")
    if a50 is None or a200 is None:
        t = "unknown"
    elif a50 and a200:
        t = "uptrend"
    elif not a50 and not a200:
        t = "downtrend"
    else:
        t = "mixed"
    if r is not None and r >= 70:
        t += ", overbought"
    elif r is not None and r <= 30:
        t += ", oversold"
    return t


# ---------------------------------------------------------------- main

def rnd(v, n=2):
    return round(v, n) if isinstance(v, float) else v


def collect(ticker):
    t = ticker.upper().strip()
    out = {"ticker": t, "errors": []}
    try:
        out.update(price_data(get_json(YAHOO_CHART.format(t=t))))
    except Exception as e:
        out["errors"].append(f"prices: {e}")
    try:
        cik = cik_for(t)
        if cik is None:
            out["errors"].append("fundamentals: not an SEC filer (e.g. ETF or foreign company)")
        else:
            out.update(fundamentals(get_json(SEC_FACTS.format(cik=cik))))
    except Exception as e:
        out["errors"].append(f"fundamentals: {e}")
    eps, price = out.get("epsTTM"), out.get("price")
    out["pe"] = (price / eps) if price and eps else None
    out.update(score(out))
    out["trend"] = trend_label(out)
    out["collectedAt"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for k in ("revenueTTM", "netIncomeTTM"):
        if isinstance(out.get(k), (int, float)):
            out[k] = round(out[k])
    return {k: rnd(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for t in dict.fromkeys(x.upper() for x in args.tickers):
        results.append(collect(t))
        time.sleep(0.3)  # stay well under SEC's 10 requests/second
    text = json.dumps(results, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    if all(r["errors"] and r.get("price") is None for r in results):
        sys.exit(2)


if __name__ == "__main__":
    main()
