"""
Fetch the Deribit DVOL index (BTC 30-day implied volatility) daily history.

DVOL is Deribit's model-free 30-day implied-volatility index, the Q-measure
leg of the variance risk premium (Ch. 8 of affine_intensity_theory):

    E^Q_t[QV_{t,t+30}]  ~  DVOL_t^2 * (30/365)

Public endpoint, no authentication:
    /api/v2/public/get_volatility_index_data
    -> result.data = [[timestamp_ms, open, high, low, close], ...]

The server caps the number of candles per response, so we walk the window in
chunks and honour the `continuation` cursor when present.

Output: dvol_btc_daily.csv with columns [date, open, high, low, close].
"""

import io
import json
import time
import urllib.request

import pandas as pd

URL = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
       "?currency={cur}&start_timestamp={t0}&end_timestamp={t1}&resolution=1D")

DAY_MS = 86_400_000
CHUNK_DAYS = 300          # stay well inside the per-response candle cap
START = "2021-01-01"      # DVOL history begins in 2021; earlier requests return []
CURRENCY = "BTC"


def _get(cur, t0, t1, retries=4):
    """One request with linear backoff; returns result.data (list of candles)."""
    url = URL.format(cur=cur, t0=int(t0), t1=int(t1))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                payload = json.loads(r.read().decode())
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload["result"]["data"]
        except Exception as exc:                       # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {exc}")
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch(currency=CURRENCY, start=START):
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t_end = int(pd.Timestamp.utcnow().timestamp() * 1000)
    rows, cursor = [], t0
    while cursor < t_end:
        stop = min(cursor + CHUNK_DAYS * DAY_MS, t_end)
        data = _get(currency, cursor, stop)
        print(f"  {pd.Timestamp(cursor, unit='ms').date()} .. "
              f"{pd.Timestamp(stop, unit='ms').date()}  -> {len(data)} candles")
        rows.extend(data)
        cursor = stop + DAY_MS
        time.sleep(0.25)                               # be polite to the endpoint

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = (df.drop(columns="ts")
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .set_index("date"))
    return df


def main():
    print(f"fetching Deribit DVOL ({CURRENCY}) daily from {START} ...")
    df = fetch()
    out = "dvol_btc_daily.csv"
    df.to_csv(out, float_format="%.4f")
    gaps = pd.date_range(df.index[0], df.index[-1], freq="D").difference(df.index)
    print(f"\n{len(df)} daily observations: {df.index[0].date()} .. {df.index[-1].date()}")
    print(f"missing calendar days: {len(gaps)}")
    print(f"close: min {df['close'].min():.2f}  median {df['close'].median():.2f}  "
          f"max {df['close'].max():.2f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
