"""
Clean ALL raw BTC index files into a single gapless 1-minute price series.

Raw inputs: 42 monthly files  YYYY-MM_btc_usd_price_index.csv, each holding
sub-minute index prints with columns "Date and Time, Price". Two quirks the
loader must absorb:
  * 2023-2024 files carry an extra first line  sep=,  (absent in 2025-2026).
  * the raw cadence is irregular (1s in places, ~5s in others).

Output: btc_1min.csv  with exactly two columns [datetime, price], where price
is the LAST index print in each 1-minute bin (previous-tick sampling -- the
convention under which realized variance converges to quadratic variation).
Crypto trades 24/7, so dead minutes are forward-filled onto a gapless grid.
Nothing else is retained.
"""

import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DT_FMT = "%Y-%m-%d %H:%M:%S"
OUT = os.path.join(ROOT, "btc_1min.csv")


def _skiprows(path):
    """Return 1 if the file opens with a 'sep=,' directive line, else 0."""
    with open(path, "r") as fh:
        return 1 if fh.readline().startswith("sep=") else 0


def load_month_1min(path):
    """Read one monthly file, resample to the 1-minute last-tick price."""
    df = pd.read_csv(path, skiprows=_skiprows(path), skipinitialspace=True)
    ts = pd.to_datetime(df["Date and Time"], format=DT_FMT)
    px = pd.Series(df["Price"].to_numpy(float), index=ts)
    return px.resample("1min").last()


def main(paths=None):
    paths = paths or sorted(glob.glob(os.path.join(ROOT, "*_btc_usd_price_index.csv")))
    if not paths:
        raise FileNotFoundError("no *_btc_usd_price_index.csv files found")
    print(f"cleaning {len(paths)} monthly files -> 1-minute last-tick price")

    chunks = []
    for p in paths:
        s = load_month_1min(p)
        chunks.append(s)
        print(f"  {os.path.basename(p):32s} {len(s):>6d} min  "
              f"[{s.index[0]} .. {s.index[-1]}]")

    px = pd.concat(chunks).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    # gapless 1-minute grid; carry the last print through dead minutes
    px = px.resample("1min").last().ffill().dropna()

    out = px.rename("price").rename_axis("datetime").reset_index()
    out.to_csv(OUT, index=False, float_format="%.2f")

    span_days = (px.index[-1] - px.index[0]).total_seconds() / 86400.0
    print(f"\n{len(out):,} one-minute bars  "
          f"[{px.index[0]} .. {px.index[-1]}]  ~{span_days:.0f} days")
    print(f"price: min {px.min():.2f}  median {px.median():.2f}  max {px.max():.2f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
