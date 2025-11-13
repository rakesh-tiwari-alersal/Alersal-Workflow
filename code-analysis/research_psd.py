#!/usr/bin/env python3
# research_psd.py

from __future__ import annotations
import os, argparse, csv
import pandas as pd
import numpy as np
from scipy.signal import periodogram, find_peaks

# RANGES: final block set to 150–1000 (top 20 there)
RANGES = [
    (15, 40),
    (30, 60),
    (50, 90),
    (150, 350),
    (200, 500),
    (350, 700),
    (150, 1000)  # final block: top 20
]

def analyze_single_range(series, rmin, rmax):
    closes = series.values
    if len(closes) < 10:
        return None

    closes_diff = closes[1:] - closes[:-1]
    freq, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')

    mask = freq > 0
    freq = freq[mask]
    psd = psd[mask]
    if len(freq) == 0:
        return None

    # periods for the full (positive-frequency) PSD
    periods_full = 1.0 / freq

    # find peaks on the full PSD (stable detection)
    try:
        # --- thresholding based on long-cycle (> = 100 days) PSD only ---
        mask_long = periods_full >= 100
        psd_for_threshold = psd[mask_long] if np.any(mask_long) else psd
        ht_full = 0.001 * np.max(psd_for_threshold)
        # --------------------------------------------------------------
    except Exception:
        return None
    if np.isnan(ht_full):
        return None
    peaks_full, _ = find_peaks(psd, height=ht_full, distance=1)
    if len(peaks_full) == 0:
        return None

    # now filter those peaks to the requested range
    if rmax is None:
        rmask = (periods_full >= rmin)
    else:
        rmask = (periods_full >= rmin) & (periods_full <= rmax)

    # keep only peaks whose periods fall inside rmin..rmax
    selected = peaks_full[rmask[peaks_full]]
    if len(selected) == 0:
        return None

    pdf = pd.DataFrame({'Period': periods_full[selected], 'Power': psd[selected]})
    pdf = pdf.sort_values('Power', ascending=False)

    top_n = 20 if (rmin == 150 and rmax == 1000) else 10
    pdf = pdf.head(top_n).copy()

    tp = pdf['Power'].sum()
    pdf['% Power'] = (pdf['Power'] / tp * 100.0) if tp > 0 else 0

    pdf['Period'] = pdf['Period'].round(2)
    pdf['% Power'] = pdf['% Power'].round(2)
    return pdf[['Period', '% Power']]

def compute_centroid(series, rmin, rmax=None):
    """
    Power-weighted centroid of PSD peak periods within [rmin, rmax].
    If rmax is None, uses [rmin, +∞).
    Returns a formatted string with 2 decimals or "" if unavailable.
    """
    closes = series.values
    if len(closes) < 10:
        return ""

    diff = closes[1:] - closes[:-1]
    f, psd = periodogram(diff, fs=1, scaling='density', window='hann')
    mask = f > 0
    f = f[mask]
    psd = psd[mask]
    if len(f) == 0:
        return ""

    periods_full = 1.0 / f

    try:
        # --- thresholding based on long-cycle (> = 100 days) PSD only ---
        mask_long = periods_full >= 100
        psd_for_threshold = psd[mask_long] if np.any(mask_long) else psd
        ht_full = 0.001 * np.max(psd_for_threshold)
        # --------------------------------------------------------------
    except Exception:
        return ""
    if np.isnan(ht_full):
        return ""

    peaks_full, _ = find_peaks(psd, height=ht_full, distance=1)
    if len(peaks_full) == 0:
        return ""

    if rmax is None:
        rm = (periods_full >= rmin)
    else:
        rm = (periods_full >= rmin) & (periods_full <= rmax)

    sel = peaks_full[rm[peaks_full]]
    if len(sel) == 0:
        return ""

    p = periods_full[sel]
    w = psd[sel]
    m = (~np.isnan(p)) & (~np.isnan(w)) & (w > 0)
    if not np.any(m):
        return ""

    p = p[m]; w = w[m]
    s = np.sum(w)
    if s <= 0:
        return ""

    c = float(np.sum(p * w) / s)
    return f"{c:.2f}"

def write_all(symbol, c165_500, c200_800, lower_half_centroid, upper_half_centroid, clear):
    os.makedirs("research_output", exist_ok=True)
    fn = "research_output/research_psd_ALL.csv"
    header = ["Symbol", "Centroid_165_500", "Centroid_200_800", "LowerHalfCentroid", "UpperHalfCentroid"]
    row = [symbol, c165_500, c200_800, lower_half_centroid, upper_half_centroid]

    if clear:
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(row)
        return

    need_header = not os.path.exists(fn) or os.stat(fn).st_size == 0
    with open(fn, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-r", "--range", default=None)
    parser.add_argument("-c", "--clear-summary", action="store_true")
    args = parser.parse_args()

    fp = os.path.join("historical_data", args.file)
    if not os.path.exists(fp):
        print(f"Error: file '{fp}' not found.")
        return

    try:
        df = pd.read_csv(fp, index_col="Date", parse_dates=True)
        if 'Close' in df.columns:
            series = df['Close']
        elif 'close' in df.columns:
            series = df['close']
        elif 'Price' in df.columns:
            series = df['Price']
        elif 'price' in df.columns:
            series = df['price']
        else:
            print("Error: CSV must contain Close/close/Price/price.")
            return
    except Exception:
        print("Error: CSV unreadable.")
        return

    # Single-range mode
    if args.range:
        try:
            rmin, rmax = map(int, args.range.split(","))
            res = analyze_single_range(series, rmin, rmax)
            if res is None:
                print("No cycles found.")
                return
            print(f"Cycle ({rmin}-{rmax})\t% Power")
            for _, row in res.iterrows():
                print(f"{row['Period']}\t{row['% Power']}")
            return
        except Exception:
            print("Format error. Use: -r 30,60")
            return

    # Bulk ranges for the per-symbol table
    results = {}
    for rmin, rmax in RANGES:
        r = analyze_single_range(series, rmin, rmax)
        if r is not None:
            results[f"{rmin}-{rmax}"] = r

    if not results:
        print("No cycles found.")
        return

    g1 = [f"{RANGES[0][0]}-{RANGES[0][1]}", f"{RANGES[1][0]}-{RANGES[1][1]}", f"{RANGES[2][0]}-{RANGES[2][1]}"]
    g2 = [f"{RANGES[3][0]}-{RANGES[3][1]}", f"{RANGES[4][0]}-{RANGES[4][1]}", f"{RANGES[5][0]}-{RANGES[5][1]}"]
    last = f"{RANGES[-1][0]}-{RANGES[-1][1]}"

    G1 = [results.get(x, pd.DataFrame(columns=['Period', '% Power'])) for x in g1]
    G2 = [results.get(x, pd.DataFrame(columns=['Period', '% Power'])) for x in g2]
    G3 = [results.get(last, pd.DataFrame(columns=['Period', '% Power']))]

    maxr = max(
        max(len(d) for d in G1) if G1 else 0,
        max(len(d) for d in G2) if G2 else 0,
        max(len(d) for d in G3) if G3 else 0
    )

    header = []
    for i, r in enumerate(g1):
        header += [f"Cycle ({r})", "% Power"]
        if i < len(g1) - 1:
            header.append("")
    header.append("")
    for i, r in enumerate(g2):
        header += [f"Cycle ({r})", "% Power"]
        if i < len(g2) - 1:
            header.append("")
    header.append("")
    header += [f"Cycle ({last})", "% Power"]

    rows = [header]
    for i in range(maxr):
        row = []
        for j, d in enumerate(G1):
            if i < len(d):
                row += [f"{d.iloc[i]['Period']:.2f}", f"{d.iloc[i]['% Power']:.2f}"]
            else:
                row += ["", ""]
            if j < len(G1) - 1:
                row.append("")
        row.append("")
        for j, d in enumerate(G2):
            if i < len(d):
                row += [f"{d.iloc[i]['Period']:.2f}", f"{d.iloc[i]['% Power']:.2f}"]
            else:
                row += ["", ""]
            if j < len(G2) - 1:
                row.append("")
        row.append("")
        for j, d in enumerate(G3):
            if i < len(d):
                row += [f"{d.iloc[i]['Period']:.2f}", f"{d.iloc[i]['% Power']:.2f}"]
            else:
                row += ["", ""]
        rows.append(row)

    os.makedirs("research_output", exist_ok=True)
    name = os.path.splitext(args.file)[0]
    out = f"research_output/research_psd_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, header=False)
    print(f"Saved {out}")

    # === Centroids for _ALL summary (Centroid_165_500, Centroid_200_800 + LowerHalfCentroid + UpperHalf) ===
    c165_500 = compute_centroid(series, 165, 500)
    c200_800 = compute_centroid(series, 200, 800)

    # LowerHalfCentroid = centroid over [200, Centroid_200_800]
    try:
        c_val = float(c200_800) if c200_800 not in ("", None) else None
    except Exception:
        c_val = None
    lower_half_centroid = compute_centroid(series, 200, c_val) if c_val else ""

    # UpperHalfCentroid = centroid over [Centroid_200_800, 800] (upper-half of 200_800 centroid)
    try:
        ch_val = float(c200_800) if c200_800 not in ("", None) else None
    except Exception:
        ch_val = None
    upper_half_centroid = compute_centroid(series, ch_val, 800) if ch_val else ""

    # === _ALL summary (Symbol, Centroid_165_500, Centroid_200_800, LowerHalfCentroid, UpperHalfCentroid) ===
    write_all(
        name,
        c165_500,            # Centroid_165_500
        c200_800,            # Centroid_200_800
        lower_half_centroid, # LowerHalfCentroid = Centroid[200, Centroid_200_800]
        upper_half_centroid, # UpperHalfCentroid = Centroid[Centroid_200_800, 800]
        args.clear_summary
    )

if __name__ == "__main__":
    main()
