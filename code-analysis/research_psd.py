#!/usr/bin/env python3
"""
research_psd.py

Produces per-symbol PSD peak tables across multiple period ranges and appends
a summary row into research_output/research_psd_ALL.csv.

Behavior:
- Six initial ranges return top 10 peaks each.
- The final range (150-700) returns top 20 peaks.
- _ALL file row format: Symbol, <centroid150_700>, <centroid200_700>, <centroid300_700>, <centroid400_700>
- Writes header only when --clear-summary is passed or first creation.
"""

from __future__ import annotations
import os, sys, argparse, csv
import pandas as pd
import numpy as np
from scipy.signal import periodogram, find_peaks

# PSD ranges (unchanged)
RANGES = [
    (15, 40),
    (30, 60),
    (50, 90),
    (150, 350),
    (200, 500),
    (350, 700),
    (150, 700)  # final block: top 20
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

    periods = 1.0 / freq
    rmask = (periods >= rmin) & (periods <= rmax)
    periods = periods[rmask]
    psd = psd[rmask]
    if len(psd) == 0:
        return None

    try:
        ht = 0.001 * np.max(psd)
    except:
        return None
    if np.isnan(ht):
        return None

    peaks, _ = find_peaks(psd, height=ht, distance=1)
    if len(peaks) == 0:
        return None

    pdf = pd.DataFrame({'Period': periods[peaks], 'Power': psd[peaks]})
    pdf = pdf.sort_values('Power', ascending=False)

    top_n = 20 if (rmin == 150 and rmax == 700) else 10
    pdf = pdf.head(top_n).copy()

    tp = pdf['Power'].sum()
    pdf['% Power'] = (pdf['Power'] / tp * 100.0) if tp > 0 else 0

    pdf['Period'] = pdf['Period'].round(2)
    pdf['% Power'] = pdf['% Power'].round(2)
    return pdf[['Period', '% Power']]


def compute_centroid(series, rmin, rmax):
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

    periods = 1.0 / f
    rm = (periods >= rmin) & (periods <= rmax)
    periods = periods[rm]
    psd = psd[rm]
    if len(psd) == 0:
        return ""

    peaks, _ = find_peaks(psd, height=0.001 * np.max(psd), distance=1)
    if len(peaks) == 0:
        return ""

    p = periods[peaks]
    w = psd[peaks]
    mask = (~np.isnan(p)) & (~np.isnan(w)) & (w > 0)
    if not np.any(mask):
        return ""

    p = p[mask]; w = w[mask]
    s = np.sum(w)
    if s <= 0:
        return ""

    c = float(np.sum(p * w) / s)
    return f"{c:.2f}"


def write_all(file, c150_700, c200_700, c300_700, c400_700, clear):
    os.makedirs("research_output", exist_ok=True)
    fn = "research_output/research_psd_ALL.csv"
    header = ["Symbol","Centroid150_700","Centroid200_700","Centroid300_700","Centroid400_700"]
    row = [file,c150_700,c200_700,c300_700,c400_700]

    if clear:
        with open(fn, "w", newline="") as f:
            csv.writer(f).writerow(header)
            csv.writer(f).writerow(row)
        return

    need_header = not os.path.exists(fn) or os.stat(fn).st_size == 0
    with open(fn,"a",newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file",required=True)
    parser.add_argument("-r","--range",default=None)
    parser.add_argument("-c","--clear-summary",action="store_true")
    args = parser.parse_args()

    fp = os.path.join("historical_data",args.file)
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
    except:
        print("Error: CSV unreadable.")
        return

    if args.range:
        try:
            rmin,rmax = map(int,args.range.split(","))
            res = analyze_single_range(series,rmin,rmax)
            if res is None:
                print("No cycles found.")
                return
            print(f"Cycle ({rmin}-{rmax})\t% Power")
            for _,row in res.iterrows():
                print(f"{row['Period']}\t{row['% Power']}")
            return
        except:
            print("Format error. Use: -r 30,60")
            return

    results = {}
    for rmin,rmax in RANGES:
        r = analyze_single_range(series,rmin,rmax)
        if r is not None:
            results[f"{rmin}-{rmax}"] = r

    if not results:
        print("No cycles found.")
        return

    g1 = [f"{RANGES[0][0]}-{RANGES[0][1]}",f"{RANGES[1][0]}-{RANGES[1][1]}",f"{RANGES[2][0]}-{RANGES[2][1]}"]
    g2 = [f"{RANGES[3][0]}-{RANGES[3][1]}",f"{RANGES[4][0]}-{RANGES[4][1]}",f"{RANGES[5][0]}-{RANGES[5][1]}"]
    last = f"{RANGES[-1][0]}-{RANGES[-1][1]}"

    G1 = [results.get(x,pd.DataFrame(columns=['Period','% Power'])) for x in g1]
    G2 = [results.get(x,pd.DataFrame(columns=['Period','% Power'])) for x in g2]
    G3 = [results.get(last,pd.DataFrame(columns=['Period','% Power']))]

    maxr = max(max(len(d) for d in G1),max(len(d) for d in G2),max(len(d) for d in G3))

    header=[]
    for i,r in enumerate(g1):
        header+= [f"Cycle ({r})","% Power"]
        if i<len(g1)-1: header.append("")
    header.append("")
    for i,r in enumerate(g2):
        header+= [f"Cycle ({r})","% Power"]
        if i<len(g2)-1: header.append("")
    header.append("")
    header+= [f"Cycle ({last})","% Power"]

    rows=[header]
    for i in range(maxr):
        row=[]
        for j,d in enumerate(G1):
            if i<len(d): row+= [f"{d.iloc[i]['Period']:.2f}",f"{d.iloc[i]['% Power']:.2f}"]
            else: row+= ["",""]
            if j<len(G1)-1: row.append("")
        row.append("")
        for j,d in enumerate(G2):
            if i<len(d): row+= [f"{d.iloc[i]['Period']:.2f}",f"{d.iloc[i]['% Power']:.2f}"]
            else: row+= ["",""]
            if j<len(G2)-1: row.append("")
        row.append("")
        for j,d in enumerate(G3):
            if i<len(d): row+= [f"{d.iloc[i]['Period']:.2f}",f"{d.iloc[i]['% Power']:.2f}"]
            else: row+= ["",""]
        rows.append(row)

    os.makedirs("research_output",exist_ok=True)
    name = os.path.splitext(args.file)[0]
    out = f"research_output/research_psd_{name}.csv"
    pd.DataFrame(rows).to_csv(out,index=False,header=False)
    print(f"Saved {out}")

    write_all(
        name,
        compute_centroid(series,150,700),
        compute_centroid(series,200,700),
        compute_centroid(series,300,700),
        compute_centroid(series,400,700),
        args.clear_summary
    )


if __name__=="__main__":
    main()
