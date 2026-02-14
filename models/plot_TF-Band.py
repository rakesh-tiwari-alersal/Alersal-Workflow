import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
from datetime import datetime
import os

# -----------------------------
# FORCE FFMPEG
# -----------------------------
mpl.rcParams['animation.writer'] = 'ffmpeg'

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol", required=True)
parser.add_argument("-t", "--start_year", type=int)
parser.add_argument("-e", "--end_year", type=int)
parser.add_argument("-p", "--poly", type=int)
args = parser.parse_args()

SYMBOL = args.symbol

# -----------------------------
# LOAD MODEL CONFIG
# -----------------------------
cfg = pd.read_excel("model_selection.xlsx")
cfg.columns = [c.strip() for c in cfg.columns]
matches = cfg[cfg["Instrument"] == SYMBOL]

if matches.empty:
    available = ", ".join(sorted(cfg["Instrument"].unique()))
    raise SystemExit(
        f"\n❌ Symbol '{SYMBOL}' not found in model_selection.xlsx.\n"
    )
row = matches.iloc[0]

LONG_CYCLE, SHORT_CYCLE = map(int, row["Model S"].split(","))
ALPHA = float(row["Alpha"])
EQUILIBRIUM_FRAME = int(row["Equilibrium Frame"])
POLY_ORDER = int(args.poly) if args.poly is not None else int(row["Poly Order"])

BAND_SHORT_ENTRY = float(row["Short Entry"]) if not pd.isna(row["Short Entry"]) else 0.1459
BAND_SHORT_EXIT  = float(row["Short Exit"])  if not pd.isna(row["Short Exit"])  else None
BAND_LONG_ENTRY  = float(row["Long Entry"])  if not pd.isna(row["Long Entry"])  else 0.1459
BAND_LONG_EXIT   = float(row["Long Exit"])   if not pd.isna(row["Long Exit"])   else None

START_DATE = pd.to_datetime(row["Start Date"]) if not pd.isna(row["Start Date"]) else None
END_DATE   = pd.to_datetime(row["End Date"])   if not pd.isna(row["End Date"])   else None

if args.start_year:
    START_DATE = pd.Timestamp(f"{args.start_year}-01-01")
if args.end_year:
    END_DATE = pd.Timestamp(f"{args.end_year}-12-31")

if START_DATE is None or END_DATE is None or END_DATE <= START_DATE:
    raise ValueError("Invalid start/end date")

years = (END_DATE - START_DATE).days / 365.25
X_TICK_MODE = "Q" if years < 4 else "Y"

print("\n=== CONFIG ===")
print(f"{SYMBOL} | Model {LONG_CYCLE},{SHORT_CYCLE} | α={ALPHA}")
print(f"Frame={EQUILIBRIUM_FRAME} | Poly={POLY_ORDER}")
print(f"Dates {START_DATE.date()} → {END_DATE.date()}")
print("=============\n")

# -----------------------------
# FILES
# -----------------------------
FILE_PATH = f"historical_data/{SYMBOL}.csv"
OUTPUT_FILE = f"plot_results/{SYMBOL}-TF-Band-Plot.mp4"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

FPS = 24
TARGET_SECONDS = 30

# -----------------------------
# COLORS
# -----------------------------
COLOR_EQ     = "#4FC3F7"
COLOR_UPPER  = "#2ECC71"
COLOR_LOWER  = "#F11B1BEF"
COLOR_EXIT   = "#9E9E9E"
COLOR_PRICE  = "#CFCFCF"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(FILE_PATH)
df.columns = [c.strip() for c in df.columns]

def _col(name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise ValueError(f"Missing column {name}")

DATE_COL  = _col("Date")
PRICE_COL = _col("Close")
HIGH_COL  = _col("High")
LOW_COL   = _col("Low")

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df[df[DATE_COL] <= END_DATE].reset_index(drop=True)

# -----------------------------
# COMPUTE E(t)
# -----------------------------
df["Y_short"] = df[PRICE_COL].shift(SHORT_CYCLE)
df["Y_long"]  = df[PRICE_COL].shift(LONG_CYCLE)
df["E_t"] = ALPHA * df["Y_short"] + (1 - ALPHA) * df["Y_long"]
df = df.dropna(subset=["E_t"]).reset_index(drop=True)

# -----------------------------
# EQUILIBRIUM TREND
# -----------------------------
df["E_trend"] = np.nan
for i in range(EQUILIBRIUM_FRAME, len(df)):
    window = df["E_t"].iloc[i - EQUILIBRIUM_FRAME:i].values
    x = np.arange(EQUILIBRIUM_FRAME)
    coef = np.polyfit(x, window, POLY_ORDER)
    poly = np.poly1d(coef)
    df.loc[df.index[i], "E_trend"] = poly(EQUILIBRIUM_FRAME - 1)

df = df.dropna(subset=["E_trend"]).reset_index(drop=True)
df = df[df[DATE_COL] >= START_DATE].reset_index(drop=True)

# -----------------------------
# PLOT WINDOW
# -----------------------------
ymin = df[PRICE_COL].min() * 0.95
ymax = df[PRICE_COL].max() * 1.05
arrow_offset = (ymax - ymin) * 0.03

frames = FPS * TARGET_SECONDS
FRAME_STEP = int(np.ceil(len(df) / frames))

x_start = START_DATE
x_end   = df[DATE_COL].iloc[-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
ax.set_facecolor("black")

# -----------------------------
# STATE
# -----------------------------
position = None
last_flat_idx = 0
long_entries, short_entries, exits = [], [], []

# -----------------------------
# UPDATE
# -----------------------------
def update(frame_idx):
    global position, last_flat_idx

    ax.clear()
    ax.set_facecolor("black")

    t = min(len(df), frame_idx * FRAME_STEP)
    d = df.iloc[:t]
    if d.empty:
        return

    eq = d["E_trend"].values

    upper_entry = eq * (1 + BAND_SHORT_ENTRY)
    lower_entry = eq * (1 - BAND_LONG_ENTRY)
    upper_exit  = eq * (1 + BAND_SHORT_EXIT) if BAND_SHORT_EXIT is not None else eq
    lower_exit  = eq * (1 - BAND_LONG_EXIT)  if BAND_LONG_EXIT  is not None else eq

    high_px = d[HIGH_COL].iloc[-1]
    low_px  = d[LOW_COL].iloc[-1]
    date = d[DATE_COL].iloc[-1]

    if position is None and (d[PRICE_COL].values[last_flat_idx:] < lower_entry[last_flat_idx:]).any():
        long_entries.append((date, lower_entry[-1] - arrow_offset))
        position = "LONG"

    elif position == "LONG" and high_px >= lower_exit[-1]:
        exits.append((date, lower_exit[-1], COLOR_LOWER))
        position = None
        last_flat_idx = len(d) - 1

    if position is None and (d[PRICE_COL].values[last_flat_idx:] > upper_entry[last_flat_idx:]).any():
        short_entries.append((date, upper_entry[-1] + arrow_offset))
        position = "SHORT"

    elif position == "SHORT" and low_px <= upper_exit[-1]:
        exits.append((date, upper_exit[-1], COLOR_UPPER))
        position = None
        last_flat_idx = len(d) - 1

    # -----------------------------
    # DRAW
    # -----------------------------
    ax.plot(d[DATE_COL], d[PRICE_COL], color=COLOR_PRICE, lw=1.1)
    ax.plot(d[DATE_COL], eq, color=COLOR_EQ, lw=2.0)
    ax.scatter(d[DATE_COL].iloc[-1], eq[-1], color=COLOR_EQ, s=40, zorder=9)

    ax.plot(d[DATE_COL], upper_entry, color=COLOR_UPPER, lw=1.2)
    ax.plot(d[DATE_COL], lower_entry, color=COLOR_LOWER, lw=1.2)

    if BAND_SHORT_EXIT is not None:
        ax.plot(d[DATE_COL], upper_exit, color=COLOR_EXIT, lw=1.0, alpha=0.5)
    if BAND_LONG_EXIT is not None:
        ax.plot(d[DATE_COL], lower_exit, color=COLOR_EXIT, lw=1.0, alpha=0.5)

    if long_entries:
        dts, prs = zip(*long_entries)
        ax.scatter(dts, prs, marker="^", s=120, color=COLOR_LOWER, zorder=7)
    if short_entries:
        dts, prs = zip(*short_entries)
        ax.scatter(dts, prs, marker="v", s=120, color=COLOR_UPPER, zorder=7)
    if exits:
        dts, prs, cols = zip(*exits)
        ax.scatter(dts, prs, marker="*", s=260, color=cols, zorder=8)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(ymin, ymax)

    if X_TICK_MODE == "Q":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_title(
        f"{SYMBOL} Daily Close Price vs Alersal™ TF-Band Equilibrium",
        color="white", fontsize=12
    )

    ax.tick_params(colors="#DDDDDD")
    for spine in ax.spines.values():
        spine.set_color("#DDDDDD")

    # -----------------------------
    # LEGEND (FIXED)
    # -----------------------------
    close_long = plt.Line2D([], [], marker="*", linestyle="None",
                            markersize=12, markerfacecolor=COLOR_LOWER,
                            markeredgewidth=0)

    close_short = plt.Line2D([], [], marker="*", linestyle="None",
                             markersize=12, markerfacecolor=COLOR_UPPER,
                             markeredgewidth=0)

    ax.legend(
        handles=[
            plt.Line2D([], [], color=COLOR_PRICE, lw=1.1),
            plt.Line2D([], [], color=COLOR_EQ, lw=2.0),
            (
                plt.Line2D([], [], marker="^", linestyle="None",
                           markerfacecolor=COLOR_LOWER, markersize=9, markeredgewidth=0),
                plt.Line2D([], [], marker="v", linestyle="None",
                           markerfacecolor=COLOR_UPPER, markersize=9, markeredgewidth=0),
            ),
            (close_long, close_short),
        ],
        labels=[
            f"{SYMBOL} Daily Close Price",
            "TF-Band Equilibrium",
            "Long / Short Markers",
            "Close Position Markers",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        labelcolor="#DDDDDD"
    )

    ax.grid(alpha=0.05)

# -----------------------------
# RUN
# -----------------------------
ani = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
ani.save(OUTPUT_FILE, fps=FPS, dpi=150, codec="h264")
plt.close()
