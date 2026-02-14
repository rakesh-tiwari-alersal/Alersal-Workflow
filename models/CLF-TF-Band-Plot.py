import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple

# -----------------------------
# FORCE FFMPEG
# -----------------------------
mpl.rcParams['animation.writer'] = 'ffmpeg'

# -----------------------------
# USER SETTINGS (PER SECURITY)
# -----------------------------
FILE_PATH = "historical_data/CLF.csv"
OUTPUT_FILE = "CLF-TF-Band-Plot.mp4"

START_DATE = "2023-07-01"
END_DATE   = "2026-02-10"

SHORT_CYCLE = 17
LONG_CYCLE  = 510
ALPHA = 0.805

EQUILIBRIUM_FRAME = 1326

X_TICK_MODE = "Q"   # "Q" = quarterly, "Y" = yearly

# ENTRY bands (renamed)
BAND_SHORT_ENTRY = 0.1459
BAND_LONG_ENTRY  = 0.1459

# EXIT bands (optional; None → defaults to equilibrium)
BAND_SHORT_EXIT = 0.0344
BAND_LONG_EXIT  = None

FPS = 24
POLY_ORDER = 3

# -----------------------------
# COLORS
# -----------------------------
COLOR_EQ     = "#4FC3F7"
COLOR_UPPER  = "#2ECC71"
COLOR_LOWER  = "#F11B1BEF"
COLOR_EXIT   = "#9E9E9E"
COLOR_PRICE  = "#CFCFCF"

# -----------------------------
# LOAD DATA (CSV, ROBUST)
# -----------------------------
df = pd.read_csv(FILE_PATH)
df.columns = [c.strip() for c in df.columns]

def _col(name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise ValueError(f"Required column '{name}' not found")

DATE_COL  = _col("Date")
PRICE_COL = _col("Close")
HIGH_COL  = _col("High")
LOW_COL   = _col("Low")

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df[df[DATE_COL] <= END_DATE].reset_index(drop=True)

# -----------------------------
# COMPUTE E(t)  (UNCHANGED)
# -----------------------------
df["Y_short"] = df[PRICE_COL].shift(SHORT_CYCLE)
df["Y_long"]  = df[PRICE_COL].shift(LONG_CYCLE)
df["E_t"] = ALPHA * df["Y_short"] + (1 - ALPHA) * df["Y_long"]
df = df.dropna(subset=["E_t"]).reset_index(drop=True)

# -----------------------------
# PRECOMPUTE EQUILIBRIUM TREND (UNCHANGED)
# -----------------------------
df["E_trend"] = np.nan
for i in range(EQUILIBRIUM_FRAME, len(df)):
    window = df["E_t"].iloc[i - EQUILIBRIUM_FRAME:i].values
    x = np.arange(EQUILIBRIUM_FRAME)
    coef = np.polyfit(x, window, POLY_ORDER)
    poly = np.poly1d(coef)
    df.loc[df.index[i], "E_trend"] = poly(EQUILIBRIUM_FRAME - 1)

df = df.dropna(subset=["E_trend"]).reset_index(drop=True)

# -----------------------------
# FINAL PLOTTING WINDOW
# -----------------------------
df = df[df[DATE_COL] >= START_DATE].reset_index(drop=True)

ymin = df[PRICE_COL].min() * 0.95
ymax = df[PRICE_COL].max() * 1.05
arrow_offset = (ymax - ymin) * 0.03

# -----------------------------
# ANIMATION TIMING
# -----------------------------
TARGET_SECONDS = 30
frames = FPS * TARGET_SECONDS
FRAME_STEP = int(np.ceil(len(df) / frames))

x_start = pd.to_datetime(START_DATE)
x_end   = df[DATE_COL].iloc[-1]

# -----------------------------
# FIGURE
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
ax.set_facecolor("black")

# -----------------------------
# STATE + MARKERS
# -----------------------------
position = None
last_flat_idx = 0
long_entries  = []
short_entries = []
exits = []   # (date, price, color)

def update(frame_idx):
    global position, last_flat_idx

    ax.clear()
    ax.set_facecolor("black")

    t = min(len(df), frame_idx * FRAME_STEP)
    d = df.iloc[:t]
    if len(d) == 0:
        return

    # PRICE
    ax.plot(d[DATE_COL], d[PRICE_COL],
            color=COLOR_PRICE, linewidth=1.1,
            label="Settlement Price")

    # EQUILIBRIUM
    eq = d["E_trend"].values
    ax.plot(d[DATE_COL], eq,
            color=COLOR_EQ, linewidth=2.0,
            label="TF-Band Equilibrium")

    # Equilibrium end knob
    ax.scatter(d[DATE_COL].iloc[-1], eq[-1],
               color=COLOR_EQ, s=40, zorder=9)

    # -----------------------------
    # FOUR BANDS
    # -----------------------------
    upper_entry = eq * (1 + BAND_SHORT_ENTRY)
    lower_entry = eq * (1 - BAND_LONG_ENTRY)

    upper_exit = eq * (1 + BAND_SHORT_EXIT) if BAND_SHORT_EXIT is not None else eq
    lower_exit = eq * (1 - BAND_LONG_EXIT)  if BAND_LONG_EXIT  is not None else eq

    ax.plot(d[DATE_COL], upper_entry, color=COLOR_UPPER, linewidth=1.2)
    ax.plot(d[DATE_COL], lower_entry, color=COLOR_LOWER, linewidth=1.2)

    if BAND_SHORT_EXIT is not None:
        ax.plot(d[DATE_COL], upper_exit, color=COLOR_EXIT, linewidth=1.0, alpha=0.5)

    if BAND_LONG_EXIT is not None:
        ax.plot(d[DATE_COL], lower_exit, color=COLOR_EXIT, linewidth=1.0, alpha=0.5)

    close_px = d[PRICE_COL].iloc[-1]
    high_px  = d[HIGH_COL].iloc[-1]
    low_px   = d[LOW_COL].iloc[-1]

    date = d[DATE_COL].iloc[-1]

    # -----------------------------
    # STATE MACHINE (FIXED)
    # -----------------------------
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
    # MARKERS
    # -----------------------------
    if long_entries:
        dts, prs = zip(*long_entries)
        ax.scatter(dts, prs, marker='^', s=120, color=COLOR_LOWER, zorder=7)

    if short_entries:
        dts, prs = zip(*short_entries)
        ax.scatter(dts, prs, marker='v', s=120, color=COLOR_UPPER, zorder=7)

    if exits:
        dts, prs, cols = zip(*exits)
        ax.scatter(dts, prs, marker='*', s=260, color=cols, zorder=8)

    # -----------------------------
    # AXES
    # -----------------------------
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(ymin, ymax)

    if X_TICK_MODE == "Q":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.set_title(
        "Crude Oil — Settlement Price vs Alersal™ TF-Band Equilibrium",
        color="white", fontsize=12
    )

    ax.tick_params(colors="#DDDDDD")
    for spine in ax.spines.values():
        spine.set_color("#DDDDDD")

    # LEGEND (unchanged)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=COLOR_PRICE, lw=1.1),
            plt.Line2D([], [], color=COLOR_EQ, lw=2.0),
            (
                plt.Line2D([], [], marker='*', linestyle='None',
                           markerfacecolor=COLOR_LOWER, markeredgecolor=COLOR_LOWER, markersize=10),
                plt.Line2D([], [], marker='*', linestyle='None',
                           markerfacecolor=COLOR_UPPER, markeredgecolor=COLOR_UPPER, markersize=10)
            )
        ],
        labels=[
            "Settlement Price",
            "TF-Band Equilibrium",
            "Close Position Markers"
        ],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        labelcolor="#DDDDDD"
    )

    ax.grid(alpha=0.05)

# -----------------------------
# ANIMATION
# -----------------------------
ani = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
ani.save(OUTPUT_FILE, fps=FPS, dpi=150, codec="h264")
plt.close()
