import matplotlib.pyplot as plt

DDPM_FLOPS  = 1166540800
SMALL_FLOPS = 713838592
BIG_FLOPS   = 11751161856
VAR_FLOPS   = 12531927040

SMALL_FID = 0.6405
SMALL_DIFF_FID = 0.9716
SMALL_NOPERC_FID = 0.9597

BIG_FID = 0.6502
BIG_DIFF_FID = 0.7753

DDPM_FID = 14.7980
VAR_FID = 0.6875

# Point list with base jitter in x
points = [
    ("Small model", SMALL_FLOPS, SMALL_FID, 0.2, -0.05),
    ("Small model diff", SMALL_FLOPS, SMALL_DIFF_FID, 0.2, 0.00),      # upward label offset
    ("Small model no perception", SMALL_FLOPS, SMALL_NOPERC_FID, 0.2,  -0.2),  # downward label offset
    
    ("Big model", BIG_FLOPS, BIG_FID, -1.75, -0.1),
    ("Big model diff", BIG_FLOPS, BIG_DIFF_FID, -2, 0.0),
    
    ("DDPM", DDPM_FLOPS, DDPM_FID, 0.2, -2.0),
    ("VAR", VAR_FLOPS, VAR_FID, -0.3, 0.05)
]

# Prepare arrays for scatter
x, x_offset, y, labels, y_offsets = [], [], [], [], []
for name, flops, fid, jitter_x, jitter_y in points:
    x.append(flops / 1e9)
    x_offset.append(jitter_x)
    y.append(fid)
    labels.append(name)
    y_offsets.append(jitter_y)

plt.figure(figsize=(6*2, 4.5*2))
plt.scatter(x, y, s=80)

# Annotate with custom y-offsets
for label, xi, x_off, yi, yoff in zip(labels, x, x_offset, y, y_offsets):
    plt.text(xi + x_off, yi + yoff, f"{label}\nFID={yi:.2f}",
            ha="left", va="bottom", fontsize=9*2)

plt.xlim(0, 14)
plt.yscale('log')
plt.ylabel("FID (log scale)", fontsize=14*2)
plt.xlabel("FLOPs (billions)", fontsize=14*2)
plt.title("FID vs. FLOPs", fontsize=18*2)
plt.grid(True, which="both", linestyle="--", alpha=1)
plt.tight_layout()
plt.savefig('FLOPS_FID.png', dpi=300)
plt.show()