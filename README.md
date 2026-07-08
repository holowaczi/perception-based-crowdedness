# Pedestrian Dynamics – Analysis Pipeline

Analysis of crossing-flow pedestrian experiments at seven crossing angles
α ∈ {0°, 30°, 60°, 90°, 120°, 150°, 180°}.

---

## Files

| File | Language | Role |
|---|---|---|
| `compute.jl` | Julia | Compute v, ρ, a, Δθ, δ per agent per frame → CSV |
| `article_figures.py` | Python | Publication-quality figures (PDF) from the CSVs |

---

## Workflow

```
1. julia compute_vrho.jl     # produces csv_data/
2. python article_figures.py # produces article_figs/
```

---

## Input data

The trajectory files and metadata originate from:

> https://zenodo.org/records/5718431

- `trajectories/bf_CF<id>.csv` – raw trajectory files
- `file_detail.csv` – metadata: file index, crossing angle α, Ti, Tf

### Trajectory file column layout (1-based)

```
col 1        = time frame  (1 frame = 1/120 s)
col 4i-2     = unfiltered X of agent i
col 4i-1     = filtered   X of agent i   ← used
col 4i       = unfiltered Y of agent i
col 4i+1     = filtered   Y of agent i   ← used
```

---

## Step 1 – `compute.jl`

### Grouping

Agents are split into groups A and B by k-means (k=2) applied to
2-D unit vectors of each agent's total displacement (first → last valid frame).
For α = 0° all agents belong to a single group.

### Time window

Ti and Tf are read from `file_detail.csv`.
For α = 0° (where the CSV contains `xxx`) the fixed values Ti = 480, Tf = n_frames − 480 are used (4 s from each end at 120 fps).

### Rescaled time

```
t' = (t − Ti) / (Tf − Ti)  ∈ [0, 1]
```

Frames from Ti+1 to Tf are processed (Ti+1 ensures t−1 is always valid).

### Velocity

```
v_i(t) = ‖r_i(t) − r_i(t−1)‖ × 120   [m/s]
```

### Density methods

All methods use the inverse-distance formula:

```
ρ_i = Σ_{j ∈ N, j≠i}  w_ij / (‖r_i − r_j‖² + ε²)     ε = 0.1 m
```

| Method tag | Neighbour set N | w_ij inside FOV | w_ij outside FOV |
|---|---|---|---|
| `all` | all agents | 1 | 1 |
| `fov_210_c0` | all agents | 1 | 0 |
| `fov_210_c5` | all agents | 1 | 0.5 |

**FOV definition** – agent j is inside agent i's FOV if the angle between
i's movement direction and the vector i→j does not exceed the half-angle φ/2:

```
d̂_i · (r_j − r_i)/‖r_j − r_i‖  ≥  cos(φ/2)
```

where d̂_i = (r_i(t) − r_i(t−1)) / ‖…‖.
If agent i is stationary the `all` density is used as fallback.

The tag encodes the total FOV angle φ and c×10, e.g. `fov_210_c5` → φ=210°, c=0.5.

### Direction change Δθ

Signed angle between two consecutive displacement windows of width W = 60 frames (0.5 s):

```
d1 = r_i(t−W) − r_i(t−2W)
d2 = r_i(t)   − r_i(t−W)

Δθ_i(t) = atan2(d̂1 × d̂2,  d̂1 · d̂2)   ∈ (−180°, 180°]
```

Positive = anti-clockwise turn, negative = clockwise turn.

### Deviation from expected direction δ

Signed angle between agent i's current movement direction and the mean
displacement direction of its group (expected direction):

```
δ_i(t) = atan2(d̂_exp × d̂_cur,  d̂_exp · d̂_cur)   ∈ (−180°, 180°]
```

Positive = deviation to the left (anti-clockwise), negative = to the right.

### Output CSV format

One file per input trajectory, per density method:

```
csv_data/<method>/Angle_<α>/bf_CF<id>_vrho.csv
```

Columns:

```
t_prime      – rescaled time t'
v_i          – velocity of agent i      [m/s]
acc_i        – acceleration of agent i  [m/s²]
rho_i        – density of agent i       (method-specific)
dangle_i     – Δθ of agent i            [°, signed]
delta_i      – δ  of agent i            [°, signed]
```

---

## Step 2 – `article_figures.py`

Reads all CSVs and produces publication-quality PDF figures in `article_figs/`:

| File | Description |
|---|---|
| `Fig01_rho_vs_tprime.pdf` | ⟨ρ⟩ vs t', three density methods, one line per α |
| `Fig02_v_vs_rho_perangle.pdf` | Median v vs ρ, per-angle scatter, three density panels |
| `Fig03_v_rho_tcolor_perangle.pdf` | ⟨v⟩ vs ⟨ρ⟩ coloured by t', 7 panels (one per α) |
| `Fig04a_delta_vs_tprime_signed.pdf` | Signed ⟨δ₁⟩ and ⟨δ₂⟩ vs t' |
| `Fig04b_delta_vs_tprime_abs.pdf` | ⟨\|δ₁\|⟩ and ⟨\|δ₂\|⟩ vs t' |
| `Fig05_delta_vs_rho.pdf` | Median \|δ₁\| and \|δ₂\| vs ρ, 2×3 grid |
| `Fig06_delta_vs_v_tcolor.pdf` | ⟨\|δ₁\|⟩ and ⟨\|δ₂\|⟩ vs ⟨v⟩ coloured by t', grouped angles |
| `Fig07_delta1_rho_tcolor_perangle.pdf` | ⟨\|δ₁\|⟩ vs ⟨ρ⟩ coloured by t', 7 panels |
| `Fig08_delta2_rho_tcolor_perangle.pdf` | ⟨\|δ₂\|⟩ vs ⟨ρ⟩ coloured by t', 7 panels |
| `Fig09_pdf_acc_perangle.pdf` | KDE of acceleration distribution, all α |
| `Fig10_v_acc_vs_tprime.pdf` | ⟨v⟩ and ⟨a⟩ vs t' |
| `Fig11_acc_vs_rho_perangle.pdf` | Median a vs ρ, three density panels |

---

## Colour and marker conventions

| α | Colour | Marker |
|---|---|---|
| 0° | black `#000000` | ● circle |
| 30° | sky blue `#87CEEB` | ■ square |
| 60° | green `#009900` | ▲ triangle up |
| 90° | violet `#7700CC` | ◆ diamond |
| 120° | yellow `#FEE12B` | ▼ triangle down |
| 150° | blue `#0033CC` | ⬡ hexagon |
| 180° | red `#CC0000` | ★ star |

---

## Dependencies

### Julia
```
CSV, DataFrames, Statistics, LinearAlgebra, Clustering, Glob
```

### Python
```
numpy, pandas, matplotlib, scipy
```

Install: `pip install numpy pandas matplotlib scipy`

