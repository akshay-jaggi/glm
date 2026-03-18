# GLM Research Notes
## For building neural GLMs on 2P imaging anndata from mouse_imaging codebase

---

## 1. Anndata Structure (from `mouse_imaging`)

### 1.1 File Locations

Preprocessed anndata files are stored at:
```
/n/data2/hms/neurobio/harvey/akshay/data/imaging/{mouse}/{date}/session_1/adata_maximin.h5ad
```
- Also: `adata_allsources_maximin.h5ad` (before cell selection)
- Example mice: `AJ016`, `AJ017`, `AJ018`, `AJ029`, `AJ032`, `AJ033`, `AJ050`, `AJ058`, `AJ065`, `AJ067`
- Example session dir: `/n/data2/hms/neurobio/harvey/akshay/data/imaging/AJ065/260129/session_1/`

Session directories contain:
```
adata_maximin.h5ad              # Anndata after cell selection
adata_allsources_maximin.h5ad   # Anndata with all sources (before cell selection)
metadata.pickle                 # Session metadata
path.pickle                     # Path dictionary
session.pickle                  # Full Session object
virmen/                         # Virmen behavioral data (.mat files)
sync/                           # Sync files (.abf or .h5)
meanRef/                        # Mean reference images
slice_*/                        # Suite2p output per plane
```

### 1.2 Loading Data

```python
from mouse_imaging import session as sess, options

# Load a single session
adata = sess.load_as_anndata(mouse='AJ065', date='260129', session='session_1',
                              ops=options.default_ops())

# Load multiple sessions
adatas = sess.load_imaging_sessions(mouse='AJ065', 
                                     dates=['260129', '260130', '260202'])
```

### 1.3 X Matrix (`.X`)

- **Shape**: `(n_timepoints, n_cells)` — rows are time bins, columns are cells/neurons
- **Content**: Deconvolved calcium activity (OASIS deconvolution output, stored as `S` in `Fc_oasis_mat`)
- Named `oasis` signal internally; represents estimated spike rates
- The deconvolution uses `tau_s = 0.80` (calcium indicator decay constant)
- Baseline method: `'maximin'` (sliding window min-max filter)

### 1.4 Layers

```python
adata.layers['dF']          # dF/F fluorescence (neuropil-subtracted)
adata.layers['dcnv']        # Same as X — deconvolved activity (OASIS)
adata.layers['dcnv_norm']   # Normalized deconvolved activity (added during processing)
```

### 1.5 obs Variables (Observation/Behavioral — per timepoint)

The `.obs` DataFrame has one row per imaging frame (volume). Key variables:

#### Time & Sync
| Variable | Description |
|----------|-------------|
| `t` | Time in seconds (from sync file) |
| `dt` | Time step between frames |
| `pitch` | Ball pitch velocity (from sync, `Ball_pitc` channel) — forward/backward running |
| `roll` | Ball roll velocity (from sync, `Ball_roll` channel) — lateral movement |
| `yaw` | Ball yaw velocity (from sync, `Ball_yaw` channel) — rotational movement |
| `sync_reward` | Reward signal from sync file |

#### VR Position & Movement (from Virmen, subsampled to imaging rate)
| Variable | Description |
|----------|-------------|
| `x` | X position in VR (lateral, flipped: `x *= -1`) |
| `y` | Y position in VR (forward along maze stem) |
| `h` | Heading angle (flipped: `h *= -1`, wrapped to [-π, π]) |
| `h_int` | Raw integrated heading from Virmen |
| `dx` | X velocity |
| `dy` | Y velocity (forward speed) |
| `dh` | Angular velocity (turning rate) |
| `world_id` | Integer world ID from Virmen (maps to maze condition) |
| `inITI` | Boolean: whether currently in inter-trial interval |
| `reward` | Binary reward signal |
| `lick` | Binary lick signal |
| `trial` | Trial number (integer, extended into ITI) |

#### Trial-Level Variables (repeated for each timepoint within a trial)
| Variable | Description |
|----------|-------------|
| `world` | Categorical: world name, e.g. `'white_right'`, `'black_left'`, `'white2_right'`, `'black2_left'` |
| `turn` | Categorical: actual turn direction, `'left'` or `'right'` |
| `rewarded_side` | Categorical: which side is rewarded, `'left'` or `'right'` |
| `correct` | Boolean: whether the mouse turned to the correct side |
| `accuracy` | Rolling 20-trial mean of `correct` |
| `target_x` | X-coordinate of reward zone (-12 for left, +12 for right) |
| `prev_correct` | Previous trial's correctness |
| `prev_world` | Previous trial's world |
| `trial_len_s` | Trial length in seconds (excluding ITI) |
| `rewarded_trial` | Whether reward was delivered on this trial |
| `trial_type` | Categorical string: e.g. `'white_right_correct_+reward'` |
| `wall_color` | First 5 chars of world: `'white'` or `'black'` |

#### Goal-Related Variables (computed by `tmaze.compute_goal_variables`)
| Variable | Description |
|----------|-------------|
| `target_y` | Y-coordinate of reward zone (clipped to reward zone range) |
| `target_dist` | Euclidean distance to target (NaN during ITI) |
| `dtarget_dist` | Rate of change of target distance |
| `ddtarget_dist` | Acceleration of target distance |
| `h_target` | Heading angle toward target |
| `h_error` | Heading error = `wrap(h - h_target)` |
| `abs_h_error` | Absolute heading error |
| `dh_error` | Rate of change of absolute heading error |
| `ddh_error` | Acceleration of heading error |

#### Movement Variables (computed by `tmaze.compute_movement_variables`)
| Variable | Description |
|----------|-------------|
| `ddh` | Angular acceleration (gradient of `dh`) |
| `ddh_0.25sigma` | Smoothed angular acceleration (Gaussian σ=0.25s) |
| `abs_ddh` | Absolute angular acceleration |

#### ITI Variables
| Variable | Description |
|----------|-------------|
| `ITI_correct` | Categorical: `'notITI'`, `'ITI_correct'`, or `'ITI_incorrect'` |
| `trial_type_ITI` | Trial type with ITI suffix |

#### Median Trajectory Variables (if `do_median_trajectory=True`)
| Variable | Description |
|----------|-------------|
| `x_mt`, `y_mt`, `h_mt` | Median trajectory position/heading values |
| `dx_mt`, `dy_mt`, `dh_mt` | Median trajectory velocity values |
| `x_mt_error`, `y_mt_error`, `h_mt_error` | Error from median trajectory |
| `dx_mt_error`, `dy_mt_error`, `dh_mt_error` | Velocity error from median trajectory |

### 1.6 var Variables (Cell-Level — per neuron)

The `.var` DataFrame has one row per cell. Index format: `'plane{N}_source{M}'`.

| Variable | Description |
|----------|-------------|
| `G` | Mean green channel fluorescence intensity (functional channel) |
| `B` | Mean blue channel intensity (if available — e.g., Sst44-nlsBFP) |
| `R` | Mean red channel intensity (if available — e.g., SstCre-RFP, ChRmine) |
| `G_spatial_corr` | Spatial correlation with green cellpose segmentation |
| `B_spatial_corr` | Spatial correlation with blue cellpose segmentation |
| `R_spatial_corr` | Spatial correlation with red cellpose segmentation |
| `isnotclipped` | Boolean: cell fluorescence not clipped at detector max |
| `isnotnearedge` | Boolean: cell center > 10 px from FOV edge |
| `iscell` | Boolean: classified as cell by convnet classifier |
| `y` | Median y-pixel of cell in FOV |
| `x` | Median x-pixel of cell in FOV |
| `area` | Brain area (for MROI sessions): `'PPC'`, `'V1'`, `'RSC'` |

#### Cell Type Variables (added by `options.call_celltypes_*` functions)
| Variable | Description |
|----------|-------------|
| `B+` | Boolean: blue-positive (e.g., B channel > threshold) |
| `R+` | Boolean: red-positive (e.g., R channel > threshold or high spatial correlation) |
| `celltype` | Categorical: e.g. `'B+R+'`, `'R+'`, `'B+'`, `'Negative'` |
| `Sst44+` | Boolean: Sst44 positive (= `B+`) |
| `R+Sst44-` | Boolean: Red positive but not Sst44 |
| `Nonlabeled` | Boolean: Neither R+ nor B+ |

**Cell type labeling logic** (from `options.label_cell`):
- A cell's label is built from its positive markers: e.g., if `B+` and `R+` are both True → `'B+R+'`
- If no markers are positive → `'Negative'`
- Thresholds are computed relative to top-N brightest cells (e.g., `n_top_B=3`, `min_B_rel=0.15`)

### 1.7 uns (Unstructured Data)

```python
adata.uns['metadata']   # Dict with: mouse, date, session, nslices, Ly, Lx, volume_rate, dt, region, maze, ...
adata.uns['path']        # Dict with all file paths
adata.uns['ops']         # Processing options dict
adata.uns['vr']          # Full VR dataframe at Virmen rate (not subsampled)
adata.uns['trials']      # Trials DataFrame (one row per trial)
adata.uns['triggers']    # Trigger timestamps
adata.uns['median_trajectory']  # Median trajectory DataFrame
```

### 1.8 World/Maze Structure

For `cued_tmaze` (most common):
```python
world_dict = {
    '1': 'white_right',   # White walls, reward on right
    '2': 'black_left',    # Black walls, reward on left
    '3': 'white2_right',  # White variant 2, reward on right
    '4': 'black2_left',   # Black variant 2, reward on left
}
```

For `tower_tmaze`:
```python
world_dict = {'1': 'tower_right', '2': 'tower_left'}
```

### 1.9 Trial Structure

Each trial consists of:
1. **Stem traversal**: Mouse runs forward along y-axis (`inITI=0`)
2. **Turn**: Mouse turns left or right at the T-junction
3. **ITI**: Inter-trial interval after reaching the end (`inITI=1`)

Trial variables are computed per trial in `tmaze.compute_trials_df()` and then repeated to the timebase via `tmaze.map_trial_variables_to_time()`.

### 1.10 Imaging Frame Rate

- Typical `volume_rate` ≈ 7.5 Hz (volumetric imaging)
- `dt` ≈ 0.133 s between frames
- Stored in `adata.uns['metadata']['volume_rate']` and `adata.uns['metadata']['dt']`

### 1.11 Virmen Mat Columns

The raw Virmen data columns are:
```python
['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'trial']
```

---

## 2. Nemos GLM API (v0.2.7)

**Package location**: `/n/data2/hms/neurobio/harvey/akshay/conda/envs/glm/lib/python3.11/site-packages/nemos/`

**Conda env**: `/n/data2/hms/neurobio/harvey/akshay/conda/envs/glm`

**Activation**: `module load conda/miniforge3 && conda activate /n/data2/hms/neurobio/harvey/akshay/conda/envs/glm`

### 2.1 Key Imports

```python
import nemos as nmo
import jax
import jax.numpy as jnp

# Enable float64 for numerical stability
jax.config.update("jax_enable_x64", True)
```

### 2.2 Basis Functions

Nemos provides two types of basis: **Eval** (for instantaneous/static features) and **Conv** (for time-history convolution).

#### Available Basis Classes

| Class | Description | Key Params |
|-------|-------------|------------|
| `nmo.basis.RaisedCosineLogConv` | Log-spaced raised cosine bumps (Conv) | `n_basis_funcs`, `window_size`, `width=2.0`, `time_scaling=50.0`, `enforce_decay_to_zero=True` |
| `nmo.basis.RaisedCosineLogEval` | Log-spaced raised cosine bumps (Eval) | `n_basis_funcs`, `width=2.0`, `time_scaling=50.0` |
| `nmo.basis.RaisedCosineLinearConv` | Linearly-spaced raised cosine bumps (Conv) | `n_basis_funcs`, `window_size`, `width=2.0` |
| `nmo.basis.RaisedCosineLinearEval` | Linearly-spaced raised cosine bumps (Eval) | `n_basis_funcs`, `width=2.0` |
| `nmo.basis.BSplineConv` | B-spline basis (Conv) | `n_basis_funcs`, `window_size`, `order=4` |
| `nmo.basis.BSplineEval` | B-spline basis (Eval) | `n_basis_funcs`, `order=4`, `bounds=None` |
| `nmo.basis.MSplineConv` | M-spline basis (Conv) | `n_basis_funcs`, `window_size`, `order=2` |
| `nmo.basis.MSplineEval` | M-spline basis (Eval) | `n_basis_funcs`, `order=2` |
| `nmo.basis.HistoryConv` | Identity/history convolution | `window_size` |
| `nmo.basis.IdentityEval` | Identity (passthrough) | (none) |

#### Creating and Using Bases

```python
# Create a basis for spike history (log-spaced cosine bumps)
spike_basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, 
    window_size=20,       # 20 time bins of history
    width=2.0,
    time_scaling=50.0,
    enforce_decay_to_zero=True,
    label="spike_history"
)

# Create a basis for a behavioral variable (B-spline, instantaneous)
position_basis = nmo.basis.BSplineEval(
    n_basis_funcs=10,
    order=4,
    label="position"
)

# Evaluate basis on a grid (useful for visualization)
sample_pts, basis_values = spike_basis.evaluate_on_grid(100)
# sample_pts: shape (100,), basis_values: shape (100, n_basis_funcs)

# Compute features from data
# For Conv basis: input is 1D signal, output is (n_samples, n_basis_funcs) convolution
features = spike_basis.compute_features(spike_train)  # spike_train shape: (n_samples,)

# For Eval basis: input is 1D signal, output is (n_samples, n_basis_funcs) evaluated at each sample
features = position_basis.compute_features(position)  # position shape: (n_samples,)
```

#### Combining Bases (Additive and Multiplicative)

```python
# Additive: concatenate feature columns
combined = basis_1 + basis_2  # AdditiveBasis
X = combined.compute_features(signal_1, signal_2)
# X shape: (n_samples, n_basis_1 + n_basis_2)

# Multiplicative: tensor product (interaction terms)
interaction = basis_1 * basis_2  # MultiplicativeBasis
X = interaction.compute_features(signal_1, signal_2)
# X shape: (n_samples, n_basis_1 * n_basis_2)
```

### 2.3 GLM Class

```python
model = nmo.glm.GLM(
    observation_model="Poisson",        # or "Gamma", "Gaussian", "Bernoulli", "NegativeBinomial"
    inverse_link_function=None,         # None = use default for obs model (exp for Poisson, 1/x for Gamma)
    regularizer=None,                   # None = UnRegularized. Or "Ridge", "Lasso", "GroupLasso", "ElasticNet"
    regularizer_strength=None,          # float, required for Ridge/Lasso/GroupLasso
    solver_name=None,                   # None = auto. Or "GradientDescent", "BFGS", "LBFGS", "ProximalGradient"
    solver_kwargs=None,                 # dict: e.g. {"stepsize": 0.01, "maxiter": 500}
)
```

#### Fitting

```python
# Single neuron GLM
model = nmo.glm.GLM(observation_model="Poisson")
model = model.fit(X, y)
# X: (n_time_bins, n_features) — design matrix
# y: (n_time_bins,) — spike counts or rates

# Access results
model.coef_       # shape (n_features,) — fitted coefficients
model.intercept_  # scalar — baseline rate parameter
model.scale_      # scale parameter
```

#### Prediction & Scoring

```python
predicted_rate = model.predict(X_new)  # shape (n_time_bins,)

# Score (log-likelihood or pseudo-R²)
ll = model.score(X, y, score_type="log-likelihood")
r2 = model.score(X, y, score_type="pseudo-r2-McFadden")
```

#### Observation Models & Default Links

| Observation Model | Default Inverse Link | Suitable For |
|-------------------|---------------------|--------------|
| `"Poisson"` | `exp(x)` | Spike counts |
| `"Gamma"` | `1/x` | Continuous positive (e.g., dF/F, deconvolved rates) |
| `"Gaussian"` | `x` (identity) | Continuous real-valued |
| `"Bernoulli"` | `1/(1+exp(-x))` (logistic) | Binary outcomes |
| `"NegativeBinomial"` | `exp(x)` | Overdispersed counts |

**For deconvolved calcium activity (continuous, non-negative)**: Use `"Gamma"` or `"Poisson"` with `jax.nn.softplus` as the inverse link.

```python
# Gamma GLM for deconvolved activity
model = nmo.glm.GLM(
    observation_model="Gamma",
    regularizer="Ridge",
    regularizer_strength=0.1,
    solver_kwargs={"maxiter": 500}
)

# Or Poisson with softplus (recommended for numerical stability)
model = nmo.glm.GLM(
    observation_model="Poisson",
    inverse_link_function=jax.nn.softplus,
    regularizer="Ridge",
    regularizer_strength=0.1,
)
```

### 2.4 PopulationGLM Class

Fits a GLM to all neurons simultaneously. Same interface as GLM but:
- `y` shape: `(n_time_bins, n_neurons)` — matrix of all neuron activities
- `coef_` shape: `(n_features, n_neurons)` — separate coefficients per neuron
- `intercept_` shape: `(n_neurons,)` — separate intercepts per neuron
- Supports `feature_mask` to select which features predict which neurons

```python
pop_model = nmo.glm.PopulationGLM(
    observation_model="Poisson",
    inverse_link_function=jax.nn.softplus,
    regularizer="GroupLasso",
    regularizer_strength=0.1,
    feature_mask=None,  # shape (n_features, n_neurons) or FeaturePytree
)

# Fit to population
pop_model = pop_model.fit(X, Y)
# X: (n_time_bins, n_features)
# Y: (n_time_bins, n_neurons)
```

#### Feature Mask

```python
import jax.numpy as jnp
# feature_mask[i, j] = 1 means feature i predicts neuron j
feature_mask = jnp.ones((n_features, n_neurons))  # all features predict all neurons

# Or use FeaturePytree for named features
from nemos.pytrees import FeaturePytree
feature_mask = FeaturePytree(
    position=jnp.ones(n_neurons),         # position predicts all neurons
    spike_history=jnp.eye(n_neurons),     # each neuron's own history only
)
```

### 2.5 Regularization

| Regularizer | Description | Solver | Key Params |
|-------------|-------------|--------|------------|
| `"UnRegularized"` | No regularization | GradientDescent, BFGS, LBFGS | — |
| `"Ridge"` | L2 penalty | GradientDescent, BFGS, LBFGS | `regularizer_strength` |
| `"Lasso"` | L1 penalty (sparse) | ProximalGradient | `regularizer_strength` |
| `"GroupLasso"` | Group L1 penalty | ProximalGradient | `regularizer_strength`, `mask` |
| `"ElasticNet"` | L1 + L2 | ProximalGradient | `regularizer_strength` |

#### GroupLasso Mask

```python
from nemos.regularizer import GroupLasso
import numpy as np

# Define groups: mask shape (n_groups, n_features)
mask = np.zeros((3, n_features))
mask[0, 0:5] = 1    # Group 0: features 0-4 (e.g., position basis)
mask[1, 5:10] = 1   # Group 1: features 5-9 (e.g., speed basis)
mask[2, 10:15] = 1  # Group 2: features 10-14 (e.g., heading basis)

group_lasso = GroupLasso(mask=mask)
model = nmo.glm.GLM(regularizer=group_lasso, regularizer_strength=0.1)
```

### 2.6 FeaturePytree (Named Design Matrix)

```python
from nemos.pytrees import FeaturePytree

X = FeaturePytree(
    position=position_features,    # shape (n_time, n_position_basis)
    speed=speed_features,          # shape (n_time, n_speed_basis)
    heading=heading_features,      # shape (n_time, n_heading_basis)
)

# Use directly in GLM
model = nmo.glm.PopulationGLM().fit(X, Y)
# model.coef_ will be a dict with keys: 'position', 'speed', 'heading'
```

### 2.7 Design Matrix Construction Workflow

```python
import nemos as nmo
import numpy as np

# 1. Define bases for each predictor
position_basis = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, label="position")
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, label="speed")
heading_basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=8, label="heading")
spike_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=8, window_size=20, label="spike_history")

# 2. Create additive basis (combines features)
full_basis = position_basis + speed_basis + heading_basis + spike_basis

# 3. Compute design matrix from data
X = full_basis.compute_features(position, speed, heading, spike_train)
# X shape: (n_time, 10 + 8 + 8 + 8) = (n_time, 34)

# 4. Split features back for interpretation
feature_dict = full_basis.split_by_feature(X, axis=1)
# Returns dict: {'position': (n_time, 10), 'speed': (n_time, 8), ...}
```

### 2.8 Solver Options

```python
# Gradient descent (default for UnRegularized/Ridge)
model = nmo.glm.GLM(solver_name="GradientDescent", 
                     solver_kwargs={"stepsize": 0.01, "maxiter": 500})

# L-BFGS (fast quasi-Newton, good for Ridge/UnRegularized)
model = nmo.glm.GLM(solver_name="LBFGS", 
                     solver_kwargs={"maxiter": 500, "tol": 1e-5})

# Proximal gradient (required for Lasso/GroupLasso)
model = nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.1,
                     solver_name="ProximalGradient",
                     solver_kwargs={"stepsize": 0.01, "maxiter": 500})

# SVRG (for very large models with batching)
model = nmo.glm.GLM(solver_name="SVRG",
                     solver_kwargs={"batch_size": 256, "stepsize": 0.001})
```

---

## 3. Integration Plan: Anndata → Nemos GLM

### 3.1 Data Preparation Pipeline

```python
import anndata
import numpy as np
import nemos as nmo
import jax
import jax.numpy as jnp
from mouse_imaging import session as sess, options, analysis as an

jax.config.update("jax_enable_x64", True)

# ---- Step 1: Load anndata ----
adata = sess.load_as_anndata(mouse='AJ065', date='260129')

# ---- Step 2: Select cells ----
adata = sess.select_cells(adata)  # removes clipped, near-edge, non-cell sources

# ---- Step 3: Filter timepoints ----
# Remove ITI periods and trials that are too long
obs_mask = (~adata.obs['inITI'].astype(bool)) & (adata.obs['trial_len_s'] < 60)
adata_filtered = adata[obs_mask]

# ---- Step 4: Extract behavioral variables for design matrix ----
y_position = np.array(adata_filtered.obs['y'], dtype=float)
speed = np.array(adata_filtered.obs['dy'], dtype=float)
heading = np.array(adata_filtered.obs['h'], dtype=float)
turn_velocity = np.array(adata_filtered.obs['dh'], dtype=float)
heading_error = np.array(adata_filtered.obs['h_error'], dtype=float)
world = adata_filtered.obs['world']  # categorical

# Additional potential predictors
pitch = np.array(adata_filtered.obs['pitch'], dtype=float)  # forward running (from sync)
roll = np.array(adata_filtered.obs['roll'], dtype=float)    # lateral movement (from sync)
yaw = np.array(adata_filtered.obs['yaw'], dtype=float)      # rotation (from sync)
lick = np.array(adata_filtered.obs['lick'], dtype=float)
reward = np.array(adata_filtered.obs['reward'], dtype=float)

# ---- Step 5: Extract neural activity (target) ----
Y = np.array(adata_filtered.layers['dcnv_norm'], dtype=float)  # (n_time, n_cells)
# Or for single neuron:
y_single = Y[:, 0]
```

### 3.2 Design Matrix Construction

```python
# ---- Define bases ----
# Position along maze stem
pos_basis = nmo.basis.BSplineEval(n_basis_funcs=15, order=4, label="y_position")

# Forward speed
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, label="speed")

# Heading / angular velocity
heading_basis = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, label="heading")
turn_basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, label="turn_velocity")

# Pitch/Roll/Yaw from ball (convolutional — they have temporal dynamics)
pitch_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=6, window_size=15, label="pitch")
roll_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=6, window_size=15, label="roll")
yaw_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=6, window_size=15, label="yaw")

# Spike history (autoregressive coupling)
spike_hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=8, window_size=20, label="spike_history")

# Trial events (reward, lick)
reward_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=30, label="reward")
lick_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=15, label="lick")

# ---- Combine bases ----
full_basis = (pos_basis + speed_basis + heading_basis + turn_basis + 
              pitch_basis + roll_basis + yaw_basis + reward_basis + lick_basis)

# ---- Compute features ----
X = full_basis.compute_features(
    y_position, speed, heading, turn_velocity,
    pitch, roll, yaw, reward, lick
)

# For spike history, add per-neuron:
# spike_features = spike_hist_basis.compute_features(y_single)
# X_with_history = np.concatenate([X, spike_features], axis=1)
```

### 3.3 Handling Categorical Variables (World)

```python
# One-hot encode world as separate indicator variables
world_dummies = pd.get_dummies(adata_filtered.obs['world'], prefix='world')
# Or: create interaction basis = position × world
# This captures different place fields for different maze conditions

# For position × world interaction:
for world_name in adata_filtered.obs['world'].cat.categories:
    world_mask = (adata_filtered.obs['world'] == world_name).astype(float).values
    pos_in_world = y_position * world_mask
    # Apply position basis to each
```

### 3.4 Fitting Strategy

```python
# ---- Single neuron GLM ----
model = nmo.glm.GLM(
    observation_model="Poisson",
    inverse_link_function=jax.nn.softplus,
    regularizer="Ridge",
    regularizer_strength=0.1,
    solver_name="LBFGS",
    solver_kwargs={"maxiter": 500}
)
model = model.fit(X, y_single)

# ---- Population GLM ----
pop_model = nmo.glm.PopulationGLM(
    observation_model="Poisson",
    inverse_link_function=jax.nn.softplus,
    regularizer="GroupLasso",
    regularizer_strength=0.1,
)
pop_model = pop_model.fit(X, Y)

# ---- Evaluate ----
r2 = model.score(X, y_single, score_type="pseudo-r2-McFadden")
predicted = model.predict(X)
```

### 3.5 Key Considerations

1. **NaN handling**: Nemos silently handles NaN in X by filtering. Make sure to clean behavioral variables that contain NaN (especially `target_dist`, `h_error` during ITI).

2. **Temporal continuity**: When filtering out ITI periods, the time series becomes discontinuous. Conv bases will convolve across breaks. Consider fitting per-trial or padding with NaN at trial boundaries.

3. **Choosing observation model**:
   - `dcnv` (OASIS deconvolved): non-negative, sparse → `"Poisson"` with `softplus` link
   - `dcnv_norm`: normalized version, still non-negative → same as above
   - `dF` (dF/F): can be negative → `"Gaussian"` or `"Gamma"` (after offsetting)

4. **Frame rate**: ~7.5 Hz means each time bin is ~133 ms. Window sizes for Conv bases should account for this (e.g., `window_size=20` ≈ 2.67 s of history).

5. **Cell type analysis**: After fitting, compare GLM coefficients between cell types:
   ```python
   sst_cells = adata.var['celltype'] == 'B+R+'  # Sst44 cells
   non_labeled = adata.var['celltype'] == 'Negative'
   ```

6. **Cross-validation**: Split by trials (not by time points) to avoid temporal leakage:
   ```python
   trial_ids = adata_filtered.obs['trial'].unique()
   train_trials = trial_ids[:int(0.8 * len(trial_ids))]
   test_trials = trial_ids[int(0.8 * len(trial_ids)):]
   train_mask = adata_filtered.obs['trial'].isin(train_trials)
   ```

7. **Multiple sessions**: Concatenate anndata objects across sessions:
   ```python
   adatas = sess.load_imaging_sessions(mouse='AJ065', dates=[...])
   adata_cat = anndata.concat(adatas, merge='first', index_unique='-')
   ```

### 3.6 Summary of Recommended Predictors for T-Maze GLM

| Predictor | Source | Basis Type | Rationale |
|-----------|--------|-----------|-----------|
| `y` (maze position) | `adata.obs['y']` | BSplineEval | Place fields |
| `dy` (forward speed) | `adata.obs['dy']` | BSplineEval | Speed modulation |
| `h` (heading) | `adata.obs['h']` | BSplineEval (cyclic) | Head direction |
| `dh` (angular velocity) | `adata.obs['dh']` | BSplineEval | Turn-related activity |
| `h_error` (heading error) | `adata.obs['h_error']` | BSplineEval | Error correction signals |
| `pitch` | `adata.obs['pitch']` | RaisedCosineLogConv | Running dynamics |
| `roll` | `adata.obs['roll']` | RaisedCosineLogConv | Lateral movement |
| `yaw` | `adata.obs['yaw']` | RaisedCosineLogConv | Rotational dynamics |
| `reward` | `adata.obs['reward']` | RaisedCosineLogConv | Reward response |
| `lick` | `adata.obs['lick']` | RaisedCosineLogConv | Lick-related activity |
| `world` | `adata.obs['world']` | Categorical / interaction | Context-dependent coding |
| `correct` | `adata.obs['correct']` | Binary indicator | Choice-related activity |
| Spike history | `adata.layers['dcnv']` | RaisedCosineLogConv | Autoregressive dynamics |
