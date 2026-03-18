#!/usr/bin/env python
"""
run_glm.py — Fit a Gamma GLM to neural calcium imaging data.

Generalised from the prototype notebook (glm_prototype.ipynb).
Reads a run-specification JSON to configure features, basis expansions,
binning, and model parameters.

Usage:
    python run_glm.py --mouse YRA084 --session 260228 --config config.json
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import anndata as ad
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "/n/data2/hms/neurobio/harvey/yasmine/data/imaging"
OUTPUT_ROOT = "/n/data2/hms/neurobio/harvey/akshay/data/glm"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(mouse: str, session: str, output_dir: str, data_path: str | None = None) -> ad.AnnData:
    """Load anndata from the standard imaging path, copying to *output_dir*.
    
    If *data_path* is given, load directly from that file instead.
    """
    if data_path is not None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Provided data path not found: {data_path}")
        adata = ad.read_h5ad(data_path)
    else:
        src_path = os.path.join(
            DATA_ROOT, mouse, session, "session_1", "adata_maximin.h5ad"
        )
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source h5ad not found: {src_path}")

        os.makedirs(output_dir, exist_ok=True)
        dst_path = os.path.join(output_dir, "adata_maximin.h5ad")

        if not os.path.exists(dst_path):
            logger.info("Copying h5ad to output directory …")
            shutil.copy2(src_path, dst_path)
        else:
            logger.info("h5ad already present in output directory, skipping copy.")

        adata = ad.read_h5ad(dst_path)
    logger.info(
        "Loaded adata: %s  (volume_rate=%.2f Hz)",
        adata.shape,
        adata.uns["metadata"]["volume_rate"],
    )
    return adata


# ---------------------------------------------------------------------------
# 2. Bin data
# ---------------------------------------------------------------------------
def bin_data(
    adata: ad.AnnData,
    n_spatial_bins: int = 100,
    n_temporal_bins: int = 50,
) -> ad.AnnData:
    """Spatially bin trial periods and temporally bin ITI periods."""

    y_min = adata.obs.loc[~adata.obs["inITI"], "y"].min()
    y_max = adata.obs.loc[~adata.obs["inITI"], "y"].max()
    spatial_edges = np.linspace(y_min, y_max, n_spatial_bins + 1)

    # Determine which obs columns are trial-level vs continuous
    trial_level_candidates = [
        "world", "turn", "rewarded_side", "correct", "prev_correct",
        "prev_world", "trial", "accuracy", "target_x", "trial_len_s",
        "rewarded_trial", "trial_type", "wall_color", "ITI_correct",
        "trial_type_ITI",
    ]
    continuous_candidates = [
        "pitch", "roll", "yaw", "x", "y", "h", "dx", "dy", "dh", "dt", "t",
        "sync_reward", "reward", "lick", "target_y", "target_dist",
        "dtarget_dist", "ddtarget_dist", "h_target", "h_error",
        "abs_h_error", "dh_error", "ddh_error", "ddh", "abs_ddh",
    ]
    trial_level_vars = [v for v in trial_level_candidates if v in adata.obs.columns]
    continuous_vars = [v for v in continuous_candidates if v in adata.obs.columns]

    binned_rows: list[dict] = []
    trial_ids = sorted(adata.obs["trial"].unique())
    logger.info("Binning %d trials …", len(trial_ids))

    for trial_id in trial_ids:
        trial_mask = adata.obs["trial"] == trial_id
        trial_adata = adata[trial_mask]

        # --- Trial portion: spatial binning ---
        not_iti = ~trial_adata.obs["inITI"].values
        if not_iti.sum() > 0:
            trial_y = trial_adata.obs.loc[not_iti, "y"].values
            bin_idx = np.clip(
                np.digitize(trial_y, spatial_edges) - 1, 0, n_spatial_bins - 1
            )
            for b in range(n_spatial_bins):
                bmask = bin_idx == b
                if bmask.sum() == 0:
                    continue
                row: dict = {}
                row["neural"] = np.asarray(
                    trial_adata[not_iti][bmask].X.mean(axis=0)
                ).flatten()
                for layer_name in trial_adata.layers:
                    row[f"layer_{layer_name}"] = np.asarray(
                        trial_adata[not_iti][bmask].layers[layer_name].mean(axis=0)
                    ).flatten()
                for var in continuous_vars:
                    vals = trial_adata.obs.loc[not_iti, var].values[bmask]
                    row[var] = np.nanmean(vals.astype(float))
                for var in trial_level_vars:
                    row[var] = trial_adata.obs.loc[not_iti, var].values[0]
                row["spatial_bin"] = b
                row["is_iti"] = False
                row["trial_id"] = trial_id
                binned_rows.append(row)

        # --- ITI portion: temporal binning ---
        iti = trial_adata.obs["inITI"].values
        if iti.sum() > 0:
            n_iti_frames = int(iti.sum())
            bin_edges_iti = np.linspace(
                0, n_iti_frames, n_temporal_bins + 1
            ).astype(int)
            for b in range(n_temporal_bins):
                start, end = bin_edges_iti[b], bin_edges_iti[b + 1]
                if end <= start:
                    continue
                idx_slice = np.arange(start, end)
                row = {}
                row["neural"] = np.asarray(
                    trial_adata[iti][idx_slice].X.mean(axis=0)
                ).flatten()
                for layer_name in trial_adata.layers:
                    row[f"layer_{layer_name}"] = np.asarray(
                        trial_adata[iti][idx_slice]
                        .layers[layer_name]
                        .mean(axis=0)
                    ).flatten()
                for var in continuous_vars:
                    vals = trial_adata.obs.loc[iti, var].values[idx_slice]
                    row[var] = np.nanmean(vals.astype(float))
                for var in trial_level_vars:
                    row[var] = trial_adata.obs.loc[iti, var].values[0]
                row["spatial_bin"] = b
                row["is_iti"] = True
                row["trial_id"] = trial_id
                binned_rows.append(row)

    # Reconstruct anndata
    neural_mat = np.vstack([r["neural"] for r in binned_rows])
    scalar_keys = [
        k for k in binned_rows[0]
        if k != "neural" and not k.startswith("layer_")
    ]
    obs_df = pd.DataFrame({k: [r.get(k, np.nan) for r in binned_rows] for k in scalar_keys})
    layers_dict = {}
    layer_names = [
        k.replace("layer_", "") for k in binned_rows[0] if k.startswith("layer_")
    ]
    for ln in layer_names:
        layers_dict[ln] = np.vstack([r[f"layer_{ln}"] for r in binned_rows])

    adata_binned = ad.AnnData(
        X=neural_mat,
        obs=obs_df,
        var=adata.var.copy(),
        layers=layers_dict,
        uns=adata.uns.copy(),
    )
    logger.info("Binned adata shape: %s", adata_binned.shape)
    return adata_binned


# ---------------------------------------------------------------------------
# 3. Build design matrix
# ---------------------------------------------------------------------------

# -- helpers for encoding trial-level variables --

_CATEGORICAL_VARS = {"world", "prev_world"}
_BINARY_VARS = {"turn", "correct", "prev_correct"}
_DROP_NONE = {"prev_world", "prev_correct"}


def _encode_trial_level(obs: pd.DataFrame, var_name: str):
    """Return (array, list_of_names) for a single trial-level variable."""
    col = obs[var_name].astype(str)

    if var_name in _CATEGORICAL_VARS:
        dummies = pd.get_dummies(col, prefix=var_name).astype(float)
        if var_name in _DROP_NONE:
            dummies = dummies.drop(
                columns=[c for c in dummies.columns if "none" in c.lower()],
                errors="ignore",
            )
        return dummies.values, list(dummies.columns)

    if var_name in _BINARY_VARS:
        if var_name == "turn":
            arr = (col == "right").astype(float).values[:, None]
            return arr, ["turn_right"]
        if var_name == "correct":
            arr = obs["correct"].astype(float).values[:, None]
            return arr, ["correct"]
        if var_name == "prev_correct":
            arr = (col == "True").astype(float).values[:, None]
            return arr, ["prev_correct"]

    # Fallback: treat as binary indicator of True-ish values
    arr = col.apply(lambda x: float(x not in ("False", "0", "none", "nan", ""))).values[:, None]
    return arr, [var_name]


def _create_basis(spec: dict):
    """Instantiate a nemos basis object from a config spec."""
    btype = spec["type"]
    n = spec["n_basis_funcs"]
    if btype == "RaisedCosineLinearEval":
        return nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n, bounds=(0, 1))
    if btype == "BSplineEval":
        order = spec.get("order", 4)
        # bounds set later when we know data range
        return order, n  # return params; caller builds with data bounds
    raise ValueError(f"Unknown basis type: {btype}")


def build_design_matrix(adata_binned: ad.AnnData, config: dict):
    """Build the complex design matrix.

    Returns
    -------
    X : np.ndarray, shape (n_bins, n_features)
    feature_names : list[str]
    group_info : list[dict]  — one per feature group with name and n_features
    """
    obs = adata_binned.obs.copy()
    n_spatial_bins = config["binning"]["n_spatial_bins"]
    n_temporal_bins = config["binning"]["n_temporal_bins"]

    # --- Compute derivatives if requested ---
    base_signals = {v for v in config["features"].get("continuous", [])}
    for var in list(base_signals):
        if var.startswith("d") and var[1:] in obs.columns and var not in obs.columns:
            parent = var[1:]
            logger.info("Computing derivative %s from %s per trial", var, parent)
            obs[var] = np.nan
            for tid in obs["trial_id"].unique():
                mask = obs["trial_id"] == tid
                vals = obs.loc[mask, parent].values.astype(float)
                obs.loc[mask, var] = np.gradient(vals)
    adata_binned.obs = obs

    # --- Encode trial-level variables ---
    trial_indicators = []
    trial_indicator_names: list[str] = []
    for var in config["features"].get("trial_level", []):
        arr, names = _encode_trial_level(obs, var)
        trial_indicators.append(arr)
        trial_indicator_names.extend(names)

    if trial_indicators:
        trial_indicator_mat = np.column_stack(trial_indicators)
    else:
        trial_indicator_mat = np.empty((len(obs), 0))

    # --- Z-score continuous features ---
    continuous_names: list[str] = []
    continuous_mat_cols = []
    for var in config["features"].get("continuous", []):
        vals = obs[var].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = (vals - np.mean(vals)) / (np.std(vals) + 1e-8)
        continuous_mat_cols.append(vals)
        continuous_names.append(var)

    # --- Basis objects ---
    basis_specs = config["basis_expansions"]
    mapping = config["feature_basis_mapping"]

    # Trial-level spatial basis
    spatial_key = mapping["trial_level"]["trial"]
    spatial_spec = basis_specs[spatial_key]
    spatial_basis = _create_basis(spatial_spec)
    n_spatial_basis = spatial_spec["n_basis_funcs"]

    # Trial-level temporal basis
    temporal_key = mapping["trial_level"]["iti"]
    temporal_spec = basis_specs[temporal_key]
    temporal_basis = _create_basis(temporal_spec)
    n_temporal_basis = temporal_spec["n_basis_funcs"]

    # Continuous bspline params
    cont_key = mapping["continuous"]
    cont_spec = basis_specs[cont_key]
    bspline_order = cont_spec.get("order", 4)
    n_bspline = cont_spec["n_basis_funcs"]

    # --- Precompute basis features ---
    is_iti = obs["is_iti"].values
    not_iti = ~is_iti

    spatial_pos = obs["spatial_bin"].values.astype(float) / n_spatial_bins
    spatial_features_raw = np.asarray(spatial_basis.compute_features(spatial_pos))

    iti_temporal = obs["spatial_bin"].values.astype(float) / n_temporal_bins
    temporal_features_raw = np.asarray(temporal_basis.compute_features(
        np.clip(iti_temporal, 0, 1)
    ))

    # --- Expand trial-level with spatial/temporal bases ---
    complex_features: list[np.ndarray] = []
    complex_feature_names: list[str] = []
    group_info: list[dict] = []

    n_features_per_trial_group = n_spatial_basis + n_temporal_basis

    for i, name in enumerate(trial_indicator_names):
        indicator = trial_indicator_mat[:, i]

        spatial_expanded = spatial_features_raw * indicator[:, None]
        spatial_expanded[is_iti] = 0

        temporal_expanded = temporal_features_raw * indicator[:, None]
        temporal_expanded[not_iti] = 0

        complex_features.append(spatial_expanded)
        complex_features.append(temporal_expanded)

        for j in range(n_spatial_basis):
            complex_feature_names.append(f"{name}_spatial_{j}")
        for j in range(n_temporal_basis):
            complex_feature_names.append(f"{name}_temporal_{j}")

        group_info.append({
            "name": name,
            "n_features": n_features_per_trial_group,
        })

    # --- Expand continuous with B-spline ---
    for idx, var in enumerate(continuous_names):
        vals = continuous_mat_cols[idx]
        vmin, vmax = vals.min(), vals.max()
        bspline = nmo.basis.BSplineEval(
            n_basis_funcs=n_bspline,
            order=bspline_order,
            bounds=(vmin - 0.01, vmax + 0.01),
        )
        expanded = np.asarray(bspline.compute_features(vals))
        complex_features.append(expanded)
        for j in range(n_bspline):
            complex_feature_names.append(f"{var}_bspline_{j}")
        group_info.append({
            "name": var,
            "n_features": n_bspline,
        })

    X = np.column_stack(complex_features) if complex_features else np.empty((len(obs), 0))
    X = np.nan_to_num(X, nan=0.0).astype(np.float64)
    logger.info(
        "Design matrix: %s  (%d groups)", X.shape, len(group_info)
    )
    return X, complex_feature_names, group_info


# ---------------------------------------------------------------------------
# 4. Fit GLM
# ---------------------------------------------------------------------------
def fit_glm(
    X: np.ndarray,
    Y: np.ndarray,
    config: dict,
    group_info: list[dict],
):
    """Fit a PopulationGLM according to the model config.

    Returns (model, group_mask_or_None).
    """
    model_cfg = config["model"]
    n_features = X.shape[1]
    n_neurons = Y.shape[1]

    obs_model = model_cfg["observation_model"]
    solver_name = model_cfg["solver_name"]
    solver_kwargs = model_cfg.get("solver_kwargs", {})
    reg_strength = model_cfg.get("regularizer_strength", 0.01)

    group_mask = None
    regularizer = None

    if model_cfg.get("regularizer") == "GroupLasso":
        n_groups = len(group_info)
        group_mask = np.zeros((n_groups, n_features, n_neurons))
        idx = 0
        for g, gi in enumerate(group_info):
            nf = gi["n_features"]
            group_mask[g, idx : idx + nf, :] = 1
            idx += nf
        assert idx == n_features, f"Group mask mismatch: {idx} != {n_features}"
        regularizer = nmo.regularizer.GroupLasso(mask=group_mask)
        logger.info(
            "GroupLasso mask: %s  (%d groups)", group_mask.shape, n_groups
        )

    logger.info(
        "Fitting PopulationGLM (obs=%s, solver=%s, reg=%s) …",
        obs_model,
        solver_name,
        model_cfg.get("regularizer", "none"),
    )
    model = nmo.glm.PopulationGLM(
        observation_model=obs_model,
        regularizer=regularizer,
        regularizer_strength=reg_strength if regularizer else None,
        solver_name=solver_name,
        solver_kwargs=solver_kwargs,
    )
    model.fit(X, Y)
    logger.info("Fit complete. coef_ shape: %s", model.coef_.shape)
    return model, group_mask


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
def _save_all(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    group_info: list[dict],
    group_mask,
    config: dict,
    output_dir: str,
):
    """Unified save routine (called from main with access to X and Y)."""
    os.makedirs(output_dir, exist_ok=True)

    W = np.array(model.coef_)
    np.save(os.path.join(output_dir, "weights.npy"), W)
    logger.info("Saved weights.npy  %s", W.shape)

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(output_dir, "group_info.json"), "w") as f:
        json.dump(group_info, f, indent=2)

    if group_mask is not None:
        np.save(os.path.join(output_dir, "group_mask.npy"), group_mask)
        logger.info("Saved group_mask.npy  %s", group_mask.shape)

    # Pseudo-R²
    r2 = float(model.score(X, Y, score_type="pseudo-r2-McFadden"))
    logger.info("Pseudo-R² (McFadden): %.6f", r2)
    with open(os.path.join(output_dir, "score.json"), "w") as f:
        json.dump({"pseudo_r2_mcfadden": r2}, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Summary text
    group_names = [g["name"] for g in group_info]
    summary_lines = [
        f"GLM fit summary",
        f"  observation_model : {config['model']['observation_model']}",
        f"  regularizer       : {config['model'].get('regularizer', 'none')}",
        f"  solver            : {config['model']['solver_name']}",
        f"  target_layer      : {config.get('target_layer', 'dcnv_norm')}",
        f"  n_bins (rows)     : {X.shape[0]}",
        f"  n_features (cols) : {X.shape[1]}",
        f"  n_neurons         : {Y.shape[1]}",
        f"  n_groups          : {len(group_info)}",
        f"  groups            : {group_names}",
        f"  pseudo_r2         : {r2:.6f}",
        f"  coef_ shape       : {W.shape}",
    ]
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info("All results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit a Gamma GLM to neural calcium imaging data."
    )
    parser.add_argument("--mouse", required=True, help="Mouse ID (e.g. YRA084)")
    parser.add_argument("--session", required=True, help="Session ID (e.g. 260228)")
    parser.add_argument("--config", required=True, help="Path to run specification JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: OUTPUT_ROOT/{mouse}/{session}/)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to a pre-existing h5ad file (skips copy from DATA_ROOT)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    logger.info("Config loaded from %s", args.config)

    output_dir = args.output or os.path.join(OUTPUT_ROOT, args.mouse, args.session)

    # 1. Load
    adata = load_data(args.mouse, args.session, output_dir, data_path=args.data_path)

    # 2. Bin
    binning = config.get("binning", {})
    adata_binned = bin_data(
        adata,
        n_spatial_bins=binning.get("n_spatial_bins", 100),
        n_temporal_bins=binning.get("n_temporal_bins", 50),
    )

    # 3. Design matrix
    X, feature_names, group_info = build_design_matrix(adata_binned, config)

    # 4. Target
    target_layer = config.get("target_layer", "dcnv_norm")
    Y = np.array(adata_binned.layers[target_layer], dtype=np.float64)
    Y = Y + 1e-6  # offset for Gamma
    logger.info("Y shape: %s  range: %.6f – %.4f", Y.shape, Y.min(), Y.max())

    # 5. Fit
    model, group_mask = fit_glm(X, Y, config, group_info)

    # 6. Save
    _save_all(model, X, Y, feature_names, group_info, group_mask, config, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
