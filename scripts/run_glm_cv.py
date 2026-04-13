#!/usr/bin/env python
"""
run_glm_cv.py — Cross-validation and regularization selection for Poisson GLM.

Implements the following protocol (adapted from published methodology):
  1. Split trials 80% train / 20% test.
  2. On training data, perform 5-fold CV (split on trials) to select the
     optimal group-lasso lambda for each neuron independently.
  3. For each fold × lambda, fit a PopulationGLM and record per-neuron
     mean deviance on the held-out fold.
  4. Select optimal lambda per neuron as the one minimizing mean CV deviance.
  5. Refit on all training data with each neuron's optimal lambda.
  6. Evaluate on test data as fraction of Poisson deviance explained.

Usage:
    python run_glm_cv.py --mouse YRA084 --session 260228 --config config.json
"""

import argparse
import json
import logging
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import nemos as nmo
import numpy as np

from preprocess import bin_adata_vectorized
from run_glm import load_data, build_design_matrix, DATA_ROOT, OUTPUT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial splitting
# ---------------------------------------------------------------------------
def split_trials(adata_binned, test_frac=0.2, n_folds=5, seed=42):
    """Split trials into train/test and create CV folds on the training set.

    All splits are at the trial level to prevent temporal leakage.

    Returns
    -------
    dict with keys:
        train_trials, test_trials : arrays of trial IDs
        train_mask, test_mask : boolean arrays over all rows
        fold_indices : list of (fold_train_mask, fold_val_mask) tuples
                       (boolean arrays over the *full* dataset, but only
                       covering training trials)
    """
    trial_ids = np.array(sorted(adata_binned.obs["trial_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(trial_ids)

    n_test = max(1, int(len(trial_ids) * test_frac))
    test_trials = trial_ids[:n_test]
    train_trials = trial_ids[n_test:]

    obs_trial = adata_binned.obs["trial_id"].values
    train_mask = np.isin(obs_trial, train_trials)
    test_mask = np.isin(obs_trial, test_trials)

    # K-fold on the training trials — masks are relative to training rows only
    train_row_trials = obs_trial[train_mask]
    fold_size = len(train_trials) // n_folds
    fold_indices = []
    for k in range(n_folds):
        if k < n_folds - 1:
            val_trials = train_trials[k * fold_size : (k + 1) * fold_size]
        else:
            val_trials = train_trials[k * fold_size :]

        fold_val_mask = np.isin(train_row_trials, val_trials)
        fold_train_mask = ~fold_val_mask
        fold_indices.append((fold_train_mask, fold_val_mask))

    logger.info(
        "Trial split: %d train, %d test, %d folds",
        len(train_trials), len(test_trials), n_folds,
    )
    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "fold_indices": fold_indices,
    }


# ---------------------------------------------------------------------------
# GroupLasso mask construction
# ---------------------------------------------------------------------------
def build_group_mask(group_info, n_features, n_neurons):
    """Build a GroupLasso mask of shape (n_groups, n_features, n_neurons)."""
    n_groups = len(group_info)
    group_mask = np.zeros((n_groups, n_features, n_neurons))
    idx = 0
    for g, gi in enumerate(group_info):
        nf = gi["n_features"]
        group_mask[g, idx : idx + nf, :] = 1
        idx += nf
    assert idx == n_features, f"Group mask mismatch: {idx} != {n_features}"
    return group_mask


# ---------------------------------------------------------------------------
# Per-neuron deviance
# ---------------------------------------------------------------------------
def compute_per_neuron_deviance(model, X, Y):
    """Compute mean Poisson deviance per neuron on held-out data.

    Returns array of shape (n_neurons,). Lower is better.
    """
    predicted_rate = jnp.array(model.predict(X))
    Y_jnp = jnp.array(Y)
    # Poisson deviance: 2 * [y*log(y/mu) - (y - mu)]
    ratio = jnp.clip(Y_jnp / predicted_rate, jnp.finfo(float).eps, jnp.inf)
    dev = 2.0 * (Y_jnp * jnp.log(ratio) - (Y_jnp - predicted_rate))
    return np.array(jnp.mean(dev, axis=0))


# ---------------------------------------------------------------------------
# CV lambda selection
# ---------------------------------------------------------------------------
def cv_select_lambda(X, Y, group_info, fold_indices, lambda_grid, solver_kwargs):
    """Run cross-validation to find optimal lambda per neuron.

    For each lambda × fold, fits a PopulationGLM on the fold's training rows
    and evaluates per-neuron Poisson deviance on the fold's validation rows.

    Returns
    -------
    cv_deviance : ndarray, shape (n_lambdas, n_folds, n_neurons)
    mean_cv_deviance : ndarray, shape (n_lambdas, n_neurons)
    optimal_lambda_idx : ndarray, shape (n_neurons,)
    optimal_lambdas : ndarray, shape (n_neurons,)
    """
    n_lambdas = len(lambda_grid)
    n_folds = len(fold_indices)
    n_neurons = Y.shape[1]
    n_features = X.shape[1]

    cv_deviance = np.full((n_lambdas, n_folds, n_neurons), np.nan)

    for li, lam in enumerate(lambda_grid):
        for fi, (fold_train_mask, fold_val_mask) in enumerate(fold_indices):
            X_ft = X[fold_train_mask]
            Y_ft = Y[fold_train_mask]
            X_fv = X[fold_val_mask]
            Y_fv = Y[fold_val_mask]

            group_mask = build_group_mask(group_info, n_features, n_neurons)
            gl_reg = nmo.regularizer.GroupLasso(mask=group_mask)

            model = nmo.glm.PopulationGLM(
                observation_model="Poisson",
                regularizer=gl_reg,
                regularizer_strength=float(lam),
                solver_name="ProximalGradient",
                solver_kwargs=solver_kwargs,
            )
            model.fit(X_ft, Y_ft)
            cv_deviance[li, fi] = compute_per_neuron_deviance(model, X_fv, Y_fv)

            logger.info(
                "  lambda=%.2e  fold %d/%d  mean_dev=%.4f",
                lam, fi + 1, n_folds, np.nanmean(cv_deviance[li, fi]),
            )

    mean_cv_deviance = np.nanmean(cv_deviance, axis=1)  # (n_lambdas, n_neurons)
    optimal_lambda_idx = np.argmin(mean_cv_deviance, axis=0)  # (n_neurons,)
    optimal_lambdas = lambda_grid[optimal_lambda_idx]

    logger.info(
        "Lambda selection complete. Unique optimal lambdas: %s",
        np.unique(optimal_lambdas),
    )
    return cv_deviance, mean_cv_deviance, optimal_lambda_idx, optimal_lambdas


# ---------------------------------------------------------------------------
# Final refit with per-neuron optimal lambda
# ---------------------------------------------------------------------------
def fit_final_models(X_train, Y_train, group_info, optimal_lambdas, lambda_grid,
                     solver_kwargs):
    """Refit on all training data, grouping neurons by their optimal lambda.

    Returns final_coef (n_features, n_neurons) and final_intercept (n_neurons,).
    """
    n_features = X_train.shape[1]
    n_neurons = Y_train.shape[1]
    final_coef = np.zeros((n_features, n_neurons))
    final_intercept = np.zeros(n_neurons)

    unique_lambdas = np.unique(optimal_lambdas)
    for lam in unique_lambdas:
        neuron_indices = np.where(optimal_lambdas == lam)[0]
        Y_sub = Y_train[:, neuron_indices]
        n_sub = len(neuron_indices)

        group_mask = build_group_mask(group_info, n_features, n_sub)
        gl_reg = nmo.regularizer.GroupLasso(mask=group_mask)

        model = nmo.glm.PopulationGLM(
            observation_model="Poisson",
            regularizer=gl_reg,
            regularizer_strength=float(lam),
            solver_name="ProximalGradient",
            solver_kwargs=solver_kwargs,
        )
        model.fit(X_train, Y_sub)

        final_coef[:, neuron_indices] = np.array(model.coef_)
        final_intercept[neuron_indices] = np.array(model.intercept_)

        logger.info(
            "Final refit lambda=%.2e: %d neurons, mean|coef|=%.4f",
            lam, n_sub, np.mean(np.abs(model.coef_)),
        )

    return final_coef, final_intercept


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------
def evaluate_test(final_coef, final_intercept, X_test, Y_test):
    """Compute fraction of Poisson deviance explained on held-out test data.

    This is Cohen's pseudo-R² = 1 - D_model / D_null, per neuron.
    """
    X_j = jnp.array(X_test)
    Y_j = jnp.array(Y_test)
    coef_j = jnp.array(final_coef)
    intercept_j = jnp.array(final_intercept)

    # Predicted rate using Poisson default link (exp)
    predicted_rate = jnp.exp(X_j @ coef_j + intercept_j)

    # Model deviance per neuron
    ratio_model = jnp.clip(Y_j / predicted_rate, jnp.finfo(float).eps, jnp.inf)
    dev_model = 2.0 * (Y_j * jnp.log(ratio_model) - (Y_j - predicted_rate))
    mean_dev_model = jnp.mean(dev_model, axis=0)

    # Null model: constant rate = mean(Y_test) per neuron
    null_rate = jnp.mean(Y_j, axis=0, keepdims=True) * jnp.ones_like(Y_j)
    ratio_null = jnp.clip(Y_j / null_rate, jnp.finfo(float).eps, jnp.inf)
    dev_null = 2.0 * (Y_j * jnp.log(ratio_null) - (Y_j - null_rate))
    mean_dev_null = jnp.mean(dev_null, axis=0)

    deviance_explained = 1.0 - mean_dev_model / mean_dev_null

    return {
        "deviance_explained": np.array(deviance_explained),
        "mean_deviance_model": np.array(mean_dev_model),
        "mean_deviance_null": np.array(mean_dev_null),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_cv_results(
    output_dir,
    final_coef,
    final_intercept,
    optimal_lambdas,
    cv_deviance,
    mean_cv_deviance,
    lambda_grid,
    test_results,
    trial_splits,
    feature_names,
    group_info,
    config,
):
    """Save all cross-validation results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "weights.npy"), final_coef)
    np.save(os.path.join(output_dir, "intercepts.npy"), final_intercept)
    np.save(os.path.join(output_dir, "optimal_lambdas.npy"), optimal_lambdas)
    np.save(os.path.join(output_dir, "cv_deviance_per_fold.npy"), cv_deviance)
    np.save(os.path.join(output_dir, "cv_mean_deviance.npy"), mean_cv_deviance)
    np.save(os.path.join(output_dir, "lambda_grid.npy"), lambda_grid)
    np.save(
        os.path.join(output_dir, "test_deviance_explained.npy"),
        test_results["deviance_explained"],
    )
    np.save(
        os.path.join(output_dir, "test_mean_deviance.npy"),
        test_results["mean_deviance_model"],
    )

    # JSON metadata
    splits_serializable = {
        "train_trials": trial_splits["train_trials"].tolist(),
        "test_trials": trial_splits["test_trials"].tolist(),
    }
    with open(os.path.join(output_dir, "trial_splits.json"), "w") as f:
        json.dump(splits_serializable, f, indent=2)

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(output_dir, "group_info.json"), "w") as f:
        json.dump(group_info, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    dev_expl = test_results["deviance_explained"]
    summary_lines = [
        "GLM Cross-Validation Summary",
        f"  observation_model      : Poisson",
        f"  regularizer            : GroupLasso",
        f"  n_lambdas              : {len(lambda_grid)}",
        f"  lambda_range           : [{lambda_grid[0]:.1e}, {lambda_grid[-1]:.1e}]",
        f"  n_folds                : {cv_deviance.shape[1]}",
        f"  n_train_trials         : {len(trial_splits['train_trials'])}",
        f"  n_test_trials          : {len(trial_splits['test_trials'])}",
        f"  n_features             : {final_coef.shape[0]}",
        f"  n_neurons              : {final_coef.shape[1]}",
        f"  n_groups               : {len(group_info)}",
        f"  unique_optimal_lambdas : {np.unique(optimal_lambdas).tolist()}",
        f"  test_dev_explained     : mean={np.mean(dev_expl):.4f}, "
        f"median={np.median(dev_expl):.4f}, "
        f"range=[{np.min(dev_expl):.4f}, {np.max(dev_expl):.4f}]",
    ]
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info("All CV results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation for GLM regularization selection."
    )
    parser.add_argument("--mouse", required=True, help="Mouse ID (e.g. YRA084)")
    parser.add_argument("--session", required=True, help="Session ID (e.g. 260228)")
    parser.add_argument("--config", required=True, help="Path to run specification JSON")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--data-path", default=None, help="Path to h5ad file")
    parser.add_argument("--n-lambdas", type=int, default=11,
                        help="Number of lambda values to search (default: 11)")
    parser.add_argument("--lambda-min", type=float, default=1e-5,
                        help="Minimum lambda (default: 1e-5)")
    parser.add_argument("--lambda-max", type=float, default=1e-1,
                        help="Maximum lambda (default: 1e-1)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of trials for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trial splitting (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = json.load(f)
    logger.info("Config loaded from %s", args.config)

    output_dir = args.output or os.path.join(
        OUTPUT_ROOT, args.mouse, args.session, "cv_results"
    )

    # 1. Load data
    adata = load_data(args.mouse, args.session, output_dir, data_path=args.data_path)

    # 2. Bin data
    binning = config.get("binning", {})
    adata_binned = bin_adata_vectorized(
        adata,
        n_spatial_bins=binning.get("n_spatial_bins", 100),
        n_temporal_bins=binning.get("n_temporal_bins", 50),
    )

    # 3. Build design matrix
    X, feature_names, group_info = build_design_matrix(adata_binned, config)

    # 4. Target variable
    target_layer = config.get("target_layer", "dcnv_norm")
    Y = np.array(adata_binned.layers[target_layer], dtype=np.float64)
    Y = Y + 1e-6  # small offset for numerical stability
    logger.info("Y shape: %s  range: %.6f - %.4f", Y.shape, Y.min(), Y.max())

    # 5. Split trials
    splits = split_trials(
        adata_binned,
        test_frac=args.test_frac,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    X_train = X[splits["train_mask"]]
    Y_train = Y[splits["train_mask"]]
    X_test = X[splits["test_mask"]]
    Y_test = Y[splits["test_mask"]]
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # 6. Lambda grid
    lambda_grid = np.logspace(
        np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambdas
    )
    logger.info("Lambda grid (%d values): %s", len(lambda_grid), lambda_grid)

    solver_kwargs = config.get("model", {}).get("solver_kwargs", {"maxiter": 500})

    # 7. Cross-validation
    logger.info("Starting %d-fold CV with %d lambdas ...", args.n_folds, args.n_lambdas)
    cv_deviance, mean_cv_deviance, optimal_lambda_idx, optimal_lambdas = (
        cv_select_lambda(
            X_train, Y_train, group_info, splits["fold_indices"],
            lambda_grid, solver_kwargs,
        )
    )

    # 8. Final refit with optimal lambdas
    logger.info("Refitting with per-neuron optimal lambdas ...")
    final_coef, final_intercept = fit_final_models(
        X_train, Y_train, group_info, optimal_lambdas, lambda_grid, solver_kwargs,
    )

    # 9. Evaluate on test set
    logger.info("Evaluating on test set ...")
    test_results = evaluate_test(final_coef, final_intercept, X_test, Y_test)
    dev_expl = test_results["deviance_explained"]
    logger.info(
        "Test deviance explained: mean=%.4f, median=%.4f",
        np.mean(dev_expl), np.median(dev_expl),
    )

    # 10. Save
    save_cv_results(
        output_dir, final_coef, final_intercept, optimal_lambdas,
        cv_deviance, mean_cv_deviance, lambda_grid, test_results,
        splits, feature_names, group_info, config,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
