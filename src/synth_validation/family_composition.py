"""
Model-family dependence of synthetic-data calibration (WP1 extension).

This module measures how the composition and size of the calibration pool
``H_cal`` (drawn from ``ModelSelectionFramework.FAMILIES`` = linear / xgboost /
random_forest / mlp) affects the quality of synthetic-data calibration:

- ``run_family_composition_experiment``: fixes ``M_calibration`` and varies
  which families ``H_cal`` is drawn from (single-family pools, leave-one-
  family-out (LOFO) pools, and a "random_mixed" baseline drawn from all
  families), then measures transfer to a fixed, diverse evaluation pool
  ``H_eval`` (the complement of ``H_cal`` within the full architecture pool).
- ``run_mcal_sweep_experiment``: fixes the family composition (default: all
  families) and varies the calibration pool size ``M_calibration``.

Both functions reuse an already-configured ``ExperimentRunner`` (data loader,
model selector, calibrator registry, generative-model training) so that, for
a given fold, the SAME synthetic draw is shared across every regime being
compared, keeping the generator fixed as a controlled variable (per the
research plan, generator hyperparameter tuning is intentionally disabled
here).

Only calibration methods that depend on the calibration-model loss matrix
("alignment", "bpr") are supported: "density" and "kmm" calibrate purely on
covariates (X_real vs. X_synth) and are therefore insensitive to the
model-family composition of ``H_cal`` by construction.
"""

import warnings
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .metrics import EvaluationMetrics
from .runner import ExperimentRunner
from .utils import CV_RANDOM_STATE

# Only these calibration methods expose a per-sample loss matrix over H_cal,
# so only these are meaningfully affected by the family composition of H_cal.
_MODEL_LOSS_BASED_METHODS = ("alignment", "bpr")


def _regime_seed(base_seed: int, fold_idx: int, regime_name: str) -> int:
    """Deterministic per-(fold, regime) seed, stable across interpreter runs."""
    offset = zlib.crc32(regime_name.encode("utf-8")) % 10_000
    return base_seed + fold_idx + offset


def _validate_and_build_pool(
    full_architectures: List[Any],
    families: Sequence[str],
    family_universe: Sequence[str],
) -> Tuple[List[str], List[Any]]:
    """Validate a family list and select the matching architectures.

    Args:
        full_architectures: Canonical list of ``ModelConfig`` (called once
            per experiment so identity/name comparisons stay consistent).
        families: Requested model families for this regime.
        family_universe: The set of valid family names (``FAMILIES``).

    Returns:
        (deduplicated family list, list of architectures belonging to them)
    """
    unique_families = list(dict.fromkeys(families))
    unknown = set(unique_families) - set(family_universe)
    if unknown:
        raise ValueError(
            f"Unknown model family(ies) {sorted(unknown)}. Expected one of {list(family_universe)}."
        )
    pool = [a for a in full_architectures if a.family in unique_families]
    return unique_families, pool


def default_family_compositions(family_universe: Sequence[str]) -> Dict[str, List[str]]:
    """Build the standard set of composition regimes for a family universe.

    Returns a dict with one "random_mixed" regime (all families), one
    single-family regime per family, and one leave-one-family-out (LOFO)
    regime per family (all families except the named one).
    """
    families = list(family_universe)
    compositions: Dict[str, List[str]] = {"random_mixed": list(families)}
    for family in families:
        compositions[f"{family}_only"] = [family]
    for family in families:
        compositions[f"lofo_{family}"] = [f for f in families if f != family]
    return compositions


def _run_regimes(
    runner: ExperimentRunner,
    regimes: Dict[str, Dict[str, Any]],
    n_folds: int,
    synth_size_multiplier: float,
    calib_test_ratio: float,
    calibration_method: str,
    random_state: int,
) -> pd.DataFrame:
    """Shared k-fold engine behind both public experiment functions.

    `regimes` maps a regime name -> {"families": [...], "M_calibration": int}.
    For each fold, ONE synthetic draw is generated and reused across every
    regime; per regime, ``M_calibration`` models are sampled from the
    regime's family-restricted pool to form H_cal, and the complement of
    H_cal within the FULL architecture pool forms H_eval (so calibration
    transfer to the complete, diverse model space can be measured even when
    H_cal is restricted to one or a few families).
    """
    if calibration_method not in _MODEL_LOSS_BASED_METHODS:
        raise ValueError(
            f"calibration_method must be one of {_MODEL_LOSS_BASED_METHODS} "
            "(density/kmm calibrate on covariates only and do not depend on "
            "the model-family composition of H_cal)."
        )

    family_universe = runner.model_selector.FAMILIES
    full_architectures = runner.model_selector.get_model_architectures()
    n_total_models = len(full_architectures)

    valid_regimes: Dict[str, Dict[str, Any]] = {}
    for name, spec in regimes.items():
        families, pool = _validate_and_build_pool(full_architectures, spec["families"], family_universe)
        m_cal = spec["M_calibration"]

        if m_cal <= 0:
            warnings.warn(f"[skip] Regime '{name}': M_calibration must be positive, got {m_cal}.")
            continue
        if m_cal > len(pool):
            warnings.warn(
                f"[skip] Regime '{name}': M_calibration={m_cal} exceeds pool size "
                f"{len(pool)} for families {families}."
            )
            continue
        if m_cal >= n_total_models:
            warnings.warn(
                f"[skip] Regime '{name}': M_calibration={m_cal} must be < total "
                f"architectures ({n_total_models})."
            )
            continue

        valid_regimes[name] = {"families": families, "M_calibration": m_cal, "pool": pool}

    if not valid_regimes:
        warnings.warn("No valid regimes to run; returning an empty DataFrame.")
        return pd.DataFrame()

    df = runner.data_loader.load_uci_dataset(runner.dataset_name)
    target_col = None
    for col in ["income", "target", "class"]:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    X_full = df.drop(columns=[target_col]).copy()
    y_full = df[target_col].copy()

    if runner.task_type == "classification":
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_iterator = list(kfold.split(X_full, y_full))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_iterator = list(kfold.split(X_full))

    calibrator_registry = runner._init_calibration_registry([calibration_method])
    calibrator = calibrator_registry[calibration_method]

    rows: List[Dict[str, Any]] = []

    for fold_idx, (train_index, test_index) in enumerate(fold_iterator):
        if runner.verbose:
            print(f"\n{'=' * 70}\nFOLD {fold_idx + 1}/{n_folds}\n{'=' * 70}")

        X_train_fold = X_full.iloc[train_index].reset_index(drop=True)
        y_train_fold = y_full.iloc[train_index].reset_index(drop=True)
        X_test_fold = X_full.iloc[test_index].reset_index(drop=True)
        y_test_fold = y_full.iloc[test_index].reset_index(drop=True)

        X_train_fold_proc, X_test_proc, y_train_fold_proc, y_test_proc, _ = runner.data_loader.prepare_data(
            X_train_fold, X_test_fold, y_train_fold, y_test_fold, task_type=runner.task_type
        )

        if runner.task_type == "classification":
            X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                X_train_fold_proc, y_train_fold_proc,
                test_size=calib_test_ratio,
                random_state=random_state + fold_idx,
                stratify=y_train_fold_proc,
            )
        else:
            X_train_calib, X_test_calib, y_train_calib, y_test_calib = train_test_split(
                X_train_fold_proc, y_train_fold_proc,
                test_size=calib_test_ratio,
                random_state=random_state + fold_idx,
            )

        # One synthetic draw per fold, shared by every regime (fixed generator).
        fold_generator = runner._train_generative_model_for_fold(
            X_train_fold_proc, y_train_fold_proc,
            use_cached_hyperparams=(runner._best_hyperparams is not None),
        )
        n_synth = int(len(X_test_proc) * synth_size_multiplier)
        X_synth, y_synth = fold_generator.generate(n_samples=n_synth)
        y_synth = y_synth.astype(int) if runner.task_type == "classification" else y_synth.astype(float)

        # Cache trained models per architecture/role within this fold, since the
        # same architecture can appear in H_cal for one regime and H_eval for
        # another; training data differs by role (calib subset vs. full fold train).
        calib_model_cache: Dict[int, Any] = {}
        eval_model_cache: Dict[int, Any] = {}

        def _get_trained(config: Any, X: pd.DataFrame, y: pd.Series, cache: Dict[int, Any]) -> Any:
            key = id(config)
            if key not in cache:
                cache[key] = runner.model_selector.train_model(config, X, y)
            return cache[key]

        for regime_name, regime in valid_regimes.items():
            pool = regime["pool"]
            m_cal = regime["M_calibration"]

            rng = np.random.RandomState(_regime_seed(random_state, fold_idx, regime_name))
            shuffled_pool = pool.copy()
            rng.shuffle(shuffled_pool)
            h_cal_architectures = shuffled_pool[:m_cal]
            h_cal_names = {a.name for a in h_cal_architectures}
            h_eval_architectures = [a for a in full_architectures if a.name not in h_cal_names]

            h_cal_models = [
                _get_trained(cfg, X_train_calib, y_train_calib, calib_model_cache)
                for cfg in h_cal_architectures
            ]

            runner._fit_calibrator_for_method(
                method=calibration_method,
                calibrator=calibrator,
                calibration_models=h_cal_models,
                X_synth=X_synth,
                y_synth=y_synth,
                X_real_val=X_test_calib,
                y_real_val=y_test_calib,
            )

            h_eval_models = [
                _get_trained(cfg, X_train_fold_proc, y_train_fold_proc, eval_model_cache)
                for cfg in h_eval_architectures
            ]

            real_losses, synth_losses, calib_losses = [], [], []
            for model in h_eval_models:
                real_losses.append(runner.model_selector.evaluate_model(model, X_test_proc, y_test_proc)["loss"])
                synth_losses.append(runner.model_selector.evaluate_model(model, X_synth, y_synth)["loss"])
                calib_losses.append(calibrator.evaluate_calibrated_loss(model, X_synth, y_synth))

            real_losses = np.asarray(real_losses, dtype=float)
            synth_losses = np.asarray(synth_losses, dtype=float)
            calib_losses = np.asarray(calib_losses, dtype=float)

            uncalib_rho, _ = runner.ci_estimator.compute_spearman(real_losses, synth_losses)
            calib_rho, _ = runner.ci_estimator.compute_spearman(real_losses, calib_losses)

            weights = np.asarray(calibrator.compute_weights_for_samples(y_synth), dtype=float)

            reg_w = EvaluationMetrics.reg_at_1(real_losses, calib_losses)
            reg_u = EvaluationMetrics.reg_at_1(real_losses, synth_losses)
            excess = EvaluationMetrics.excess_harm_gain(reg_w, reg_u)

            loss_matrix = getattr(calibrator, "loss_matrix", None)
            hcal_real_losses = getattr(calibrator, "real_losses", None)
            if loss_matrix is not None and hcal_real_losses is not None:
                kappa = EvaluationMetrics.condition_number(loss_matrix)
                rho_cal = EvaluationMetrics.calibration_spearman_correlation(
                    loss_matrix.mean(axis=0), hcal_real_losses
                )["correlation"]
            else:
                kappa, rho_cal = np.nan, np.nan

            rows.append({
                "regime": regime_name,
                "families": ",".join(regime["families"]),
                "fold": fold_idx + 1,
                "M_calibration": m_cal,
                "n_h_cal_pool": len(pool),
                "n_h_eval": len(h_eval_architectures),
                "calibration_method": calibration_method,
                "uncalibrated_spearman": uncalib_rho,
                "calibrated_spearman": calib_rho,
                "Reg@1_w": reg_w,
                "Reg@1_u": reg_u,
                "NormReg@1": EvaluationMetrics.norm_reg_at_1(real_losses, calib_losses, synth_losses),
                "ExcessHarm": excess["excess_harm"],
                "Gain": excess["gain"],
                "ESS": EvaluationMetrics.effective_sample_size(weights),
                "w_max": EvaluationMetrics.weight_max(weights),
                "weight_entropy": EvaluationMetrics.weight_entropy(weights),
                "condition_number": kappa,
                "rho_cal": rho_cal,
            })

            if runner.verbose:
                print(
                    f"   [{regime_name}] families={regime['families']} M_cal={m_cal} "
                    f"calibrated_rho={calib_rho:.3f} (uncalibrated={uncalib_rho:.3f})"
                )

    return pd.DataFrame(rows)


def run_family_composition_experiment(
    runner: ExperimentRunner,
    compositions: Optional[Dict[str, Sequence[str]]] = None,
    M_calibration: int = 10,
    n_folds: int = 5,
    synth_size_multiplier: float = 1.0,
    calib_test_ratio: float = 0.2,
    calibration_method: str = "alignment",
    random_state: int = CV_RANDOM_STATE,
) -> pd.DataFrame:
    """Measure how the family composition of H_cal affects calibration quality.

    For each named composition, ``M_calibration`` models are sampled from the
    architectures belonging to the composition's families to form H_cal; the
    complement within the full architecture pool forms H_eval. All
    compositions share the same per-fold synthetic draw for a fair,
    apples-to-apples comparison.

    Args:
        runner: A configured ``ExperimentRunner`` (dataset/synth_method/task_type
            already set; a pretrained generator may be loaded via
            ``runner.load_gan_model(...)`` beforehand to avoid retraining it).
        compositions: Mapping of regime name -> list of family names (subset of
            ``ModelSelectionFramework.FAMILIES``). Defaults to
            ``default_family_compositions(runner.model_selector.FAMILIES)``,
            i.e. one "random_mixed", one single-family, and one leave-one-
            family-out (LOFO) regime per family.
        M_calibration: Calibration pool size used for every composition.
            Compositions whose family-restricted pool is smaller than this
            are skipped with a warning (e.g. a single-family pool of 15
            models cannot support ``M_calibration=20``).
        n_folds: Number of cross-validation folds.
        synth_size_multiplier: Multiplier for synthetic sample size relative
            to the fold's held-out test size.
        calib_test_ratio: Fraction of fold-train data held out for fitting
            the calibrator against real losses.
        calibration_method: One of "alignment" or "bpr" (methods that use a
            per-model loss matrix over H_cal; "density"/"kmm" are covariate-
            only and thus insensitive to H_cal's family composition).
        random_state: Base random seed (folds, calib splits, and per-regime
            model sampling all derive from this).

    Returns:
        A tidy DataFrame with one row per (regime, fold), including
        calibrated/uncalibrated Spearman correlation, NormReg@1, ExcessHarm/
        Gain, and the WP3 deployment diagnostics (ESS, w_max, weight_entropy,
        condition_number, rho_cal).
    """
    if compositions is None:
        compositions = default_family_compositions(runner.model_selector.FAMILIES)

    regimes = {
        name: {"families": list(families), "M_calibration": M_calibration}
        for name, families in compositions.items()
    }

    return _run_regimes(
        runner=runner,
        regimes=regimes,
        n_folds=n_folds,
        synth_size_multiplier=synth_size_multiplier,
        calib_test_ratio=calib_test_ratio,
        calibration_method=calibration_method,
        random_state=random_state,
    )


def run_mcal_sweep_experiment(
    runner: ExperimentRunner,
    M_calibration_values: Sequence[int],
    families: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    synth_size_multiplier: float = 1.0,
    calib_test_ratio: float = 0.2,
    calibration_method: str = "alignment",
    random_state: int = CV_RANDOM_STATE,
) -> pd.DataFrame:
    """Measure how the SIZE of H_cal affects calibration quality.

    Fixes the family composition of H_cal (default: all families, i.e. a
    "random_mixed" pool) and varies ``M_calibration`` over
    ``M_calibration_values``. Every value shares the same per-fold synthetic
    draw for a fair comparison.

    Args:
        runner: A configured ``ExperimentRunner`` (see
            ``run_family_composition_experiment``).
        M_calibration_values: Calibration pool sizes to sweep over. Values
            that exceed the (family-restricted) pool size are skipped with a
            warning.
        families: Families H_cal is drawn from at every sweep point. Defaults
            to all of ``runner.model_selector.FAMILIES`` (random_mixed).
        n_folds: Number of cross-validation folds.
        synth_size_multiplier: Multiplier for synthetic sample size relative
            to the fold's held-out test size.
        calib_test_ratio: Fraction of fold-train data held out for fitting
            the calibrator against real losses.
        calibration_method: One of "alignment" or "bpr".
        random_state: Base random seed.

    Returns:
        A tidy DataFrame with one row per (M_calibration, fold); the
        "regime" column is named ``f"M{m}"`` and "M_calibration" holds the
        integer sweep value for convenient plotting.
    """
    if families is None:
        families = list(runner.model_selector.FAMILIES)

    regimes = {
        f"M{m}": {"families": list(families), "M_calibration": m}
        for m in M_calibration_values
    }

    return _run_regimes(
        runner=runner,
        regimes=regimes,
        n_folds=n_folds,
        synth_size_multiplier=synth_size_multiplier,
        calib_test_ratio=calib_test_ratio,
        calibration_method=calibration_method,
        random_state=random_state,
    )
