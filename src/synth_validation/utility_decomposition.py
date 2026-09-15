import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.base import clone


def feature_fidelity_proxy(
    X_real,
    X_synth,
    xgb_kwargs=None,
    random_state=None,
    n_splits=5,
):
    X_real_arr = np.asarray(X_real)
    X_synth_arr = np.asarray(X_synth)
    n_real, n_synth = len(X_real_arr), len(X_synth_arr)

    X = np.vstack([X_real_arr, X_synth_arr])
    y = np.concatenate([np.ones(n_real), np.zeros(n_synth)])

    kwargs = dict(xgb_kwargs or {})
    if random_state is not None:
        kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("n_estimators", 200)
    kwargs.setdefault("max_depth", 3)
    kwargs.setdefault("learning_rate", 0.05)
    kwargs.setdefault("tree_method", "hist")
    kwargs.setdefault("n_jobs", -1)
    kwargs.setdefault("verbosity", 0)

    n_splits = max(2, min(n_splits, n_real, n_synth))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    p_real_oof = np.full(n_synth, np.nan, dtype=np.float64)

    for train_idx, test_idx in splitter.split(X, y):
        clf = XGBClassifier(**kwargs)
        clf.fit(X[train_idx], y[train_idx])

        classes = clf.classes_
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        if 1 not in class_to_idx:
            continue

        synth_test_mask = test_idx >= n_real
        synth_test_idx = test_idx[synth_test_mask]
        if len(synth_test_idx) == 0:
            continue

        proba = clf.predict_proba(X[synth_test_idx])
        p_real_oof[synth_test_idx - n_real] = proba[:, class_to_idx[1]]

    if np.isnan(p_real_oof).any():
        n_missing = int(np.isnan(p_real_oof).sum())
        clf = XGBClassifier(**kwargs)
        clf.fit(X, y)
        classes = clf.classes_
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        if 1 in class_to_idx:
            fallback_idx = np.where(np.isnan(p_real_oof))[0]
            proba = clf.predict_proba(X_synth_arr[fallback_idx])
            p_real_oof[fallback_idx] = proba[:, class_to_idx[1]]
        else:
            p_real_oof[np.isnan(p_real_oof)] = 0.5

    p_real_oof = np.clip(p_real_oof, 1e-7, 1 - 1e-7)
    return (p_real_oof / (1 - p_real_oof)) * (n_synth / n_real)

def relationship_estimation_proxy(
    X_synth,
    y_synth,
    X_real,
    y_real,
    task_type="classification",
    n_splits=5,
    base_estimator=None,
    random_state=None,
):
    
    X_real_arr = np.asarray(X_real)
    y_real_arr = np.asarray(y_real)
    X_synth_arr = np.asarray(X_synth)
    y_synth_arr = np.asarray(y_synth)
    n_synth = len(X_synth_arr)

    if base_estimator is None:
        est_cls = XGBClassifier if task_type == "classification" else XGBRegressor
        base_estimator = est_cls(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )

    n_splits = max(2, min(n_splits, len(X_real_arr)))

    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X_real_arr, y_real_arr)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X_real_arr)

    fold_preds = []
    for train_idx, _ in split_iter:
        model = clone(base_estimator)
        model.fit(X_real_arr[train_idx], y_real_arr[train_idx])

        if task_type == "classification":
            proba = model.predict_proba(X_synth_arr)
            classes = model.classes_
            class_to_idx = {label: idx for idx, label in enumerate(classes)}
            p_true = np.full(n_synth, 1e-7, dtype=np.float64)
            for i in range(n_synth):
                idx = class_to_idx.get(y_synth_arr[i])
                if idx is not None:
                    p_true[i] = proba[i, idx]
            p_true = np.clip(p_true, 1e-7, 1 - 1e-7)
            fold_preds.append(1.0 - p_true)
        else:
            preds = model.predict(X_synth_arr)
            fold_preds.append(np.abs(preds - y_synth_arr))

    return np.mean(np.stack(fold_preds, axis=0), axis=0).astype(np.float64)


def model_specification_proxy(
    eval_models,
    X_synth,
    task_type: str = "classification",
    oracle_real_preds=None,
    oracle_synth_preds=None,
    oracle_estimator=None,
    X_real=None,
    y_real=None,
    n_splits=5,
    random_state=None,
) -> np.ndarray:
 
    X_synth_arr = np.asarray(X_synth)
    n_synth = len(X_synth_arr)

    use_cross_fitted_oracle = oracle_estimator is not None
    if use_cross_fitted_oracle:
        if X_real is None or y_real is None:
            raise ValueError("oracle_estimator requires X_real and y_real.")
        X_real_arr = np.asarray(X_real)
        y_real_arr = np.asarray(y_real)
        n_real = len(X_real_arr)
        n_splits_eff = max(2, min(n_splits, n_real))

        if task_type == "classification":
            splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
            split_iter = splitter.split(X_real_arr, y_real_arr)
        else:
            splitter = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
            split_iter = splitter.split(X_real_arr)

        oracle_real_preds = np.empty(n_real, dtype=np.float64 if task_type != "classification" else int)
        synth_pred_accum = []

        for train_idx, test_idx in split_iter:
            oracle = clone(oracle_estimator)
            oracle.fit(X_real_arr[train_idx], y_real_arr[train_idx])

            if task_type == "classification":
                oracle_real_preds[test_idx] = oracle.predict(X_real_arr[test_idx])
            else:
                oracle_real_preds[test_idx] = oracle.predict(X_real_arr[test_idx])

            synth_pred_accum.append(oracle.predict(X_synth_arr))

        if task_type == "classification":
            stacked = np.stack(synth_pred_accum, axis=1)
            oracle_synth_preds = (np.mean(stacked, axis=1) >= 0.5).astype(int)
        else:
            oracle_synth_preds = np.mean(np.stack(synth_pred_accum, axis=1), axis=1)

    if task_type == "classification":
        probas_real, probas_synth = [], []
        for m in eval_models:
            if hasattr(m, "predict_proba"):
                probas_real.append(m.predict_proba(X_real_arr if use_cross_fitted_oracle else X_synth_arr)[:, -1])
                probas_synth.append(m.predict_proba(X_synth_arr)[:, -1])
            else:
                probas_real.append((m.predict(X_real_arr if use_cross_fitted_oracle else X_synth_arr) > 0.5).astype(float))
                probas_synth.append((m.predict(X_synth_arr) > 0.5).astype(float))

        fF_real_consensus = (np.stack(probas_real, axis=1).mean(axis=1) >= 0.5).astype(int)
        fF_synth_consensus = (np.stack(probas_synth, axis=1).mean(axis=1) >= 0.5).astype(int)

        delta_real = (fF_real_consensus != oracle_real_preds).astype(np.float64)
        delta_synth = (fF_synth_consensus != oracle_synth_preds).astype(np.float64)
    else:
        X_for_real = X_real_arr if use_cross_fitted_oracle else X_synth_arr
        preds_real = np.stack([m.predict(X_for_real) for m in eval_models], axis=1)
        fF_real_consensus = preds_real.mean(axis=1)

        preds_synth = np.stack([m.predict(X_synth_arr) for m in eval_models], axis=1)
        fF_synth_consensus = preds_synth.mean(axis=1)

        delta_real = np.abs(fF_real_consensus - oracle_real_preds)
        delta_synth = np.abs(fF_synth_consensus - oracle_synth_preds)

    if len(delta_real) != len(delta_synth):
        real_disagreement_rate = float(np.mean(delta_real))
        v = np.maximum(real_disagreement_rate, delta_synth)
    else:
        v = np.maximum(delta_real, delta_synth)

    return v.astype(np.float64)


def fit_weight_surrogate(w, r_hat, eps_hat, v_hat, random_state=None, cv_folds=5):

    w = np.asarray(w, dtype=np.float64)
    r_hat = np.asarray(r_hat, dtype=np.float64)
    eps_hat = np.asarray(eps_hat, dtype=np.float64)
    v_hat = np.asarray(v_hat, dtype=np.float64)

    proxies = np.column_stack([r_hat, eps_hat, v_hat])
    proxy_names = ["feature_fidelity", "relationship_estimation", "model_specification"]

    proxy_corr = pd.DataFrame(proxies, columns=proxy_names).corr()

    kwargs = dict(n_estimators=50, max_depth=3)
    if random_state is not None:
        kwargs["random_state"] = random_state

    surrogate = XGBRegressor(**kwargs)

    n_splits = max(2, min(cv_folds, len(w) // 10))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_r2 = float(np.mean(cross_val_score(surrogate, proxies, w, cv=cv, scoring="r2")))

    surrogate.fit(proxies, w)
    in_sample_r2 = float(surrogate.score(proxies, w))

    gain_importance = surrogate.get_booster().get_score(importance_type="gain")
    feature_importance = {
        proxy_names[int(k[1:])]: v
        for k, v in gain_importance.items()
    }
    for name in proxy_names:
        feature_importance.setdefault(name, 0.0)

    base_pred = surrogate.predict(proxies)
    interaction_strength = {}
    for i in range(3):
        for j in range(i + 1, 3):
            ablated = proxies.copy()
            ablated[:, i] = proxies[:, i].mean()
            ablated[:, j] = proxies[:, j].mean()
            pred_ablate_both = surrogate.predict(ablated)

            ablated_i = proxies.copy()
            ablated_i[:, i] = proxies[:, i].mean()
            pred_ablate_i = surrogate.predict(ablated_i)

            ablated_j = proxies.copy()
            ablated_j[:, j] = proxies[:, j].mean()
            pred_ablate_j = surrogate.predict(ablated_j)

            joint_effect = np.var(base_pred - pred_ablate_both)
            additive_effect = np.var(base_pred - pred_ablate_i) + np.var(base_pred - pred_ablate_j)
            interaction_strength[f"{proxy_names[i]}_x_{proxy_names[j]}"] = float(
                max(joint_effect - additive_effect, 0.0)
            )

    proxy_df = pd.DataFrame({
        "feature_fidelity": r_hat,
        "relationship_estimation": eps_hat,
        "model_specification": v_hat,
        "weight": w,
    })

    diagnostics = {
        "proxy_correlation": proxy_corr,
        "feature_importance": feature_importance,
        "interaction_strength": interaction_strength,
    }

    return surrogate, in_sample_r2, cv_r2, proxy_df, diagnostics
