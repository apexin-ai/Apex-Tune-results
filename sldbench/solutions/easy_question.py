'''
Six-parameter U-shaped scaling law (exponential baseline + Gaussian bump)
with cross-validated selection of the non-linear geometry.

Model (unchanged from the champion):
    y(x) = a + b*exp(c*x) + d*exp(-0.5*((x - e)/f)**2),  f = exp(log_f) > 0.

Only (c, e, log_f) are non-linear; (a, b, d) come from a closed-form
least-squares solve, so the search is 3-dimensional and the model keeps six
free parameters.

Delta w.r.t. the champion: candidate triplets are no longer ranked by TRAIN
SSE.  Each candidate (every DE optimum, the heuristic seed, and both local
refinements) is scored by a deterministic, x-stratified 5-fold CV error where
the three linear coefficients are refit on the training folds and evaluated on
the held-out fold.  The CV winner is then re-solved on the full data.  Train
SSE is used only as a fallback tie-breaker if CV is unavailable (tiny groups).
'''

import numpy as np
import os
from scipy.optimize import differential_evolution, least_squares


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _prepare_x(data_points):
    '''Extract a 1-D array of log-FLOPs from the input.'''
    arr = np.asarray(data_points, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr[:, 0]


def _design(x, c, e, f):
    '''Design matrix [1, exp(c*x), gauss(x; e, f)].'''
    exp_part = np.exp(np.clip(c * x, -200.0, 200.0))
    gauss_part = np.exp(np.clip(-0.5 * ((x - e) / f) ** 2, -200.0, 200.0))
    return np.column_stack((np.ones_like(x), exp_part, gauss_part))


def _solve_linear_coeffs(x, y, c, e, f):
    '''Solve analytically for the linear coefficients (a, b, d).'''
    A = _design(x, c, e, f)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def _sse_for_triplet(x, y, c, e, log_f):
    '''SSE and the full 6-parameter vector for a given (c, e, log_f).'''
    f = np.exp(log_f)
    A = _design(x, c, e, f)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coeffs
    sse = float(np.sum((pred - y) ** 2))
    if not np.isfinite(sse):
        sse = np.inf
    return sse, np.array([coeffs[0], coeffs[1], coeffs[2], c, e, log_f], dtype=float)


def _make_folds(x, n_folds):
    '''Deterministic x-stratified folds: sort by x, then interleave.'''
    order = np.argsort(x, kind='stable')
    return [order[k::n_folds] for k in range(n_folds)]


def _cv_error(x, y, c, e, log_f, folds):
    '''Mean held-out squared error; only linear coeffs are refit per fold.'''
    if not np.all(np.isfinite([c, e, log_f])):
        return np.inf
    f = np.exp(log_f)
    A = _design(x, c, e, f)
    if not np.all(np.isfinite(A)):
        return np.inf
    n = x.size
    total = 0.0
    count = 0
    for te in folds:
        if te.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        if int(mask.sum()) < 6:
            continue
        try:
            coeffs, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        except Exception:
            return np.inf
        r = A[te] @ coeffs - y[te]
        total += float(np.sum(r * r))
        count += int(te.size)
    if count == 0 or not np.isfinite(total):
        return np.inf
    return total / float(count)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def scaling_law_func(data_points, params):
    '''Predict Brier-score from log-FLOPs using the fitted scaling law.'''
    x = _prepare_x(data_points)                          # (N,)
    P = np.atleast_2d(np.asarray(params, dtype=float))   # (T,6)

    a = P[:, 0][:, None]
    b = P[:, 1][:, None]
    d = P[:, 2][:, None]
    c = P[:, 3][:, None]
    e = P[:, 4][:, None]
    log_f = P[:, 5][:, None]
    f = np.exp(np.clip(log_f, -50.0, 50.0))

    x_row = x[None, :]

    exp_arg = np.clip(c * x_row, -200.0, 200.0)
    gauss_arg = np.clip(-0.5 * ((x_row - e) / f) ** 2, -200.0, 200.0)

    preds = a + b * np.exp(exp_arg) + d * np.exp(gauss_arg)   # (T, N)
    preds = preds.T

    if preds.shape[1] == 1:
        return preds[:, 0]
    return preds


def fit_scaling_law(data_points, loss_values):
    '''Fit the six-parameter U-shaped law, selecting geometry by CV error.'''
    x = _prepare_x(data_points)
    Y = np.asarray(loss_values, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    N, T = Y.shape

    rng = np.random.default_rng(int(os.environ.get('EQ_SEED', '46')))
    n_folds = 5 if N >= 25 else (3 if N >= 12 else 0)
    folds = _make_folds(x, n_folds) if n_folds > 0 else []

    all_params = []

    for t in range(T):
        y = Y[:, t]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        c_bounds = (-30.0, -1e-6)
        e_bounds = (x_min, x_max)
        logf_bounds = (-8.0, 8.0)
        nonlin_bounds = [c_bounds, e_bounds, logf_bounds]
        lo = [c_bounds[0], e_bounds[0], logf_bounds[0]]
        hi = [c_bounds[1], e_bounds[1], logf_bounds[1]]

        def _candidate_sse(p):
            c, e, log_f = p
            sse, _ = _sse_for_triplet(x, y, c, e, log_f)
            return sse

        def _ls_residual(p):
            c, e, log_f = p
            f = np.exp(log_f)
            A = _design(x, c, e, f)
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = A @ coeffs - y
            return np.where(np.isfinite(r), r, 1e6)

        triplets = []

        # 1) Multiple differential-evolution runs - keep EVERY optimum
        for _ in range(4):
            de_res = differential_evolution(
                _candidate_sse,
                bounds=nonlin_bounds,
                strategy='best1bin',
                maxiter=250,
                popsize=20,
                tol=1e-7,
                polish=False,
                updating='deferred',
                seed=int(rng.integers(2**32 - 1)),
                disp=False,
            )
            triplets.append(np.asarray(de_res.x, dtype=float))

        # 2) Heuristic seed - bump centred at the worst observed point
        heuristic = np.array([
            float(np.clip(-1.0, *c_bounds)),
            float(np.clip(float(x[np.argmax(y)]), *e_bounds)),
            float(np.clip(np.log(max((x_max - x_min) / 4.0, 1e-3)), *logf_bounds)),
        ])
        triplets.append(heuristic)

        # 3) Local refinements (soft-L1 and L2) seeded from the best-SSE triplet
        seed_triplet = min(triplets, key=lambda p: _candidate_sse(p))
        for loss_name in ('soft_l1', 'linear'):
            try:
                res = least_squares(
                    _ls_residual, x0=seed_triplet, bounds=(lo, hi),
                    method='trf', loss=loss_name,
                    ftol=1e-9, xtol=1e-9, max_nfev=3000,
                )
                if res.success:
                    triplets.append(np.asarray(res.x, dtype=float))
            except Exception:
                pass

        # 4) Selection by cross-validated held-out error (NOT train SSE)
        scored = []
        for p in triplets:
            if not np.all(np.isfinite(p)):
                continue
            p = np.array([
                float(np.clip(p[0], *c_bounds)),
                float(np.clip(p[1], *e_bounds)),
                float(np.clip(p[2], *logf_bounds)),
            ])
            sse_tr, params_full = _sse_for_triplet(x, y, *p)
            if not np.isfinite(sse_tr) or not np.all(np.isfinite(params_full)):
                continue
            cv = _cv_error(x, y, p[0], p[1], p[2], folds) if folds else np.inf
            key = cv if np.isfinite(cv) else (1e12 + sse_tr)
            scored.append((key, sse_tr, params_full))

        if scored:
            scored.sort(key=lambda tup: (tup[0], tup[1]))
            best_params = scored[0][2]
        else:
            best_params = np.array([float(np.mean(y)), 0.0, 0.0, -1.0,
                                    0.5 * (x_min + x_max), 0.0])

        all_params.append(best_params)

    params_opt = np.vstack(all_params)
    if params_opt.shape[0] == 1:
        return params_opt[0]
    return params_opt
