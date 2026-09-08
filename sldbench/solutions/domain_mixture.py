'''
Champion 35-parameter form, UNCHANGED:

    c = exp(rc);  d = exp(rd)
    phi = (p ** c) * d
    z   = phi @ W.T + b            (W symmetric 5x5, 15 free entries)
    yhat = a + exp(clip(z, -30, 30))

Single mechanism changed: per-domain residual weighting.

The champion minimises raw linear-space SSE, so the output columns with the
widest loss spread dominate the fit and the CV lambda selection; a narrow-spread
domain contributes almost nothing to the objective even when its predictions are
relatively poor.  Held-out R^2 over a multi-output target is normally the
uniform average of the per-column R^2 values, i.e. each column's error is
divided by that column's variance.  This fitter matches that: every residual is
divided by the training standard deviation of its own domain, and the same
weighting is applied when scoring the CV folds so lambda is selected against the
same quantity that is optimised.

Weights are re-normalised to mean 1, so the soft_l1 f_scale and the lambda grid
keep the scale they had in the champion.  Form, parameter layout, bounds, CV
protocol and the 50 fixed-seed restarts are byte-identical.
'''

import numpy as np
from scipy.optimize import least_squares


def _sym_from_params(sym_vec):
    '''Re-assemble a 5x5 symmetric matrix from its 15-element upper-triangular vector.'''
    W = np.zeros((5, 5), dtype=float)
    idx = 0
    for i in range(5):
        for j in range(i, 5):
            val = sym_vec[idx]
            W[i, j] = val
            W[j, i] = val
            idx += 1
    return W


def _flatten_sym(W):
    '''Extract the 15 upper-triangular (incl. diagonal) entries of a symmetric 5x5 matrix.'''
    return np.array([W[i, j] for i in range(5) for j in range(i, 5)], dtype=float)


def scaling_law_func(data_points, params):
    '''
    Predict multi-domain loss values.

    params layout: [a (5), b (5), rc (5), rd (5), w_sym (15)]
    '''
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 35:
        raise ValueError('Expected 35 parameters, got %d' % p.size)

    a = p[:5]
    b = p[5:10]
    rc = p[10:15]
    rd = p[15:20]
    w_sym = p[20:]

    c = np.exp(rc)
    d = np.exp(rd)
    c = np.clip(c, 0.01, 5.0)
    d = np.clip(d, 0.01, 10.0)

    W = _sym_from_params(w_sym)

    Xc = X ** c
    phi = Xc * d

    z = phi @ W.T + b
    z = np.clip(z, -30.0, 30.0)
    return a + np.exp(z)


def _init_params_linear_log(X, Y):
    '''Log-linear initialisation: a below the floor, then OLS in log space, symmetrised.'''
    eps = 1e-8
    margin = 0.05
    a0 = np.min(Y, axis=0) - margin

    Y_adj = np.clip(Y - a0, eps, None)
    Z = np.log(Y_adj)

    N = X.shape[0]
    A = np.column_stack([np.ones(N), X])
    coeffs, _r, _rank, _sv = np.linalg.lstsq(A, Z, rcond=None)

    b0 = coeffs[0]
    W0 = coeffs[1:].T

    W_sym0 = (W0 + W0.T) / 2.0
    w_sym0 = _flatten_sym(W_sym0)

    rc0 = np.zeros(5, dtype=float)
    rd0 = np.zeros(5, dtype=float)

    return np.concatenate([a0, b0, rc0, rd0, w_sym0])


def _shrink_target(init):
    '''Shrinkage target for p[10:]: rc = rd = 0 (c = d = 1), w_sym = log-linear OLS.'''
    t = np.array(init[10:], dtype=float, copy=True)
    t[:10] = 0.0
    return t


def _domain_weights(Y):
    '''Per-output-column inverse-scale weights, re-normalised to mean 1.'''
    sd = np.std(np.asarray(Y, dtype=float), axis=0)
    sd = np.clip(sd, 1e-6, None)
    w = 1.0 / sd
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        return np.ones(5, dtype=float)
    return w / m


def _bounds():
    lower = np.concatenate([
        np.full(5, -np.inf),
        np.full(5, -np.inf),
        np.full(5, np.log(0.01)),
        np.full(5, np.log(0.01)),
        np.full(15, -np.inf),
    ])
    upper = np.concatenate([
        np.full(5, np.inf),
        np.full(5, np.inf),
        np.full(5, np.log(5.0)),
        np.full(5, np.log(10.0)),
        np.full(15, np.inf),
    ])
    return lower, upper


def _residual_factory(X, Y, lam, target, wcol):
    '''lam = 0 keeps the tiny 1e-12 ridge on p[5:]; data rows are column-weighted.'''
    s = float(np.sqrt(lam))

    def residuals(p):
        pred = scaling_law_func(X, p)
        data_res = ((pred - Y) * wcol).ravel()
        if s > 0.0:
            return np.concatenate([data_res, s * (p[10:] - target)])
        return np.concatenate([data_res, np.sqrt(1e-12) * p[5:]])

    return residuals


def _fit_once(X, Y, lam, x0, target, lower, upper, max_nfev, wcol):
    return least_squares(
        _residual_factory(X, Y, lam, target, wcol),
        x0=x0,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        verbose=0,
    )


def fit_scaling_law(data_points, loss_values):
    '''
    Fit the champion law with domain-standardised residuals and a CV-selected
    shrinkage strength.

    1. deterministic 5-fold CV over a fixed lambda grid, scored by held-out
       domain-standardised squared error;
    2. refit on all rows at the selected lambda with the primary solve plus
       50 fixed-seed restarts.
    '''
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    Y = np.atleast_2d(np.asarray(loss_values, dtype=float))

    if X.shape[1] != 5:
        raise ValueError('Expected 5 input dimensions, got %d' % X.shape[1])
    if Y.shape != X.shape:
        raise ValueError('Loss array must match shape of inputs')

    lower, upper = _bounds()
    init_full = _init_params_linear_log(X, Y)
    target_full = _shrink_target(init_full)
    wcol = _domain_weights(Y)

    N = X.shape[0]
    lam_grid = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    best_lam = 0.0

    if N >= 10:
        k = 5
        idx = np.arange(N)
        folds = [(idx[idx % k != f], idx[idx % k == f]) for f in range(k)]

        cv = np.full(lam_grid.size, np.inf)
        for li in range(lam_grid.size):
            lam = float(lam_grid[li])
            total = 0.0
            ok = True
            for tr, te in folds:
                if tr.size < 6 or te.size == 0:
                    continue
                Xtr = X[tr]
                Ytr = Y[tr]
                init_tr = _init_params_linear_log(Xtr, Ytr)
                tgt_tr = _shrink_target(init_tr)
                r = _fit_once(Xtr, Ytr, lam, init_tr, tgt_tr, lower, upper, 3000, wcol)
                p_hat = r.x if r.success else init_tr
                pred = scaling_law_func(X[te], p_hat)
                if not np.all(np.isfinite(pred)):
                    ok = False
                    break
                total += float(np.sum(((pred - Y[te]) * wcol) ** 2))
            cv[li] = total if ok else np.inf

        best_cv = float(np.min(cv))
        if np.isfinite(best_cv):
            tol = 1.02 * best_cv
            cands = [float(lam_grid[i]) for i in range(lam_grid.size) if cv[i] <= tol]
            best_lam = max(cands) if len(cands) > 0 else float(lam_grid[int(np.argmin(cv))])

    best_params = init_full
    best_cost = np.inf

    result = _fit_once(X, Y, best_lam, init_full, target_full, lower, upper, 15000, wcol)
    if result.success:
        best_params = result.x
        best_cost = result.cost

    rng = np.random.default_rng(12345)
    for _ in range(50):
        scale = 0.2
        perturb = best_params + rng.normal(scale=scale, size=best_params.shape) * np.maximum(np.abs(best_params), 1.0)
        perturb = np.clip(perturb, lower, upper)

        res = _fit_once(X, Y, best_lam, perturb, target_full, lower, upper, 8000, wcol)
        if res.success and res.cost < best_cost:
            best_params = res.x
            best_cost = res.cost

    return best_params
