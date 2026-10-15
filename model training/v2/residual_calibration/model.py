"""Support-only personal residual regression; query labels are not an input."""
import numpy as np


def candidates():
    rows = [dict(name='bias', family='bias'), dict(name='calibration_mean', family='mean')]
    for family in ('prediction_linear', 'feature_linear', 'feature_rbf'):
        for width in ((.25, 1., 4.) if family == 'feature_rbf' else (1.,)):
            for penalty in (.01, .1, 1., 10.):
                rows.append(dict(name=f'{family}_w{width:g}_a{penalty:g}', family=family,
                                 width=width, penalty=penalty))
    return rows


def kernel(a, b, family, width):
    if family.endswith('linear'):
        return a @ b.T / a.shape[1]
    distance = np.maximum((a*a).mean(1)[:, None] + (b*b).mean(1)[None, :]
                          - 2*a@b.T/a.shape[1], 0.)
    return np.exp(-distance/width)


def fit(config, features, predictions, labels, source_center, source_scale):
    h, p, y = [np.asarray(v, np.float64) for v in (features, predictions, labels)]
    if len(y) < 1 or p.shape != y.shape or y.shape[1] != 2 or len(h) != len(y):
        raise ValueError('Misaligned or empty support')
    if not all(np.isfinite(v).all() for v in (h, p, y)):
        raise ValueError('Nonfinite support')
    state = dict(config=config, mean_y=y.mean(0), mean_residual=(y-p).mean(0),
                 source_center=np.asarray(source_center), source_scale=np.asarray(source_scale))
    family = config['family']
    if family in ('bias', 'mean'):
        return state
    z = (p-state['source_center'])/state['source_scale'] if family == 'prediction_linear' else h
    k = kernel(z, z, family, config['width'])
    column_mean, grand = k.mean(0), k.mean()
    kc = k - column_mean[None, :] - column_mean[:, None] + grand
    residual = y-p-state['mean_residual']
    coef = np.linalg.solve(kc + len(y)*config['penalty']*np.eye(len(y)), residual)
    state.update(support_features=z, column_mean=column_mean, grand=grand, coefficients=coef)
    return state


def predict(state, features, predictions):
    p = np.asarray(predictions, np.float64)
    family = state['config']['family']
    if family == 'mean':
        return np.tile(state['mean_y'], (len(p), 1))
    base = p + state['mean_residual']
    if family == 'bias':
        return base
    z = (p-state['source_center'])/state['source_scale'] if family == 'prediction_linear' else np.asarray(features, np.float64)
    k = kernel(z, state['support_features'], family, state['config']['width'])
    centered = k - k.mean(1)[:, None] - state['column_mean'][None, :] + state['grand']
    return base + centered @ state['coefficients']


def forward_cv(features, predictions, labels, start, end, source_center, source_scale):
    """Select each output using only early calibration records, then refit all."""
    h, p, y, start, end = [np.asarray(v) for v in (features, predictions, labels, start, end)]
    lo, hi = float(start.min()), float(end.max())
    folds = []
    for begin, stop in ((.5, .75), (.75, 1.)):
        left, right = lo + begin*(hi-lo), lo + stop*(hi-lo)
        train = np.flatnonzero(end + 10 <= left + 1e-6)
        validation = np.flatnonzero((start >= left) & (end <= right+1e-6))
        if len(train) >= 2 and len(validation):
            folds.append((train, validation))
    configs = candidates()
    errors = np.zeros((len(configs), 2)); count = 0
    for train, validation in folds:
        count += len(validation)
        for i, cfg in enumerate(configs):
            state = fit(cfg, h[train], p[train], y[train], source_center, source_scale)
            errors[i] += abs(predict(state, h[validation], p[validation])-y[validation]).sum(0)
    selected = np.argmin(errors, axis=0) if count else np.zeros(2, int)
    states = [fit(configs[int(i)], h, p, y, source_center, source_scale) for i in selected]
    audit = dict(folds=[dict(train=t.tolist(), validation=v.tolist()) for t, v in folds],
                 selected=[configs[int(i)]['name'] for i in selected],
                 count=count, query_labels_used=False,
                 validation_mae=(errors/count).tolist() if count else None)
    return states, audit


def predict_endpoint_states(states, features, predictions):
    return np.column_stack([predict(state, features, predictions)[:, j] for j, state in enumerate(states)])
