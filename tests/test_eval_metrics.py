# Validate the evaluation logic (MAE / RMSE / MAPE) against hand-computed fixtures.
import os, sys, math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import eval_metrics as em


def test_metrics_hand_fixture():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([0.0, 3.0, 2.0])
    m = em.metrics(pred, true, robust_thresh=1.0)
    assert math.isclose(m['mae'], (1 + 1 + 1) / 3, abs_tol=1e-9)          # 1.0
    assert math.isclose(m['rmse'], math.sqrt((1 + 1 + 1) / 3), abs_tol=1e-9)  # 1.0
    # naive MAPE = mean(|(true-pred)/true|)*100 over the 3 points (guarded)
    mae_pts = np.abs((true - pred) / (np.abs(true) + 1e-8)) * 100.0
    assert math.isclose(m['mape'], mae_pts.mean(), rel_tol=1e-6)


def test_robust_mape_subset():
    pred = np.array([1.0, 100.0, 0.0])
    true = np.array([1.0, 90.0, 0.0])   # last point: y==0 -> excluded from robust subset
    m = em.metrics(pred, true, robust_thresh=1.0)
    # robust subset = points with |true|>=1 -> indices 0 and 1
    sub = np.abs((true[:2] - pred[:2]) / np.abs(true[:2])) * 100.0
    assert math.isclose(m['robust_mape'], sub.mean(), rel_tol=1e-6)
    assert m['frac_small'] == pytest.approx(1 / 3)


def test_load_pair_shape_normalization(tmp_path):
    pred = np.zeros((4, 2, 5, 3)); true = np.zeros((4, 2, 5, 3))
    np.save(os.path.join(tmp_path, 'tag_pred.npy'), pred)
    np.save(os.path.join(tmp_path, 'tag_true.npy'), true)
    r = em.load_pair('tag', str(tmp_path))
    assert r is not None
    p, t = r
    assert p.ndim == 3 and p.shape == (8, 5, 3)


def test_load_pair_missing(tmp_path):
    assert em.load_pair('nope', str(tmp_path)) is None
