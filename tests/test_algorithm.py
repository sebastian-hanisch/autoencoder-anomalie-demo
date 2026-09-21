"""Autoencoder als Anomalie-Detektor: Gradienten gegen Finite Differenzen, Aufbau (Engpass, Parameterzahl), das PCA-Residuum gegen die Singulärwertzerlegung, das lineare Autoencoder gegen den PCA-Unterraum (Baldi und Hornik),
Schwelle, Determinismus und Netz-Seed; die kopierten Bausteine der Vorgänger (One-Class SVM mit sklearn-Kreuzprüfung, Deep SVDD, LOF, Isolation Forest, χ², MCD, ECOD)."""

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.svm import OneClassSVM

import aeano_algorithm as ae
import aeano_dsvdd as dv
import aeano_ecod as ecod
import aeano_ee_algorithm as alg
import aeano_isolation_forest as isf
import aeano_lof as lof
import aeano_ocsvm as svm


def _planar(n=200, p=6, n_out=10, noise=0.05, seed=0):
    """Touren, die (bis auf kleines Rauschen) in einer zweidimensionalen Ebene liegen; die letzten n_out verlassen die Ebene."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((n, 2))
    W = rng.standard_normal((2, p))
    X = t @ W + noise * rng.standard_normal((n, p))
    X[n - n_out:] += 2.0 * rng.standard_normal((n_out, p))
    return X, np.arange(n) >= n - n_out


def _auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- Gradienten und Aufbau ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("n_code", [1, 3])
def test_backpropagation_matches_finite_differences(activation, depth, n_code):
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((20, 5))
    w, b = ae.init_parameters(ae.layer_sizes(5, depth, 6, n_code), activation, 1)
    b = [rng.normal(size=len(x)) * 0.1 for x in b]
    loss, gw, gb = ae.loss_and_gradients(w, b, Z, activation, depth)
    eps = 1e-6

    def numeric(arr, idx):
        old = arr[idx]
        arr[idx] = old + eps
        up = ae.loss_and_gradients(w, b, Z, activation, depth)[0]
        arr[idx] = old - eps
        down = ae.loss_and_gradients(w, b, Z, activation, depth)[0]
        arr[idx] = old
        return (up - down) / (2 * eps)
    for layer in range(len(w)):
        for idx in [(0, 0), (w[layer].shape[0] - 1, w[layer].shape[1] - 1), (1 % w[layer].shape[0], 2 % w[layer].shape[1])]:
            assert numeric(w[layer], idx) == pytest.approx(gw[layer][idx], rel=1e-4, abs=1e-7)
        assert numeric(b[layer], (0,)) == pytest.approx(gb[layer][0], rel=1e-4, abs=1e-7)
    assert loss == pytest.approx(float(((ae.forward(w, b, Z, activation, depth)[-1] - Z) ** 2).mean()))


def test_activation_derivatives_match_finite_differences():
    x = np.linspace(-2, 2, 9) + 0.0123
    for name in ae.ACTIVATIONS:
        h = 1e-6
        numeric = (ae.activate(name, x + h) - ae.activate(name, x - h)) / (2 * h)
        assert np.allclose(ae.activation_grad(name, ae.activate(name, x)), numeric, atol=1e-6)


def test_layer_sizes_bottleneck_and_parameter_count():
    assert ae.layer_sizes(12, 0, 16, 2) == [12, 2, 12] and ae.layer_sizes(12, 2, 16, 3) == [12, 16, 16, 3, 16, 16, 12]
    assert ae.n_parameters([12, 2, 12]) == 12 * 2 + 2 + 2 * 12 + 12
    X, _ = _planar()
    for depth, k in ((0, 1), (1, 2), (2, 3)):
        m = ae.fit_autoencoder(X, n_code=k, depth=depth, width=8, n_epochs=5)
        assert m.code.shape == (len(X), k) and m.n_parameters == ae.n_parameters(ae.layer_sizes(X.shape[1], depth, 8, k))
        assert np.array_equal(ae.encode(m, X), m.code)


# --- Wert, Schwelle, Training ------------------------------------------------------------------------------------------------------------------


def test_score_is_the_mean_squared_reconstruction_error_and_the_feature_errors_sum_to_it():
    X, _ = _planar()
    m = ae.fit_autoencoder(X, n_code=2, depth=1, width=8, n_epochs=20, seed=1)
    Zn, rec, err = ae.reconstruct(m)
    assert np.allclose(err, (Zn - rec) ** 2) and np.allclose(err.mean(axis=1), m.errors) and np.array_equal(ae.score(m), m.errors)
    assert np.allclose(ae.score(m, X[:9]), m.errors[:9])                                                      # neue Touren durch dasselbe Netz


def test_threshold_is_the_one_minus_nu_quantile_and_flags_at_most_nu():
    X, _ = _planar(n=300)
    for nu in (0.05, 0.1, 0.3):
        m = ae.fit_autoencoder(X, nu=nu, n_epochs=30, seed=2)
        assert m.threshold == ae.quantile_threshold(m.errors, nu) and float((m.errors > m.threshold).mean()) <= nu + 1e-9 and float((m.errors >= m.threshold).mean()) >= nu - 1e-9


def test_the_loss_falls_during_training_and_zero_epochs_is_the_untrained_network():
    X, _ = _planar()
    m = ae.fit_autoencoder(X, n_code=2, depth=1, width=16, n_epochs=300, seed=1)
    assert m.loss_history[-1] < 0.3 * m.loss_history[0] and np.isfinite(m.loss_history).all() and set(m.snapshots) >= {0, 300}
    assert np.array_equal(m.snapshots[300], m.errors) and np.array_equal(m.code_snapshots[300], m.code)
    z = ae.fit_autoencoder(X, n_epochs=0, seed=1)
    assert len(z.loss_history) == 0 and np.array_equal(z.snapshots[0], z.errors) and z.n_epochs == 0


def test_deterministic_for_a_seed_and_different_for_another():
    X, _ = _planar()
    a = ae.fit_autoencoder(X, n_epochs=40, seed=5)
    b = ae.fit_autoencoder(X, n_epochs=40, seed=5)
    c = ae.fit_autoencoder(X, n_epochs=40, seed=6)
    assert np.array_equal(a.errors, b.errors) and np.array_equal(a.loss_history, b.loss_history) and not np.allclose(a.errors, c.errors)


def test_a_trained_autoencoder_finds_the_tours_that_leave_the_plane():
    X, y = _planar()
    m = ae.fit_autoencoder(X, n_code=2, depth=1, width=16, n_epochs=300, seed=0)
    assert _auc(m.errors, y) > 0.95 and _auc(ae.pca_residual(X, 2), y) > 0.95


def test_invalid_arguments_are_rejected():
    X, _ = _planar(n=30)
    for kw in ({"activation": "sigmoid"}, {"nu": 0.0}, {"nu": 1.5}, {"depth": -1}, {"n_code": 0}):
        with pytest.raises(ValueError):
            ae.fit_autoencoder(X, n_epochs=2, **kw)


# --- PCA-Residuum und das lineare Autoencoder ---------------------------------------------------------------------------------------------------


def test_pca_residual_hand_instance_and_svd_reference():
    t = np.linspace(-1.0, 1.0, 21)
    X = np.stack([t, 2.0 * t + 1.0, -3.0 * t], axis=1)                                      # Rang 1
    assert np.allclose(ae.pca_residual(X, 1), 0.0, atol=1e-12) and ae.pca_residual(X, 3).max() < 1e-12
    Xr, _ = _planar()
    sd = Xr.std(axis=0, ddof=1)
    Z = (Xr - Xr.mean(axis=0)) / sd
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    for k in (1, 2, 3):
        reference = ((Z - Z @ vt[:k].T @ vt[:k]) ** 2).mean(axis=1)
        assert np.allclose(ae.pca_residual(Xr, k), reference)
        assert ae.pca_residual(Xr, k).mean() == pytest.approx(float((s[k:] ** 2).sum() / Z.size))                # Eckart-Young
    assert all(ae.pca_residual(Xr, k).mean() >= ae.pca_residual(Xr, k + 1).mean() for k in range(1, 5)) and (ae.pca_residual(Xr, 2) >= 0).all()


def test_the_linear_autoencoder_converges_to_the_pca_subspace():
    X, _ = _planar(n=300, noise=0.2)
    m = ae.fit_autoencoder(X, n_code=2, depth=0, n_epochs=2000, lr=0.01, seed=0)
    angles = ae.linear_angle_curve(m, ae.pca_axes(m.Z, 2))
    assert angles[2000] < 8.0 and angles[2000] < angles[0]                                                    # nach ausreichend Training derselbe Unterraum (nicht dieselben Achsen)
    assert m.errors.mean() == pytest.approx(ae.pca_residual(X, 2).mean(), rel=0.1)
    with pytest.raises(ValueError):
        ae.linear_angle_curve(ae.fit_autoencoder(X, depth=1, n_epochs=2), ae.pca_axes(m.Z, 2))


def test_principal_angles_of_equal_and_orthogonal_subspaces():
    A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert np.allclose(ae.principal_angles(A, A), 0.0, atol=1e-6) and ae.principal_angles(A, np.array([[0, 0, 1.0]])).max() == pytest.approx(90.0)


# --- Kopierte Komponenten der Vorgänger -----------------------------------------------------------------------------------------------------


def test_one_class_svm_still_matches_scikit_learn():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 5))
    X[:10] += 4
    g = svm.gamma_scale(X)
    f = svm.fit_ocsvm(X, 0.1, g, tol=1e-9)
    m = OneClassSVM(kernel="rbf", gamma=g, nu=0.1, tol=1e-9).fit(X)
    Z = rng.standard_normal((30, 5))
    assert np.abs(svm.decision_function(f, Z) - m.decision_function(Z)).max() < 1e-6


def test_deep_svdd_kernel_is_bias_free_and_deterministic():
    X, _ = _planar()
    a = dv.fit_deep_svdd(X, n_epochs=20, seed=2)
    b = dv.fit_deep_svdd(X, n_epochs=20, seed=2)
    assert a.biases is None and np.array_equal(a.outputs, b.outputs) and float((dv.distances2(a) > a.r2).mean()) <= 0.1 + 1e-9


def test_other_copied_components_still_behave_like_their_predecessors():
    rng = np.random.default_rng(8)
    X = np.concatenate([rng.standard_normal((200, 3)), 7 + rng.standard_normal((10, 3))])
    y = np.arange(210) >= 200
    f = lof.fit_lof(X, 20)
    assert f.lof[y].min() > 1.5 and np.median(f.lof[~y]) < 1.2
    forest = isf.fit_forest(X, 100, 128, 0)
    sc = isf.score_from_paths(isf.path_lengths(forest, X), forest.psi)
    assert sc[y].min() > sc[~y].mean() and (sc >= 0).all() and (sc <= 1).all()
    r = alg.fit_mcd(X, 0.75, 0)
    assert r.d2[y].min() > alg.threshold(r, 0.975)
    assert alg.chi2_ppf(0.975, 12) == pytest.approx(chi2.ppf(0.975, 12), rel=1e-6)
    left, right = ecod.tail_probabilities(np.array([[1.0], [2.0], [2.0], [3.0], [10.0]]))
    assert np.allclose(left[:, 0], [0.2, 0.6, 0.6, 0.8, 1.0]) and np.allclose(right[:, 0], [1.0, 0.8, 0.8, 0.4, 0.2])
