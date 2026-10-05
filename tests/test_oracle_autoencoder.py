"""Unabhängiges Orakel für den Autoencoder und die kopierten Bausteine: Verlauf des Trainingsverlusts (Index k = nach k Schritten, der letzte Wert gehört zum fertigen Netz), Gradient jeder einzelnen
Komponente per zentraler Differenz, Adam gegen die Lehrbuchformel, PCA-Residuum und Hauptwinkel gegen scikit-learn und scipy, Schwelle, LOF gegen scikit-learn (Gleichstände ausgeschlossen)."""

import numpy as np
import pytest

import aeano_algorithm as ae
import aeano_lof as lof


def _data(n=60, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + 3.0


def test_loss_history_ends_with_the_loss_of_the_finished_network():
    """Früher maß die Schleife den Verlust vor dem Schritt: der letzte Wert gehörte zum Netz nach n - 1 Schritten und wich bis zu 11 % vom Verlust des gezeigten Netzes ab (Adam schwingt)."""
    X = _data()
    for seed, epochs in ((0, 7), (1, 40), (2, 120)):
        m = ae.fit_autoencoder(X, n_code=2, depth=1, width=8, lr=0.01, n_epochs=epochs, seed=seed)
        _, rec, _ = ae.reconstruct(m)
        assert len(m.loss_history) == epochs + 1
        assert m.loss_history[-1] == pytest.approx(float(((rec - m.Z) ** 2).mean()), rel=1e-12)
        assert m.loss_history[-1] == pytest.approx(float(m.errors.mean()), rel=1e-12)
    short = ae.fit_autoencoder(X, n_code=2, depth=1, width=8, lr=0.01, n_epochs=5, seed=3)
    long = ae.fit_autoencoder(X, n_code=2, depth=1, width=8, lr=0.01, n_epochs=9, seed=3)
    assert np.allclose(long.loss_history[:6], short.loss_history)                      # Index k = Verlust nach k Schritten, unabhängig von der Gesamtzahl der Epochen


def _full_gradient_error(loss_fn, arrays, grads, eps=1e-6):
    worst = 0.0
    for arr, g in zip(arrays, grads):
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = loss_fn()
            arr[idx] = old - eps
            down = loss_fn()
            arr[idx] = old
            num = (up - down) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / max(1.0, abs(num), abs(g[idx])))
    return worst


@pytest.mark.parametrize("activation,depth,n_code,width", [("tanh", 0, 1, 3), ("tanh", 1, 2, 4), ("tanh", 2, 3, 3), ("relu", 1, 2, 5), ("relu", 2, 1, 4)])
def test_every_gradient_component_matches_central_differences(activation, depth, n_code, width):
    rng = np.random.default_rng(10 + depth)
    Z = rng.standard_normal((9, 4))
    w, b = ae.init_parameters(ae.layer_sizes(4, depth, width, n_code), activation, 7)
    b = [rng.normal(size=len(x)) * 0.3 for x in b]
    _, gw, gb = ae.loss_and_gradients(w, b, Z, activation, depth)
    f = lambda: ae.loss_and_gradients(w, b, Z, activation, depth)[0]
    assert _full_gradient_error(f, w, gw) < 1e-6 and _full_gradient_error(f, b, gb) < 1e-6


def test_training_follows_textbook_adam():
    """Adam nach Kingma und Ba (Algorithmus 1, mit Bias-Korrektur von m und v), mit den Gradienten der Demo, 25 Schritte: gleiche Gewichte bis auf die Stellung von epsilon (< 1e-3)."""
    X = _data(40, 4, 1)
    m = ae.fit_autoencoder(X, n_code=2, depth=1, width=6, lr=0.01, n_epochs=25, seed=4)
    w, b = ae.init_parameters(ae.layer_sizes(4, 1, 6, 2), "tanh", 4)
    params = w + b
    mom, var = [np.zeros_like(x) for x in params], [np.zeros_like(x) for x in params]
    for t in range(1, 26):
        _, gw, gb = ae.loss_and_gradients(w, b, m.Z, "tanh", 1)
        for k, g in enumerate(gw + gb):
            mom[k] = 0.9 * mom[k] + 0.1 * g
            var[k] = 0.999 * var[k] + 0.001 * g ** 2
            params[k] -= 0.01 * (mom[k] / (1 - 0.9 ** t)) / (np.sqrt(var[k] / (1 - 0.999 ** t)) + 1e-8)
    assert max(np.abs(a - c).max() for a, c in zip(m.weights + m.biases, w + b)) < 1e-3


def test_pca_residual_equals_scikit_learn_and_principal_angles_equal_scipy():
    decomposition = pytest.importorskip("sklearn.decomposition")
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(5)
    for _ in range(25):
        n, p = int(rng.integers(8, 60)), int(rng.integers(2, 7))
        k = int(rng.integers(1, p))
        X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + rng.normal(size=p) * 10
        Z = (X - X.mean(0)) / X.std(0, ddof=1)
        pca = decomposition.PCA(n_components=k).fit(Z)
        ref = ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).mean(axis=1)
        assert np.allclose(ae.pca_residual(X, k), ref, atol=1e-12)
        A, B = rng.standard_normal((k, p)), rng.standard_normal((k, p))
        assert np.allclose(np.sort(ae.principal_angles(A, B)), np.sort(np.degrees(linalg.subspace_angles(A.T, B.T))), atol=1e-4)


def test_threshold_flags_at_most_the_share_nu_also_with_ties():
    rng = np.random.default_rng(6)
    for i in range(400):
        n, nu = int(rng.integers(2, 120)), float(rng.choice([0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0]))
        e = rng.standard_normal(n)
        if i % 3 == 0:
            e = np.round(e, 1)
        assert (e > ae.quantile_threshold(e, nu)).sum() <= nu * n + 1e-9


def test_lof_equals_scikit_learn_including_new_points():
    neighbors = pytest.importorskip("sklearn.neighbors")
    rng = np.random.default_rng(7)
    for _ in range(40):
        n, p = int(rng.integers(6, 60)), int(rng.integers(1, 4))
        X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3, p)
        X[: max(1, n // 10)] += rng.uniform(3, 8)
        k = lof.clamp_k(int(rng.integers(1, n)), n)
        fit = lof.fit_lof(X, k)
        sk = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute").fit(X)
        assert np.allclose(fit.lof, -sk.negative_outlier_factor_, rtol=1e-8)
        Q = rng.standard_normal((6, p)) * 3
        skn = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute", novelty=True).fit(X)
        assert np.allclose(lof.score_new(X, fit, Q), -skn.score_samples(Q), rtol=1e-8)
