"""Autoencoder als Anomalie-Detektor (numpy von Grund auf, Backpropagation von Hand, Adam, Vollbatch-Training) und das PCA-Residuum als Bezug.

Ein Netz aus **Encoder** (p -> Breite^Tiefe -> Engpass k, linear) und **Decoder** (k -> Breite^Tiefe -> p, linear) wird auf den Trainingstouren trainiert, sie möglichst genau zu rekonstruieren (mittleres Fehlerquadrat).
Der **Anomalie-Wert** einer Tour ist ihr Rekonstruktionsfehler (mittleres Fehlerquadrat über die Merkmale, in z-Einheiten): was das Netz nicht rekonstruieren kann, wird als Anomalie gewertet.

- **Tiefe 0**: linearer Autoencoder p -> k -> p. Es spannt (bei ausreichendem Training) denselben Unterraum auf wie die ersten k Hauptkomponenten der PCA (Baldi und Hornik); sein Rekonstruktionsfehler ist dann das **PCA-Residuum**.
- **Tiefe >= 1**: nichtlineare Aktivierung (tanh oder ReLU) in den verdeckten Schichten; der Engpass und die Ausgabeschicht bleiben linear.
- **Schwelle**: (1 - nu)-Quantil der Trainingsfehler (markiert höchstens den Anteil nu) - wie bei One-Class SVM und Deep SVDD ist der Anteil eingebaut.
- Initialisierung Xavier (tanh) bzw. He (ReLU), Bias-Terme bei 0; der Netz-Seed ist vom Datensatz-Seed entkoppelt."""

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("tanh", "relu")
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def activate(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def activation_grad(name, out):
    """Ableitung der Aktivierung, ausgedrückt durch ihren AUSGABEwert."""
    if name == "tanh":
        return 1.0 - out ** 2
    if name == "relu":
        return (out > 0).astype(float)
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def layer_sizes(n_features, depth, width, n_code):
    return [n_features] + [width] * depth + [n_code] + [width] * depth + [n_features]


def n_parameters(sizes):
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def _has_activation(layer, depth, n_layers):
    """Aktivierung nach jeder Schicht außer dem Engpass (Index `depth`) und der Ausgabeschicht."""
    return layer != depth and layer != n_layers - 1


def init_parameters(sizes, activation, seed):
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if activation == "relu":
            weights.append(rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out))
    return weights, biases


def forward(weights, biases, Z, activation, depth):
    """Ausgaben aller Schichten (Index 0 = Eingabe); der Engpass liegt bei Index depth + 1, die Rekonstruktion bei -1."""
    n_layers = len(weights)
    outs = [Z]
    for i in range(n_layers):
        z = outs[-1] @ weights[i] + biases[i]
        outs.append(activate(activation, z) if _has_activation(i, depth, n_layers) else z)
    return outs


def loss_and_gradients(weights, biases, Z, activation, depth):
    """Mittlerer quadratischer Rekonstruktionsfehler (je Element) und seine Gradienten (Backpropagation von Hand)."""
    n, d = Z.shape
    outs = forward(weights, biases, Z, activation, depth)
    diff = outs[-1] - Z
    loss = float((diff ** 2).mean())
    delta = 2.0 * diff / (n * d)
    n_layers = len(weights)
    gw, gb = [None] * n_layers, [None] * n_layers
    for i in reversed(range(n_layers)):
        gw[i] = outs[i].T @ delta
        gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = delta @ weights[i].T
            if _has_activation(i - 1, depth, n_layers):
                delta = delta * activation_grad(activation, outs[i])
    return loss, gw, gb


def snapshot_epochs(n_epochs, count=12):
    grid = np.unique(np.round(np.geomspace(1, max(n_epochs, 2), count)).astype(int))
    return sorted(set(int(e) for e in grid) | {0, int(n_epochs)})


def quantile_threshold(errors, nu):
    """Das (1 - nu)-Quantil der Fehler (untere Interpolation: höchstens ein Anteil nu der Touren liegt strikt darüber)."""
    return float(np.quantile(errors, 1.0 - nu, method="higher"))


@dataclass(frozen=True)
class Model:
    weights: list
    biases: list
    activation: str
    depth: int
    width: int
    n_code: int
    mean: np.ndarray
    scale: np.ndarray
    Z: np.ndarray                    # standardisierte Trainingstouren
    errors: np.ndarray               # [n] Rekonstruktionsfehler der Trainingstouren (mittleres Fehlerquadrat je Merkmal)
    code: np.ndarray                 # [n, k] Engpass-Werte der Trainingstouren
    snapshots: dict                  # Epoche -> Fehler je Tour (Kopie), inkl. 0 und n_epochs
    code_snapshots: dict             # Epoche -> Kodes (Kopie)
    weight_snapshots: dict           # Epoche -> (Gewichte, Biases) (Kopien)
    loss_history: np.ndarray         # [n_epochs + 1] Trainingsverlust, Index k = nach k Schritten (0 = untrainiert, letzter = fertiges Netz); leer bei 0 Epochen
    threshold: float                 # (1 - nu)-Quantil der Trainingsfehler
    nu: float
    lr: float
    n_epochs: int
    seed: int

    @property
    def n(self):
        return len(self.Z)

    @property
    def n_parameters(self):
        return n_parameters(layer_sizes(self.Z.shape[1], self.depth, self.width, self.n_code))


def _errors_and_code(weights, biases, Z, activation, depth):
    outs = forward(weights, biases, Z, activation, depth)
    return ((outs[-1] - Z) ** 2).mean(axis=1), outs[depth + 1]


def fit_autoencoder(X, n_code=2, depth=1, width=16, activation="tanh", nu=0.1, lr=0.01, n_epochs=300, seed=0):
    """Trainiert den Autoencoder auf den Kennzahlen X [n, p] (werden intern standardisiert)."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation in {ACTIVATIONS}")
    if depth < 0 or n_code < 1 or not 0 < nu <= 1:
        raise ValueError("depth >= 0, n_code >= 1, 0 < nu <= 1")
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    sizes = layer_sizes(Z.shape[1], depth, width, n_code)
    weights, biases = init_parameters(sizes, activation, seed)
    params = weights + biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n_layers = len(weights)
    wanted = set(snapshot_epochs(n_epochs))
    loss_history = np.zeros(n_epochs)
    e0, c0 = _errors_and_code(weights, biases, Z, activation, depth)
    snapshots = {0: e0.copy()} if 0 in wanted else {}
    code_snapshots = {0: c0.copy()} if 0 in wanted else {}
    weight_snapshots = {0: ([w.copy() for w in weights], [b.copy() for b in biases])} if 0 in wanted else {}
    for epoch in range(1, n_epochs + 1):
        loss, gw, gb = loss_and_gradients(weights, biases, Z, activation, depth)
        loss_history[epoch - 1] = loss
        lr_t = lr * np.sqrt(1.0 - BETA2 ** epoch) / (1.0 - BETA1 ** epoch)
        for k, g in enumerate(gw + gb):
            m[k] = BETA1 * m[k] + (1 - BETA1) * g
            v[k] = BETA2 * v[k] + (1 - BETA2) * g ** 2
            params[k] -= lr_t * m[k] / (np.sqrt(v[k]) + ADAM_EPS)
        if epoch in wanted:
            e, c = _errors_and_code(weights, biases, Z, activation, depth)
            snapshots[epoch] = e.copy()
            code_snapshots[epoch] = c.copy()
            weight_snapshots[epoch] = ([w.copy() for w in weights], [b.copy() for b in biases])
    errors, code = _errors_and_code(weights, biases, Z, activation, depth)
    if n_epochs > 0:
        loss_history = np.append(loss_history, float(errors.mean()))                             # Verlust des fertigen Netzes: Index k = Verlust nach k Schritten (die Schleife misst vor dem Schritt)
    return Model(weights=weights, biases=biases, activation=activation, depth=depth, width=width, n_code=n_code, mean=mean, scale=scale, Z=Z, errors=errors, code=code, snapshots=snapshots,
                 code_snapshots=code_snapshots, weight_snapshots=weight_snapshots, loss_history=loss_history, threshold=quantile_threshold(errors, nu), nu=float(nu), lr=float(lr), n_epochs=int(n_epochs), seed=int(seed))


def reconstruct(model, X_new=None, weights=None, biases=None):
    """Rekonstruktion (in z-Einheiten) und Fehler je Merkmal [m, p] für neue Touren (Trainingstouren, wenn X_new fehlt); mit `weights`/`biases` aus einer Momentaufnahme."""
    Zn = model.Z if X_new is None else (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    ws = model.weights if weights is None else weights
    bs = model.biases if biases is None else biases
    outs = forward(ws, bs, Zn, model.activation, model.depth)
    return Zn, outs[-1], (outs[-1] - Zn) ** 2


def score(model, X_new=None):
    """Anomalie-Wert: Rekonstruktionsfehler (mittleres Fehlerquadrat über die Merkmale, z-Einheiten)."""
    if X_new is None:
        return model.errors
    return reconstruct(model, X_new)[2].mean(axis=1)


def encode(model, X_new):
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    return forward(model.weights, model.biases, Zn, model.activation, model.depth)[model.depth + 1]


# --- PCA-Residuum ------------------------------------------------------------------------------------------------------------------------


def pca_axes(Z, n_components):
    """Erste Hauptachsen (Zeilen) der (zentrierten) z-Werte."""
    Zc = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    return vt[:n_components]


def pca_residual(X, n_components):
    """PCA-Residuum je Tour: mittleres Fehlerquadrat der Rekonstruktion aus den ersten k Hauptkomponenten (Kennzahlen standardisiert)."""
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0, ddof=1)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    axes = pca_axes(Z, n_components)
    Zc = Z - Z.mean(axis=0)
    rec = Zc @ axes.T @ axes
    return ((Zc - rec) ** 2).mean(axis=1)


def principal_angles(A, B):
    """Hauptwinkel (Grad) zwischen den von den ZEILEN von A und B aufgespannten Unterräumen; 0 = gleicher Unterraum."""
    qa, _ = np.linalg.qr(A.T)
    qb, _ = np.linalg.qr(B.T)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def decoder_subspace(weights):
    """Zeilenraum des linearen Decoders eines linearen Autoencoders (Tiefe 0): [k, p]."""
    return weights[1]


def linear_angle_curve(model, pca_ax):
    """Größter Hauptwinkel (Grad) zwischen dem Decoder-Unterraum des linearen Autoencoders und dem PCA-Unterraum an den Momentaufnahmen. Nur Tiefe 0."""
    if model.depth != 0:
        raise ValueError("nur für den linearen Autoencoder (Tiefe 0)")
    return {e: float(principal_angles(decoder_subspace(ws), pca_ax).max()) for e, (ws, bs) in sorted(model.weight_snapshots.items())}
