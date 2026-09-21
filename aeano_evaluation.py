"""Auswertung der Autoencoder-Anomalie-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für den Autoencoder,
den Isolation Forest, die robuste Schätzung der Wurzel und Deep SVDD (dazu das PCA-Residuum, das untrainierte Netz, One-Class SVM, LOF und ECOD als Bezug), Sweeps, Experimente auf Abruf (Szenarien, Epochenverlauf,
Netz-Seeds, Engpass, lineares Autoencoder gegen PCA, Schwelle, Kosten) und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import aeano_algorithm as ae
import aeano_dsvdd as dv
import aeano_ocsvm as svm
import aeano_ecod as ecod
import aeano_lof as lof
import aeano_isolation_forest as isf
import aeano_ee_algorithm as alg
import aeano_constants as C
import aeano_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1 (One-Class SVM und LOF rechnen mit Abständen; Isolation Forest und robuste Schätzung brauchen das nicht; Autoencoder und Deep SVDD skalieren selbst)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


@dataclass(frozen=True)
class Settings:
    n_code: int = C.DEFAULT_CODE                        # Engpass des Autoencoders
    depth: int = C.DEFAULT_DEPTH
    width: int = C.DEFAULT_WIDTH
    epochs: int = C.DEFAULT_EPOCHS
    nu: float = C.DEFAULT_NU                            # Schwelle = (1 - nu)-Quantil der Trainingsfehler
    activation: str = C.DEFAULT_ACTIVATION
    net_seed: int = C.DEFAULT_NET_SEED                  # Initialisierung des Netzes (getrennt vom Seed der Aufnahme)
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": (1 - nu)-Quantil (Autoencoder und Deep SVDD), Score 0.5 (Isolation Forest), chi²-Quantil (robust); "share": der erwartete Anteil für alle vier
    quantile: float = C.DEFAULT_QUANTILE                # chi²-Quantil der Wurzel
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed des Isolation Forest und der MCD-Starts


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("ae", "iforest", "robust", "dsvdd")
EXTRAS = ("pca", "untrained", "ocsvm", "lof", "ecod")              # Bezugsdetektoren, die im Urteil und in den Balken nicht auftauchen
DETECTOR_NAMES = {"ae": "Autoencoder", "iforest": "Isolation Forest", "robust": "robust (MCD)", "dsvdd": "Deep SVDD", "pca": "PCA-Residuum", "untrained": "Autoencoder ohne Training",
                  "ocsvm": "One-Class SVM", "lof": "LOF", "ecod": "ECOD (zweiseitig)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")
IF_TREES, IF_PSI, IF_CUTOFF, LOF_CUTOFF, LOF_K = C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_CUTOFF_IF, C.DEFAULT_CUTOFF_LOF, C.DEFAULT_LOF_K


def lof_k(n):
    """k des LOF im Vergleich: 20, höchstens n / 2 (k nahe n macht LOF unbrauchbar)."""
    return int(max(3, min(LOF_K, n // 2)))


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


def fit_net(X, settings, n_epochs=None, n_code=None):
    """Autoencoder mit den Reglern der Einstellungen (n_epochs und n_code überschreibbar)."""
    return ae.fit_autoencoder(X, n_code=settings.n_code if n_code is None else n_code, depth=settings.depth, width=settings.width, activation=settings.activation, nu=settings.nu, lr=C.DEFAULT_LR,
                              n_epochs=settings.epochs if n_epochs is None else n_epochs, seed=settings.net_seed)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    p_total: int                  # Zahl der Merkmale einschließlich Rauschmerkmale
    net: ae.Model                 # das trainierte Autoencoder
    dsvdd_model: dv.Model         # Deep SVDD des Vergleichs (Voreinstellung)
    ocsvm_fit: svm.Fit
    forest_if: isf.Forest
    values: dict                  # Detektor -> Anomalie-Wert je Tour (dazu die Bezugsdetektoren)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den Reglern des Netzes: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, robust, net, dsvdd_model):
    """Markierung und Schwellenwert je Detektor: Standard = (1 - nu)-Quantil der Trainingsfehler (Autoencoder), R^2 (Deep SVDD), Score 0.5 (Isolation Forest), chi²-Quantil (robust),
    sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["ae"] = net.threshold
        thr["dsvdd"] = dsvdd_model.r2
        thr["iforest"] = IF_CUTOFF
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    p_total = X.shape[1]
    t0 = time.perf_counter()
    net = fit_net(X, settings)
    secs["ae"] = time.perf_counter() - t0
    Z = standardise(X)
    t0 = time.perf_counter()
    dsvdd_model = dv.fit_deep_svdd(X, depth=C.DSVDD_DEPTH, width=C.DSVDD_WIDTH, n_out=C.DSVDD_N_OUT, nu=settings.nu, lr=C.DSVDD_LR, n_epochs=C.DSVDD_EPOCHS, seed=settings.net_seed)
    secs["dsvdd"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    oc = svm.fit_ocsvm(Z, C.OCSVM_NU, C.OCSVM_GAMMA_FACTOR * svm.gamma_scale(Z))
    secs["ocsvm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lof_values = lof.fit_lof(Z, lof_k(len(X))).lof
    secs["lof"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, IF_TREES, min(IF_PSI, len(X)), settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    t0 = time.perf_counter()
    pca_values = ae.pca_residual(X, min(settings.n_code, p_total))
    secs["pca"] = time.perf_counter() - t0
    values = {"ae": ae.score(net), "iforest": isf.score_from_paths(paths_if, forest_if.psi), "robust": robust.d2, "dsvdd": dv.score(dsvdd_model), "pca": pca_values, "untrained": net.snapshots[0],
              "ocsvm": svm.score(oc), "lof": lof_values, "ecod": ecod.score(X, "twosided")}
    flags, thr = _thresholds(values, settings, robust, net, dsvdd_model)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    extra_flags = {"pca": pca_values > ae.quantile_threshold(pca_values, settings.nu), "untrained": net.snapshots[0] > ae.quantile_threshold(net.snapshots[0], settings.nu), "ocsvm": values["ocsvm"] > C.FLAG_EPS,
                   "lof": lof_values > LOF_CUTOFF, "ecod": np.zeros(len(X), bool)}
    for d in EXTRAS:
        m = flag_metrics(extra_flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=None)
        scores[d] = m
    scores["ae"].update(loss=float(net.loss_history[-1]) if len(net.loss_history) else float("nan"))
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in DETECTORS + EXTRAS}
    return Analysis(ds, settings, params, p_total, net, dsvdd_model, oc, forest_if, values, classical, robust, scores, flags, oracle, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


def score_maps(X2, settings=Settings(), grid=80, pad=0.5):
    """Rekonstruktionsfehler-Karte des Autoencoders und Isolation-Forest-Score auf einem Raster für zweidimensionale Kennzahlen: (xs, ys, E [grid, grid], S [grid, grid], außerhalb [n]).
    Das Autoencoder wird dafür auf nur diesen zwei Merkmalen mit Engpass 1 neu trainiert (bei zwei Merkmalen ist 1 der einzige Engpass); E = Fehler - Schwelle (negativ: gut rekonstruiert)."""
    X2 = np.asarray(X2, dtype=float)
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    net = fit_net(X2, settings, n_code=1)
    E = (ae.score(net, pts) - net.threshold).reshape(grid, grid)
    forest = isf.fit_forest(X2, IF_TREES, min(IF_PSI, len(X2)), settings.start)
    S = isf.score_from_paths(isf.path_lengths(forest, pts), forest.psi).reshape(grid, grid)
    return xs, ys, E, S, net.errors > net.threshold


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "n_code": (1, 2, 3, 4, 6, 8),
    "depth": (0, 1, 2, 3),
    "width": C.WIDTH_OPTIONS,
    "epochs": (0, 10, 30, 100, 300, 1000),
    "nu": (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "n_code": "Engpass (Kode-Werte)", "depth": "Tiefe des Netzes (verdeckte Schichten je Seite)",
                "width": "Breite des Netzes", "epochs": "Trainings-Epochen", "nu": "ν (Schwelle)", "quantile": "chi²-Quantil der Schwelle (robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("n_code", "depth", "width", "epochs", "nu", "quantile", "share")


def _record(a):
    names = DETECTORS + EXTRAS
    out = {f"{d}_{k}": a.scores[d][k] for d in names for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    for d in names:
        out[f"{d}_oracle_f1"] = a.oracle_f1[d]
    for d in DETECTORS + ("lof", "pca"):
        out[f"{d}_seconds"] = a.seconds[d]
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


SCENARIOS = (
    ("Standardfall", {}),
    ("gekrümmter Normalbereich", {"curvature": 1.0}),
    ("dichte Gruppe 30 %", {"kind": "cluster", "contamination": 30}),
    ("Lücke, 2 Betriebsarten", {"kind": "gap", "n_modes": 2}),
    ("Lücke, 3 Betriebsarten", {"kind": "gap", "n_modes": 3}),
    ("Korrelationsbruch", {"kind": "decorrelated"}),
    ("45 % verstreut", {"contamination": 45}),
    ("40 Rauschmerkmale", {"n_noise": 40}),
)


def scenario_table(settings=Settings(), scenarios=SCENARIOS, **base):
    """Die Detektoren und die Bezugsdetektoren in den Szenarien: AUC, F1 und F1 mit bekanntem Anteil (Mittel über die festen Sweep-Datensätze)."""
    rows = []
    for label, extra in scenarios:
        row = _mean_over_seeds(settings, **{**base, **extra})
        row["scenario"] = label
        rows.append(row)
    return rows


def epoch_curves(settings=Settings(), scenarios=SCENARIOS, net_seeds=(0, 1, 2), max_epochs=600, **base):
    """AUC des Rekonstruktionsfehlers über die Trainings-Epochen (Momentaufnahmen inkl. Epoche 0 = untrainiert), gemittelt über die festen Sweep-Datensätze und `net_seeds` Netze, je Szenario; dazu der Trainingsverlust.
    Rückgabe: [{scenario, epochs, auc, loss}]."""
    out = []
    for label, extra in scenarios:
        per, losses = [], []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base, **extra})
            for ns in net_seeds:
                net = fit_net(ds.X, Settings(**{**settings.__dict__, "net_seed": ns, "epochs": max_epochs}))
                per.append({e: roc_auc(err, ds.anomaly) for e, err in sorted(net.snapshots.items())})
                losses.append({e: (float(net.loss_history[e - 1]) if e > 0 else float("nan")) for e in sorted(net.snapshots)})
        epochs = sorted(per[0])
        aucs = np.array([[r[e] for e in epochs] for r in per])
        out.append({"scenario": label, "epochs": epochs, "auc": aucs.mean(axis=0).tolist(), "auc_std": aucs.std(axis=0).tolist(), "loss": [float(np.nanmean([l[e] for l in losses])) if e > 0 else float("nan") for e in epochs]})
    return out


def seed_table(settings=Settings(), epochs_list=(0, 30, 100, 300), net_seeds=tuple(range(10)), **base):
    """Streuung über die Netz-Seeds: je Epochenzahl AUC und F1 (an der Schwelle), gemittelt über die fünf Datensätze; Streuung, Minimum und Maximum über alle Datensatz-Netz-Paare."""
    rows = []
    for ep in epochs_list:
        aucs, f1s = [], []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
            for ns in net_seeds:
                net = fit_net(ds.X, Settings(**{**settings.__dict__, "net_seed": ns}), n_epochs=ep)
                aucs.append(roc_auc(ae.score(net), ds.anomaly))
                f1s.append(flag_metrics(ae.score(net) > net.threshold, ds.anomaly)["f1"])
        rows.append({"epochs": ep, "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs)), "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})
    return rows


BOTTLENECK_KINDS = (("verstreut", {}), ("Korrelationsbruch", {"kind": "decorrelated"}), ("gekrümmt", {"curvature": 1.0}))


def bottleneck_table(settings=Settings(), codes=(1, 2, 3, 4, 6, 8), kinds=BOTTLENECK_KINDS, **base):
    """Engpass × Art der Anomalien: AUC des Autoencoders und des PCA-Residuums bei gleichem Engpass (Mittel über die festen Sweep-Datensätze)."""
    rows = []
    for label, extra in kinds:
        for k in codes:
            aucs, pcas = [], []
            for seed in C.SWEEP_SEEDS:
                ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base, **extra})
                aucs.append(roc_auc(ae.score(fit_net(ds.X, settings, n_code=k)), ds.anomaly))
                pcas.append(roc_auc(ae.pca_residual(ds.X, k), ds.anomaly))
            rows.append({"kind": label, "n_code": k, "auc": float(np.mean(aucs)), "pca_auc": float(np.mean(pcas))})
    return rows


def linear_table(settings=Settings(), n_code=2, epochs=(0, 10, 30, 100, 300, 1000), **base):
    """Lineares Autoencoder (Tiefe 0) gegen PCA-Residuum: größter Hauptwinkel des Decoder-Unterraums zum PCA-Unterraum und AUC über die Epochen; das PCA-Residuum ist die Grenze, gegen die das Autoencoder konvergiert
    (Mittel über die festen Sweep-Datensätze; das Netz wird dafür 1000 Epochen trainiert)."""
    angles, aucs, pca_aucs = [], [], []
    for seed in C.SWEEP_SEEDS:
        ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
        net = fit_net(ds.X, Settings(**{**settings.__dict__, "depth": 0, "epochs": max(epochs)}), n_code=n_code)
        axes = ae.pca_axes(net.Z, n_code)
        curve = ae.linear_angle_curve(net, axes)
        angles.append({e: curve[e] for e in sorted(net.snapshots)})
        aucs.append({e: roc_auc(err, ds.anomaly) for e, err in sorted(net.snapshots.items())})
        pca_aucs.append(roc_auc(ae.pca_residual(ds.X, n_code), ds.anomaly))
    keys = sorted(angles[0])
    return {"epochs": keys, "angle": [float(np.mean([a[e] for a in angles])) for e in keys], "auc": [float(np.mean([a[e] for a in aucs])) for e in keys], "pca_auc": float(np.mean(pca_aucs))}


def cost_table(settings=Settings(), **base):
    """Rechenzeit des Autoencoders (Training mit den Epochen der Einstellung und mit 1000), des PCA-Residuums, Deep SVDD, One-Class SVM, LOF und Isolation Forest über die Tourenzahl bei 12 und bei 52 Merkmalen."""
    times = []
    for n_noise in (0, 40):
        for n in (100, 300, 600):
            a_rows = [_analyse_seed(seed, settings, {**base, "n": n, "n_noise": n_noise}) for seed in C.SWEEP_SEEDS[:3]]
            long_secs = []
            for seed in C.SWEEP_SEEDS[:3]:
                ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base, "n": n, "n_noise": n_noise})
                t0 = time.perf_counter()
                fit_net(ds.X, settings, n_epochs=1000)
                long_secs.append(time.perf_counter() - t0)
            times.append({"n": n, "p": 12 + n_noise, "ae": float(np.mean([a.seconds["ae"] for a in a_rows])), "ae_1000": float(np.mean(long_secs)),
                          **{d: float(np.mean([a.seconds[d] for a in a_rows])) for d in ("pca", "dsvdd", "ocsvm", "lof", "iforest")}})
    return {"times": times}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen über ν (Schwelle = (1 - ν)-Quantil der Trainingsfehler); (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil."""
    cut = sweep("nu", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    return {"cutoff": cut, "wrong_share": wrong}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Detektor als besser gilt
GAP_AUC = 0.8
BLIND_AUC = 0.6
F1_WIN_MARGIN = 0.10          # F1-Abstand an der Schwelle, ab dem der Autoencoder als besser gilt
GOOD_AUC = 0.8
THRESHOLD_F1_DROP = 0.15
TRAINING_LOSS = 0.05          # AUC-Verlust gegenüber dem untrainierten Netz, ab dem das Training als schädlich gilt


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke (keiner findet sie), blind (Autoencoder unter 0.6, ein anderer über 0.8), Training schadet (das untrainierte Netz ist besser), anderer Detektor besser, Autoencoder besser,
    falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    e = a.scores["ae"]
    others = {d: a.scores[d]["auc"] for d in ("iforest", "robust", "dsvdd")}
    best_other = max(others.values())
    best_other_f1 = max(a.scores[d]["f1"] for d in ("iforest", "robust", "dsvdd"))
    data = {"n": ds.n, "p": a.p_total, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()), "nu": a.settings.nu,
            "epochs": a.settings.epochs, "n_code": a.settings.n_code, "best_other_auc": best_other, "best_other_f1": best_other_f1, "untrained_auc": a.scores["untrained"]["auc"], "pca_auc": a.scores["pca"]["auc"],
            **{f"oracle_{d}": a.oracle_f1[d] for d in DETECTORS + EXTRAS}, **{f"{d}_{k}": v for d in DETECTORS + EXTRAS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and max(best_other, e["auc"]) < GAP_AUC:
        return "warning", "gap", data
    if e["auc"] < BLIND_AUC and best_other >= GOOD_AUC:
        return "warning", "blind", data
    if a.scores["untrained"]["auc"] - e["auc"] >= TRAINING_LOSS and a.settings.epochs > 0:
        return "warning", "training_hurts", data
    if best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "others_win", data
    if e["auc"] - best_other >= WIN_MARGIN or (e["f1"] - best_other_f1 >= F1_WIN_MARGIN and e["auc"] >= best_other - 0.02):
        return "success", "ae_wins", data
    if e["auc"] >= 0.95 and a.oracle_f1["ae"] - e["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
