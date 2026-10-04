"""Plotly-Visualisierungen der Autoencoder-Anomalie-Demo: Touren in der Ebene der größten Streuung, Rekonstruktion einzelner Touren, Trainingsverlauf, Fehler je Merkmal, Rekonstruktionsfehler in der Ebene zweier Merkmale,
Wert-Histogramm mit Schwelle, ROC-Kurven, Kennzahlen-Balken, Sweeps und die Experimente (Szenarien, Epochenverlauf, Netz-Seeds, Engpass, linearer Autoencoder gegen PCA, Schwelle, Kosten). Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import aeano_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"ae": PINK, "iforest": PURPLE, "robust": "#5f6b7a", "dsvdd": BLUE, "pca": GREEN, "untrained": "#e58ab4", "ocsvm": TEAL, "lof": ORANGE, "ecod": "#b5a642"}
DETECTOR_NAMES = {"ae": "Autoencoder", "iforest": "Isolation Forest", "robust": "robust (MCD)", "dsvdd": "Deep SVDD", "pca": "PCA-Residuum", "untrained": "Autoencoder ohne Training", "ocsvm": "One-Class SVM",
                  "lof": "LOF", "ecod": "ECOD (zweiseitig)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke", "decorrelated": "Korrelationsbruch"}
DETECTORS = ("ae", "iforest", "robust", "dsvdd")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_score_hist(values, anomaly, thr, title="Rekonstruktionsfehler"):
    """Histogramm des Rekonstruktionsfehlers (Schwelle: (1 − ν)-Quantil der Trainingsfehler) (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle."""
    hi = float(max(values.max(), thr * 1.05))
    lo = float(min(values.min(), thr * 0.95))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 50.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.55, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.9, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=title), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_rank_compare(ae_values, robust_values, anomaly):
    """Rang des Autoencoders gegen Rang der Wurzel (robuste Mahalanobis-Abstände) je Tour: Touren, die der Autoencoder und die Wurzel gleich einschätzen, liegen auf der Diagonalen; Sonderfahrten als Rauten."""
    n = len(ae_values)
    re_ = np.argsort(np.argsort(ae_values)) / (n - 1)
    rr = np.argsort(np.argsort(robust_values)) / (n - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr[~anomaly], y=re_[~anomaly], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.5), name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rr[anomaly], y=re_[anomaly], mode="markers", marker=dict(size=9, color=RED, symbol="diamond"), name="Sonderfahrten", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Rang bei der Wurzel (robuste Mahalanobis-Abstände)", range=[-0.02, 1.02]),
                      yaxis=dict(title="Rang beim Autoencoder", range=[-0.02, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


# --- Schritte: Netz, Training, Fehler -----------------------------------------------------------------------------------------------------


def build_reconstruction(names, entries, epochs_labels=None):
    """Originalwerte (z-Einheiten) und Rekonstruktion einer normalen Tour und einer Sonderfahrt je Merkmal; `entries` = [(Titel, Original [p], Rekonstruktion [p], Fehler)] in beliebiger Zahl von Feldern."""
    fig = make_subplots(rows=1, cols=len(entries), subplot_titles=[e[0] for e in entries], horizontal_spacing=0.07)
    labels = [names[k] if k < len(names) else f"Merkmal {k + 1}" for k in range(len(entries[0][1]))]
    for col, (title, orig, rec, err) in enumerate(entries, start=1):
        fig.add_trace(go.Bar(x=labels, y=orig, name="Original", marker_color=BLUE, showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Bar(x=labels, y=rec, name="Rekonstruktion", marker_color=ORANGE, showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
    fig.update_xaxes(tickangle=-60, showticklabels=len(labels) <= 14)
    fig.update_yaxes(title="z-Wert", col=1)
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_feature_errors(names, err_normal, err_anomaly, top=12):
    """Fehlerquadrat je Merkmal: Mittel der normalen Touren (blau) gegen eine Sonderfahrt (rot); die größten `top` Merkmale der Sonderfahrt."""
    order = np.argsort(-err_anomaly)[:top]
    labels = [names[k] if k < len(names) else f"Merkmal {k + 1}" for k in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[float(err_normal[k]) for k in order], name="normale Touren (Mittel)", marker_color=BLUE, hoverinfo="skip"))
    fig.add_trace(go.Bar(x=labels, y=[float(err_anomaly[k]) for k in order], name="Sonderfahrt", marker_color=RED, hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=90), yaxis=dict(title="Fehlerquadrat"), legend=dict(orientation="h", y=-0.55), xaxis=dict(tickangle=-40))
    return lock_axes(fig)


def build_code_space(code, anomaly, epoch):
    """Die Engpass-Werte der Trainingstouren (die ersten zwei Kode-Werte; bei einem Kode-Wert gegen die Tour-Nummer) nach `epoch` Epochen."""
    normal = ~anomaly
    y = code[:, 1] if code.shape[1] > 1 else np.arange(len(code)) / len(code)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=code[normal, 0], y=y[normal], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=code[anomaly, 0], y=y[anomaly], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), name="Sonderfahrten", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Kode-Wert 1", zeroline=False), yaxis=dict(title="Kode-Wert 2" if code.shape[1] > 1 else "(nur ein Kode-Wert)", zeroline=False),
                      legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_training(loss_history, epochs, aucs):
    """Links der mittlere Rekonstruktionsfehler über die Epochen (logarithmische Achsen), rechts die AUC des Rekonstruktionsfehlers an den Momentaufnahmen (Epoche 0 = untrainiert)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trainingsverlust", "AUC des Rekonstruktionsfehlers"), horizontal_spacing=0.12)
    if len(loss_history):
        fig.add_trace(go.Scatter(x=np.arange(1, len(loss_history) + 1), y=loss_history, mode="lines", line=dict(color=BLUE, width=3), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[str(e) for e in epochs], y=aucs, mode="lines+markers", line=dict(color=PINK, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=2)
    fig.update_xaxes(title="Epoche", type="log", col=1)
    fig.update_yaxes(title="mittleres Fehlerquadrat", type="log", col=1)
    fig.update_xaxes(title="Epoche", type="category", col=2)
    fig.update_yaxes(title="AUC", range=[0, 1.05], col=2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_boundary(xs, ys, E, S, X2, anomaly, outside, names):
    """Rekonstruktionsfehler in der Ebene zweier Rohmerkmale (der Autoencoder wird dafür auf nur diese zwei Merkmale mit Engpass 1 neu trainiert): links Fehler minus Schwelle (schwarze Linie: Schwelle; blau = gut
    rekonstruiert), rechts zum Vergleich der Score des Isolation Forest (Linie bei 0.5). Umkreiste Punkte liegen über der Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Autoencoder: Fehler − Schwelle", "Isolation Forest: Score"), horizontal_spacing=0.10)
    lim = float(np.quantile(np.abs(E), 0.9)) or 1e-9
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=-E, colorscale="RdBu", zmid=0.0, zmin=-lim, zmax=lim, showscale=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Contour(x=xs, y=ys, z=E, contours=dict(start=0.0, end=0.0, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="YlOrRd", zmin=0.3, zmax=0.8, showscale=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Contour(x=xs, y=ys, z=S, contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=2)
    normal = ~anomaly
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=X2[normal, 0], y=X2[normal, 1], mode="markers", marker=dict(size=5, color=BLUE, line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=X2[anomaly, 0], y=X2[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond", line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
    if outside.any():
        fig.add_trace(go.Scatter(x=X2[outside, 0], y=X2[outside, 1], mode="markers", marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=1.5, color="black")), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.update_xaxes(title=names[0])
    fig.update_yaxes(title=names[1], col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------


def build_epoch_curves(rows):
    """AUC des Rekonstruktionsfehlers über die Trainings-Epochen für mehrere Szenarien (Epoche 0 = untrainiertes Netz); Mittel über Datensätze und Netz-Seeds; die gepunktete Linie bei 0.5 ist Raten."""
    palette = [PINK, ORANGE, PURPLE, TEAL, GREEN, BLUE, RED, GRAY]
    fig = go.Figure()
    for k, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=[str(e) for e in r["epochs"]], y=r["auc"], mode="lines+markers", name=r["scenario"], line=dict(color=palette[k % len(palette)], width=3), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Trainings-Epochen (0 = untrainiert)", type="category"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_seeds(rows):
    """AUC über die Epochenzahl: Mittel (Balken), Streuung über alle Datensatz-Netz-Paare (Fehlerbalken), Minimum und Maximum (Rauten)."""
    xs = [str(r["epochs"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["auc"] for r in rows], error_y=dict(type="data", array=[r["auc_std"] for r in rows], visible=True), marker_color=PINK, name="Mittel ± Streuung", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_min"] for r in rows], mode="markers", marker=dict(symbol="diamond", size=9, color="black"), name="Minimum", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_max"] for r in rows], mode="markers", marker=dict(symbol="diamond-open", size=9, color="black"), name="Maximum", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Trainings-Epochen", type="category"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_bottleneck(rows):
    """AUC über den Engpass für drei Arten der Anomalien: Autoencoder (durchgezogen) und PCA-Residuum (gestrichelt) bei gleichem Engpass; die gepunktete Linie bei 0.5 ist Raten."""
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    colors = {"verstreut": PINK, "Korrelationsbruch": TEAL, "gekrümmt": ORANGE}
    fig = go.Figure()
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        xs = [str(r["n_code"]) for r in sub]
        color = colors.get(kind, GRAY)
        fig.add_trace(go.Scatter(x=xs, y=[r["auc"] for r in sub], mode="lines+markers", name=f"Autoencoder, {kind}", line=dict(color=color, width=3), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=[r["pca_auc"] for r in sub], mode="lines+markers", name=f"PCA-Residuum, {kind}", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Engpass (Kode-Werte)", type="category"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_linear(result):
    """Linearer Autoencoder (Tiefe 0, Engpass 2): links der größte Hauptwinkel zwischen Decoder-Unterraum und PCA-Unterraum (Grad) über die Epochen, rechts die AUC des Rekonstruktionsfehlers gegen die des PCA-Residuums (Linie)."""
    epochs = [str(e) for e in result["epochs"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Winkel zum PCA-Unterraum", "AUC"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=epochs, y=result["angle"], mode="lines+markers", line=dict(color=PINK, width=3), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=result["auc"], mode="lines+markers", line=dict(color=PINK, width=3), name="linearer Autoencoder", hoverinfo="skip"), row=1, col=2)
    fig.add_hline(y=result["pca_auc"], line=dict(color="black", dash="dash"), row=1, col=2, annotation_text="PCA-Residuum", annotation_position="top left")
    fig.update_xaxes(title="Epochen", type="category")
    fig.update_yaxes(title="Grad", col=1)
    fig.update_yaxes(range=[0, 1.05], col=2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_scenarios(rows):
    """AUC des Autoencoders, des untrainierten Netzes, des PCA-Residuums, Deep SVDD, Isolation Forest und Wurzel in den Szenarien (links) und F1 mit bekanntem Anteil (rechts); die Linie bei 0.5 ist Raten."""
    labels = [r["scenario"] for r in rows]
    names = ("ae", "untrained", "pca", "dsvdd", "iforest", "robust")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC", "F1 mit bekanntem Anteil"), horizontal_spacing=0.10)
    for det in names:
        for col, key in ((1, f"{det}_auc"), (2, f"{det}_oracle_f1")):
            y = [r[key] for r in rows]
            fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], showlegend=col == 1, text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7),
                                 hoverinfo="skip"), row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    fig.update_yaxes(range=[0, 1.15])
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(height=560, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_cutoff(rows):
    """Autoencoder: F1 (durchgezogen), Recall (gepunktet), Fehlalarmrate (gestrichelt) und AUC (grau) über ν bei der Schwelle (1 − ν)-Quantil."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["ae_f1"] for r in rows], mode="lines+markers", name="F1", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ae_recall"] for r in rows], mode="lines+markers", name="Recall", line=dict(color=PINK, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ae_false_alarm"] for r in rows], mode="lines+markers", name="Fehlalarmrate", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ae_auc"] for r in rows], mode="lines+markers", name="AUC", line=dict(color=GRAY, width=2), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="ν", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Training bzw. Anpassen und Wert je Tour) des Autoencoders (Epochen der Einstellung und 1000), des PCA-Residuums, Deep SVDD, One-Class SVM, LOF und Isolation Forest über die Tourenzahl bei 12 und 52 Merkmalen."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("12 Merkmale", "52 Merkmale (40 Rauschmerkmale)"), horizontal_spacing=0.10)
    for col, p in ((1, 12), (2, 52)):
        rows = [t for t in times if t["p"] == p]
        for key, color, name in (("ae", PINK, "Autoencoder (Epochen der Einstellung)"), ("ae_1000", "#7b1044", "Autoencoder (1000 Epochen)"), ("pca", GREEN, "PCA-Residuum"), ("dsvdd", BLUE, "Deep SVDD"),
                                 ("ocsvm", TEAL, "One-Class SVM"), ("lof", ORANGE, "LOF"), ("iforest", PURPLE, "Isolation Forest")):
            y = [t[key] for t in rows]
            fig.add_trace(go.Bar(x=[f"n = {t['n']}" for t in rows], y=y, name=name, marker_color=color, showlegend=col == 1, text=[f"{v:.3f}" for v in y], textposition="outside", textfont=dict(size=7), hoverinfo="skip"),
                          row=1, col=col)
    fig.update_yaxes(title="Sekunden", col=1)
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)
