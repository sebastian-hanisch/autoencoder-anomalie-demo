"""Autoencoder - der Rekonstruktionsfehler als Anomalie-Wert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Autoencoder als Anomalie-Detektor - und lässt stattdessen das Beispiel wachsen.
Neuntes und letztes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: der unabhängige Rekonstruktions-Ast direkt nach der Wurzel. Gemessen wird, was das bringt - und was nicht. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import aeano_algorithm as ae
import aeano_constants as C
from aeano_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    Settings,
    analyse_for,
    bottleneck_table,
    cost_table,
    epoch_curves,
    linear_table,
    roc_auc,
    roc_curve,
    scenario_table,
    score_maps,
    seed_table,
    sweep,
    threshold_table,
    verdict,
)
from aeano_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    randomize_net_seed,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from aeano_visualization import (
    build_boundary,
    build_bottleneck,
    build_code_space,
    build_costs,
    build_cutoff,
    build_epoch_curves,
    build_feature_errors,
    build_features,
    build_linear,
    build_method_bars,
    build_rank_compare,
    build_reconstruction,
    build_roc,
    build_scatter,
    build_scenarios,
    build_score_hist,
    build_seeds,
    build_sweep,
    build_training,
    build_wrong_share,
    projection,
)

st.set_page_config(page_title="Autoencoder – Sebastian Hanisch", layout="wide")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _maps(data_params, settings, i, j):
    return score_maps(_analysis(data_params, settings).ds.X[:, [i, j]], settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _scenarios(settings):
    return scenario_table(settings)


@st.cache_data(show_spinner=False)
def _curves(settings):
    return epoch_curves(settings)


@st.cache_data(show_spinner=False)
def _seeds(base, settings):
    return seed_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _bottleneck(base, settings):
    return bottleneck_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _linear(base, settings):
    return linear_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("🪞 Autoencoder – der Rekonstruktionsfehler als Anomalie-Wert")
st.markdown(
    """
Die anderen Äste dieser Linie fragen, **wie weit** eine Tour von den Normalen entfernt ist (Ellipse, Dichte, Kernel, Kugel) oder **wie leicht sie sich isolieren** lässt. Der **Autoencoder** fragt etwas anderes: er lernt, die normalen Touren durch einen **Engpass** aus wenigen Zahlen hindurch möglichst genau wieder aufzubauen -
und eine Tour ist um so auffälliger, je **schlechter** ihr das gelingt. Anomalien sind demnach das, was **außerhalb der gelernten Struktur** liegt. Das ist genau die Stärke - und die Schwäche: eine Anomalie, die **auf** der gelernten Struktur liegt, wird mitrekonstruiert und fällt nicht auf.
Die Demo misst, wo das trägt und wo nicht.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - neuntes und letztes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, Isolation Forest, die robuste Schätzung der Wurzel und Deep SVDD sind wortgleich aus den Vorgänger-Demos übernommen (die klassische Schätzung entfällt, damit die Balken lesbar bleiben); als Bezug laufen das **untrainierte** Netz, das **PCA-Residuum**, "
    "die One-Class SVM, LOF und ECOD mit. Die Linie hat keinen Konvergenzpunkt; der Autoencoder schließt sie ab - seine Nachbarn sind alle anderen Äste (alle gebaut)."
)

with st.expander("So funktioniert der Autoencoder als Detektor", expanded=True):
    st.markdown(
        """
1. **Netz.** Ein Encoder bildet die $p$ standardisierten Kennzahlen einer Tour auf $k$ Werte ab (den **Engpass**, linear), ein Decoder baut daraus wieder $p$ Werte auf; dazwischen verdeckte Schichten mit tanh oder ReLU (Tiefe 0 = rein linear).
2. **Training.** Minimiere den mittleren quadrierten Rekonstruktionsfehler $\\tfrac1{np}\\sum_i\\lVert \\hat x_i - x_i\\rVert^2$ auf den Touren (Adam, Vollbatch; die Demo rechnet die Gradienten selbst und prüft sie gegen Finite Differenzen). Es gibt keine Etiketten.
3. **Wert und Schwelle.** Der Wert einer Tour ist ihr Rekonstruktionsfehler; als Schwelle dient das $(1-\\nu)$-Quantil der Trainingsfehler - der Anteil ν ist wie bei One-Class SVM und Deep SVDD eingebaut.
4. **Verbindung zur PCA.** Ein **lineares** Autoencoder (Tiefe 0) lernt denselben Unterraum wie die ersten $k$ Hauptkomponenten (Baldi und Hornik); sein Fehler ist dann das **PCA-Residuum**. Nichtlineare Schichten lassen gebogene Strukturen zu.

Was die Methode **voraussetzt**: dass Anomalien die gelernte Struktur **verlassen**, dass Engpass und Netz nicht zu groß (dann rekonstruiert es alles) und nicht zu klein (dann rekonstruiert es nichts) sind - und dass man rechtzeitig aufhört zu trainieren.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. AUC des Autoencoders (Voreinstellung) bei 20 / 30 / 50 / 100 / 200 / 400 / 600 Touren: 0.00 / 0.15 / 0.29 / 0.43 / 0.57 / 0.70 / 0.72 - bei wenigen Touren **unter Raten**: der mittlere Fehler der Sonderfahrten ist 0.001 gegenüber 0.020 bei den normalen Touren (20 Touren; 50 Touren 0.019 gegenüber 0.027), sie werden besser rekonstruiert. "
             "Das untrainierte Netz hat 0.99-1.00, Deep SVDD 0.72 / 0.82 / 0.91 / 0.94 / 0.94 / 0.92 / 0.93.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). AUC des Autoencoders bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: 0.94 / 0.67 / 0.68 / 0.68 / 0.79 / 0.86; das PCA-Residuum bei Engpass 2 hat 0.76-0.80 (rechnerabhängig) / 0.51 / 0.47 / 0.49 / 0.47 / 0.46. "
             "Bei zwei Merkmalen ist Engpass 2 kein Engpass mehr.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. AUC des Autoencoders bei 0 / 5 / 10 / 20 / 30 / 40 Rauschmerkmalen: 0.68 / 0.48 / 0.45 / 0.34 / 0.26 / 0.25 - **unter Raten**: bei 40 Rauschmerkmalen kommen 96 % des Fehlers der normalen Touren aus den Rauschspalten (95 % bei den Sonderfahrten), der Fehler der 12 Kennzahlen ist bei Sonderfahrten und Normalen gleich (0.108 gegenüber 0.112). "
             "Sie werden gleich gut rekonstruiert. Mit Engpass 1: 0.88 / 0.86 / 0.82 / 0.75 / 0.71 / 0.68. Isolation Forest 0.99, robust 0.88, das untrainierte Netz 0.995.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. AUC des Autoencoders bei 1 / 2 / 3 Betriebsarten 0.68 / 0.72 / 0.73 (untrainiert 1.00 / 1.00 / 0.99), robust 1.00 / 0.95 / 0.85.",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 steigt die AUC des Autoencoders von 0.68 auf 0.96 / 0.99 / 0.99 / 0.98 und der F1 von 0.26 auf 0.77 / 0.82 / 0.84 / 0.83: die Anomalien verlassen jetzt die gerade Ebene; "
             "das PCA-Residuum (Engpass 2) hat 0.49 / 0.99 / 0.98 / 0.99 / 0.99.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. AUC des Autoencoders bei 0 / 0.25 / 0.5 / 0.75 / 1.0: 0.95 / 0.68 / 0.55 / 0.50 / 0.48, der F1 0.73 / 0.26 / 0.14 / 0.10 / 0.11. Ohne Rauschen liegen die Normalen genau in der Ebene und jede Abweichung fällt auf; "
             "mit Rauschen ist die Ebene 'dick' und die Sonderfahrten in ihr verschwinden.",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. AUC des Autoencoders bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % verstreuten Anomalien: 0.85 / 0.80 / 0.68 / 0.66 / 0.68 / 0.60 / 0.63; mit Engpass 1 0.92 / 0.92 / 0.88 / 0.89 / 0.89 / 0.86 / 0.87 - das Training lernt die Anomalien mit, das untrainierte Netz bleibt bei 1.00.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung des Faktorraums - **in der Ebene** der Normalen. Dichte Gruppe (30 %): AUC des Autoencoders 0.50 (untrainiert 0.95, PCA-Residuum 0.48, Deep SVDD 0.18, Isolation Forest 0.70, One-Class SVM 0.74, ECOD 0.84). "
             "In der Lücke: 0.49 bei 2 wie bei 3 Betriebsarten (untrainiert 0.06 und 0.01). Korrelationsbruch: die Anomalien **verlassen die Ebene** - AUC **1.00**, F1 0.98 (PCA-Residuum 1.00, robust 1.00, LOF 0.99, Deep SVDD 0.92, One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56).",
    )
    if kind not in ("gap", "decorrelated"):
        seed_widget("strength_slider")
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**Autoencoder**")
    n_code = st.slider(
        "Engpass (Kode-Werte)", *bounds("n_code_slider"), key="n_code_slider",
        help="Wie viele Zahlen das Netz zwischen Encoder und Decoder hat. Das Szenario hat **2 latente Faktoren**. AUC bei Engpass 1 / 2 / 3 / 4 / 6 / 8 (verstreute Anomalien): 0.88 / 0.68 / 0.64 / 0.78 / 0.81 / 0.88, F1 0.69 / 0.26 / 0.22 / 0.38 / 0.43 / 0.56; "
             "das PCA-Residuum 0.91 / 0.49 / 0.49 / 0.50 / 0.50 / 0.52. Beim Korrelationsbruch 0.88 / 1.00 / 1.00 / 1.00 / 1.00 / 0.98. Ein Engpass unter der latenten Dimension hilft gegen Anomalien in der Ebene und schadet beim Korrelationsbruch.",
    )
    depth = st.slider(
        "Tiefe (verdeckte Schichten je Seite)", *bounds("depth_slider"), key="depth_slider",
        help="Verdeckte Schichten im Encoder und im Decoder (0 = linearer Autoencoder = PCA). AUC bei 0 / 1 / 2 / 3: 0.49 / 0.68 / 0.71 / 0.76 (F1 0.12 / 0.26 / 0.27 / 0.37) - der lineare Autoencoder ist nach 300 Epochen das PCA-Residuum (0.49), tiefere Netze sind noch nicht so weit konvergiert.",
    )
    width = st.select_slider(
        "Breite der verdeckten Schichten", options=list(C.WIDTH_OPTIONS), key="width_select",
        help="Neuronen je verdeckter Schicht. AUC bei Breite 4 / 8 / 16 / 32 / 64: 0.89 / 0.85 / 0.68 / 0.73 / 0.52 - je mehr Spielraum, desto besser rekonstruiert das Netz auch die Anomalien.",
    )
    epochs = st.slider(
        "Epochen", *bounds("epochs_slider"), key="epochs_slider", step=10,
        help="Trainingsschritte (Vollbatch, Adam, Lernrate 0.01). **0 = das untrainierte Netz.** Bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen: AUC 1.00 / 1.00 / 0.97 / 0.87 / 0.68 / 0.56 (nach 1000 Epochen je nach Rechner bis 0.67), F1 an der Schwelle 0.98 / 0.97 / 0.86 / 0.58 / 0.26 / 0.14 (nach 1000 Epochen je nach Rechner bis 0.33): "
             "je besser das Netz rekonstruiert, desto schlechter die Rangfolge (bei ReLU 0.96 / 1.00 / 1.00 / 0.89 / 0.77 / 0.59).",
    )
    nu = st.slider(
        "ν (Anteil oberhalb der Schwelle)", *bounds("nu_slider"), key="nu_slider", step=0.01, format="%.2f",
        help="Die Schwelle ist das (1 − ν)-Quantil der Trainingsfehler: es werden etwa ν der Touren markiert. Die Rangfolge (AUC 0.68) ändert sich nicht. Bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5: F1 0.08 / 0.17 / 0.25 / 0.26 / 0.26 / 0.25 / 0.24, "
             "Recall 0.04 / 0.10 / 0.18 / 0.26 / 0.38 / 0.49 / 0.70, Fehlalarmrate 0.3 % / 0.7 % / 3.2 % / 7.9 % / 17.6 % / 27.5 % / 47.4 %.",
    )
    activation = st.selectbox(
        "Aktivierung", C.ACTIVATIONS, key="activation_select",
        help="tanh oder ReLU (Tiefe 0: ohne Wirkung). Bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen ist die AUC mit ReLU 0.96 / 1.00 / 1.00 / 0.89 / 0.77 / 0.59, mit tanh 1.00 / 1.00 / 0.97 / 0.87 / 0.68 / 0.56.",
    )
    net_seed = st.number_input(
        "Netz-Seed (Initialisierung)", *bounds("net_seed_input"), key="net_seed_input", step=1,
        help="Zufall der Anfangsgewichte, getrennt vom Seed der Touren. Über 10 Netz-Seeds und 5 Datensätze ist die AUC nach 0 / 30 / 100 / 300 Epochen 1.00 ± 0.00 / 0.98 ± 0.01 / 0.88 ± 0.04 / 0.73 ± 0.07 (schlechtestes Netz 1.00 / 0.95 / 0.77 / 0.61).",
    )
    st.button("🎲 Neues Netz würfeln", width="stretch", on_click=randomize_net_seed, help="Würfelt einen neuen Netz-Seed: dieselben Touren, andere Anfangsgewichte.")
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: Autoencoder über dem (1 − ν)-Quantil der Trainingsfehler, Deep SVDD über R², Isolation Forest über Score 0.5, robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        seed_widget("quantile_slider")
        quantile = st.slider(
            "Schwelle: χ²-Quantil (robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei der robusten Schätzung als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90.",
        )
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
    else:
        seed_widget("share_slider")
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % beim Autoencoder (Voreinstellung) "
                 "0.18 / 0.25 / 0.27 / 0.25 / 0.25, beim Isolation Forest 0.33 / 0.67 / 0.97 / 0.67 / 0.40, bei robust 0.33 / 0.67 / 0.90 / 0.66 / 0.40.",
        )
        st.session_state["_share_kept"] = share
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
    seed = st.number_input("Zufalls-Seed (Touren)", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "n_code_slider": int(n_code), "depth_slider": int(depth), "width_select": int(width),
    "epochs_slider": int(epochs), "nu_slider": float(nu), "activation_select": activation, "net_seed_input": int(net_seed), "threshold_kind_select": threshold_kind, "quantile_slider": float(quantile),
    "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(n_code=int(n_code), depth=int(depth), width=int(width), epochs=int(epochs), nu=float(round(nu, 2)), activation=activation, net_seed=int(net_seed), threshold_kind=threshold_kind,
                    quantile=float(quantile), share=int(share))
with st.spinner("Trainiere das Netz..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ae_s, if_s, rb_s, dv_s = (a.scores[d] for d in ("ae", "iforest", "robust", "dsvdd"))
un_s, pc_s = a.scores["untrained"], a.scores["pca"]
n_anom = int(ds.anomaly.sum())
p_total = a.p_total
net = a.net
values = a.values["ae"]
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- Autoencoder in Aktion ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Autoencoder in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Netz", 3: "3 · Training", 4: "4 · Fehler", 5: "5 · Ergebnis"}
if "aeano_step" not in st.session_state or st.session_state.get("aeano_step_owner") != data_key:
    st.session_state["aeano_step"] = 1
    st.session_state["aeano_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="aeano_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
normal_idx = np.flatnonzero(~ds.anomaly)
anom_idx = np.flatnonzero(ds.anomaly)
i_anom = int(anom_idx[np.argsort(values[anom_idx])[len(anom_idx) // 2]])                          # eine typische Sonderfahrt: die mit dem mittleren Fehler
i_norm = int(normal_idx[np.argmin(np.abs(P[normal_idx]).sum(axis=1))])                             # eine normale Tour nahe der Mitte
if ds.kind == "decorrelated" and ds.p >= 2:
    corr = np.abs(np.corrcoef(ds.X[~ds.anomaly][:, : ds.p].T))
    np.fill_diagonal(corr, 0.0)
    pair_i, pair_j = (int(v) for v in np.unravel_index(np.argmax(corr), corr.shape))
else:
    pair_i, pair_j = 0, min(1, ds.p - 1)
names_all = ds.names
snap_auc = {e: roc_auc(err, ds.anomaly) for e, err in sorted(net.snapshots.items())}
w0, b0 = net.weight_snapshots[0]


def _rec(idx, weights=None, biases=None):
    Zn, rec, err = ae.reconstruct(net, weights=weights, biases=biases)
    return Zn[idx], rec[idx], err[idx]


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {names_all[pair_i]} gegen {names_all[pair_j]}** (Einheiten wie gemessen)" + (" - das am stärksten korrelierte Paar der Normalen" if ds.kind == "decorrelated" else ""))
            c2.plotly_chart(build_features(ds.X, ds.anomaly, names_all, pair_i, pair_j), width="stretch", key="step_features")
        elif current_step == 2:
            zn_n, rec_n, err_n = _rec(i_norm, w0, b0)
            zn_a, rec_a, err_a = _rec(i_anom, w0, b0)
            st.markdown("**Das zufällig initialisierte Netz rekonstruiert eine normale Tour und eine Sonderfahrt** (z-Werte je Merkmal; blau = Original, orange = Rekonstruktion)")
            st.plotly_chart(build_reconstruction(names_all, [(f"normale Tour (Fehler {err_n.mean():.3f})", zn_n, rec_n, err_n), (f"Sonderfahrt (Fehler {err_a.mean():.3f})", zn_a, rec_a, err_a)]), width="stretch", key="step_recon0")
        elif current_step == 3:
            st.markdown("**Verlauf des Trainings** (links: Verlust, rechts: AUC des Rekonstruktionsfehlers - Epoche 0 ist das untrainierte Netz)")
            st.plotly_chart(build_training(net.loss_history, list(snap_auc), list(snap_auc.values())), width="stretch", key="step_training")
            xs_, ys_, E_, S_, outside_ = _maps(data_params, settings, pair_i, pair_j)
            st.markdown(f"**Der Rekonstruktionsfehler in der Ebene {names_all[pair_i]} gegen {names_all[pair_j]}** (das Netz wird dafür auf nur diese zwei Merkmale mit Engpass 1 neu trainiert; schwarze Linie: Schwelle, umkreist: Touren darüber)")
            st.plotly_chart(build_boundary(xs_, ys_, E_, S_, ds.X[:, [pair_i, pair_j]], ds.anomaly, outside_, [names_all[pair_i], names_all[pair_j]]), width="stretch", key="step_boundary")
        elif current_step == 4:
            zn_n, rec_n, err_n = _rec(i_norm)
            zn_a, rec_a, err_a = _rec(i_anom)
            st.markdown(f"**Nach {net.n_epochs} Epochen: dieselben zwei Touren** (blau = Original, orange = Rekonstruktion)")
            st.plotly_chart(build_reconstruction(names_all, [(f"normale Tour (Fehler {err_n.mean():.3f})", zn_n, rec_n, err_n), (f"Sonderfahrt (Fehler {err_a.mean():.3f})", zn_a, rec_a, err_a)]), width="stretch", key="step_recon")
            c1, c2 = st.columns(2)
            c1.markdown("**Fehler je Merkmal** (Sonderfahrt gegen Mittel der normalen Touren)")
            c1.plotly_chart(build_feature_errors(names_all, ae.reconstruct(net)[2][~ds.anomaly].mean(axis=0), err_a), width="stretch", key="step_feature_errors")
            c2.markdown("**Rekonstruktionsfehler je Tour** (Schwelle gestrichelt)")
            c2.plotly_chart(build_score_hist(values, ds.anomaly, net.threshold), width="stretch", key="step_score_hist")
            st.markdown("**Autoencoder gegen die Wurzel**: Rang jeder Tour bei beiden")
            st.plotly_chart(build_rank_compare(values, a.values["robust"], ds.anomaly), width="stretch", key="step_rank_compare")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("ae", "iforest", "robust", "dsvdd")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {p_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Das Netz bekommt alle {p_total} standardisierten Kennzahlen, hat {net.n_parameters:,} Parameter ({net.depth} verdeckte Schichten je Seite der Breite {net.width}, Engpass {net.n_code}).".replace(",", ".")
               + (" Beim Korrelationsbruch haben die Anomalien in jedem Merkmal normale Werte - nur die Kombination stimmt nicht (sie verlassen die Ebene der Normalen)." if ds.kind == "decorrelated" else
                  " Die Sonderfahrten sind Verschiebungen der zwei latenten Faktoren - sie liegen in derselben Ebene wie die Normalen." if ds.kind == "scattered" else ""))
elif step == 2:
    st.caption(f"Vor dem Training: Der Fehler der normalen Tour ist {_rec(i_norm, w0, b0)[2].mean():.3f}, der der Sonderfahrt {_rec(i_anom, w0, b0)[2].mean():.3f}; als Detektor hat dieses Netz AUC **{un_s['auc']:.2f}** und F1 {un_s['f1']:.2f}. "
               "Ein zufälliges Netz gibt fast nichts wieder (die Rekonstruktion ist fast null): der Fehler ist im Wesentlichen die Länge der Tour, also ihr Abstand von der Mitte - das ist schon ein guter Detektor für weit entfernte Touren.")
elif step == 3:
    st.caption((f"Nach {net.n_epochs} Epochen: AUC {snap_auc[net.n_epochs]:.2f} (untrainiert {snap_auc[0]:.2f}); Trainingsverlust {net.loss_history[-1]:.4f} (Start {net.loss_history[0]:.4f}). "
                "Der Verlust fällt immer - die Rangfolge nicht: je besser das Netz rekonstruiert, desto besser rekonstruiert es auch die Anomalien, die in der gelernten Struktur liegen.") if net.n_epochs > 0 else
               "0 Epochen gewählt: es wird nicht trainiert, der Fehler ist der des zufälligen Netzes.")
elif step == 4:
    st.caption(f"Mittlerer Fehler der normalen Touren {values[~ds.anomaly].mean():.3f}, der Sonderfahrten {values[ds.anomaly].mean():.3f}; die Schwelle {net.threshold:.3f} markiert {ae_s['n_flagged']} von {ds.n} Touren. "
               "Die Fehler je Merkmal zeigen, welche Kennzahlen das Netz an der Sonderfahrt nicht wiedergeben kann.")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Detektoren gefunden haben – Autoencoder gegen Isolation Forest gegen Wurzel gegen Deep SVDD")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: Autoencoder über dem (1 − ν)-Quantil der Trainingsfehler, Isolation Forest über Score 0.5, robust über dem χ²-Quantil, Deep SVDD über R²). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (Autoencoder)", f"{ae_s['auc']:.2f}", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m1.caption(f"untrainiert {un_s['auc']:.2f} · PCA-Residuum {pc_s['auc']:.2f} · Isolation Forest {if_s['auc']:.2f} · robust {rb_s['auc']:.2f} · Deep SVDD {dv_s['auc']:.2f}")
m2.metric("Recall (Autoencoder)", _pct(ae_s["recall"]), help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m2.caption(f"untrainiert {_pct(un_s['recall'])} · PCA-Residuum {_pct(pc_s['recall'])} · Isolation Forest {_pct(if_s['recall'])} · robust {_pct(rb_s['recall'])} · Deep SVDD {_pct(dv_s['recall'])}")
m3.metric("Fehlalarmrate (Autoencoder)", f"{ae_s['false_alarm']:.1%}", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m3.caption(f"untrainiert {un_s['false_alarm']:.1%} · PCA-Residuum {pc_s['false_alarm']:.1%} · Isolation Forest {if_s['false_alarm']:.1%} · robust {rb_s['false_alarm']:.1%} · Deep SVDD {dv_s['false_alarm']:.1%}")
m4.metric("F1 (Autoencoder)", f"{ae_s['f1']:.2f}", help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter die F1 der anderen Detektoren und die des Autoencoders, wenn der wahre Anteil bekannt wäre.")
m4.caption(f"untrainiert {un_s['f1']:.2f} · PCA-Residuum {pc_s['f1']:.2f} · Isolation Forest {if_s['f1']:.2f} · robust {rb_s['f1']:.2f} · Deep SVDD {dv_s['f1']:.2f} · Autoencoder mit bekanntem Anteil {a.oracle_f1['ae']:.2f}")

_t = vd
if code == "ae_wins":
    st.success(f"✅ Der Autoencoder trifft: AUC {_t['ae_auc']:.2f} (F1 {_t['ae_f1']:.2f}, Recall {_pct(_t['ae_recall'])}, Fehlalarmrate {_t['ae_false_alarm']:.1%}) gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor "
               f"(Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, Deep SVDD {_t['dsvdd_auc']:.2f}; F1 {_t['best_other_f1']:.2f}). Die Anomalien verlassen hier die gelernte Struktur (PCA-Residuum {_t['pca_auc']:.2f}, untrainiert {_t['untrained_auc']:.2f}).")
elif code == "comparable":
    st.success(f"✅ Der Autoencoder ist ebenbürtig: AUC {_t['ae_auc']:.2f} (untrainiert {_t['untrained_auc']:.2f}, PCA-Residuum {_t['pca_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, Deep SVDD {_t['dsvdd_auc']:.2f}); "
               f"bei der Schwelle F1 {_t['ae_f1']:.2f} (Isolation Forest {_t['iforest_f1']:.2f}, robust {_t['robust_f1']:.2f}, Deep SVDD {_t['dsvdd_f1']:.2f}), mit bekanntem Anteil {_t['oracle_ae']:.2f}.")
elif code == "training_hurts":
    st.warning(f"⚠️ Das Training schadet: nach {_t['epochs']} Epochen ist die AUC {_t['ae_auc']:.2f}, das **untrainierte** Netz hat {_t['untrained_auc']:.2f} (F1 {_t['ae_f1']:.2f} gegen {_t['untrained_f1']:.2f}). "
               f"Das PCA-Residuum bei Engpass {_t['n_code']} hat {_t['pca_auc']:.2f}: je besser das Netz die Ebene lernt, desto besser rekonstruiert es auch die Anomalien, die in ihr liegen. Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}. "
               "Weniger Epochen (oder gar keine) wären besser gewesen - aber ohne Etiketten weiß man das nicht.")
elif code == "blind":
    st.warning(f"⚠️ Der Autoencoder ist blind: AUC {_t['ae_auc']:.2f} (untrainiert {_t['untrained_auc']:.2f}, PCA-Residuum {_t['pca_auc']:.2f}), während Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f} und Deep SVDD {_t['dsvdd_auc']:.2f} erreichen. "
               "Die Anomalien liegen in der Struktur, die das Netz gelernt hat (oder werden bei zu vielen Rauschspalten und zu wenigen Touren vom Fehler der übrigen Merkmale überdeckt) - sie werden mitrekonstruiert.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['ae_auc']:.2f} beim Autoencoder (untrainiert {_t['untrained_auc']:.2f}), {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['robust_auc']:.2f} robust, {_t['dsvdd_auc']:.2f} bei Deep SVDD: keiner findet sie zuverlässig. "
               "Die Lücke liegt in der Ebene der Normalen; das Netz lernt sie mit.")
elif code == "others_win":
    st.warning(f"⚠️ Ein anderer Detektor ist besser: AUC Autoencoder {_t['ae_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, Deep SVDD {_t['dsvdd_auc']:.2f}; an der Schwelle F1 {_t['ae_f1']:.2f} gegen {_t['best_other_f1']:.2f}.")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['ae_auc']:.2f}), aber die Schwelle markiert nur ν = {_t['nu']:.2f} der Touren: F1 {_t['ae_f1']:.2f} gegen {_t['oracle_ae']:.2f} mit bekanntem Anteil.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"), ("Fehlalarmrate", "false_alarm", "{:.3f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "Autoencoder": [r[2].format(ae_s[r[1]]) for r in rows], "untrainiert": [r[2].format(un_s[r[1]]) for r in rows], "PCA-Residuum": [r[2].format(pc_s[r[1]]) for r in rows],
              "Isolation Forest": [r[2].format(if_s[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rb_s[r[1]]) for r in rows], "Deep SVDD": [r[2].format(dv_s[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Parameter", "Bewertung", "mittlerer Wert (normal / Anomalie)"],
              "Autoencoder": [f"{a.seconds['ae'] * 1000:.0f} ms", f"{net.n_parameters:,} Gewichte".replace(",", "."), f"Rekonstruktionsfehler ({net.n_epochs} Epochen)", f"{values[~ds.anomaly].mean():.3f} / {values[ds.anomaly].mean():.3f}"],
              "PCA-Residuum": [f"{a.seconds['pca'] * 1000:.1f} ms", f"Engpass {net.n_code}", "Abstand zum Hauptraum (geschlossene Form)", f"{a.values['pca'][~ds.anomaly].mean():.3f} / {a.values['pca'][ds.anomaly].mean():.3f}"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} Bäume × ψ = {a.forest_if.psi}", "Pfadlänge in zufälligen Bäumen", f"{a.values['iforest'][~ds.anomaly].mean():.2f} / {a.values['iforest'][ds.anomaly].mean():.2f}"]})
    st.caption("Der Autoencoder ist bei festem Seed deterministisch, der Zufall steckt in den Anfangsgewichten (Netz-Seed). Die klassische Schätzung der Wurzel entfällt in dieser Demo; das untrainierte Netz ist dasselbe Netz vor dem ersten Schritt, das PCA-Residuum die geschlossene Form des linearen Falls.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Daten und Reglern ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind in ("gap", "decorrelated") and k in ("strength",)) or (kind == "gap" and k == "n_modes") or (threshold_kind == "share" and k == "quantile") or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination), "strength": float(strength),
           "n_code": int(n_code), "depth": int(depth), "width": int(width), "epochs": int(epochs), "nu": float(settings.nu), "quantile": float(quantile), "share": int(share)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, SWEEP_VALUES[sweep_param])
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben), Netz-Seed wie in der Seitenleiste; alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Bei den Reglern des Netzes ändern sich nur die Linien des Autoencoders.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Das Training schadet – außer, wo die Struktur erst gelernt werden muss")
if st.button("AUC über die Epochen in acht Szenarien berechnen (dauert etwa 30 Sekunden)", key="curves_start"):
    st.session_state["curves_on"] = True
if st.session_state.get("curves_on"):
    with st.spinner("Trainiere 8 Szenarien × 5 Aufnahmen × 3 Netze..."):
        cv = _curves(settings)
    st.plotly_chart(build_epoch_curves(cv), width="stretch", key="curves_chart")
    st.caption("AUC des Rekonstruktionsfehlers an Momentaufnahmen des Trainings (600 Epochen); Epoche 0 = untrainiertes Netz. Mittel über 5 feste Datensätze und 3 Netz-Seeds. **Standardfall**: 1.00 → 0.97 (33 Epochen) → 0.87 (105) → 0.64 (600); "
               "**gekrümmter Normalbereich**: 1.00 → 0.96; **45 % verstreut**: 1.00 → 0.79 (18) → 0.65 (33) → 0.56 (600); **40 Rauschmerkmale**: 0.99 → 0.84 (33) → 0.59 (59) → 0.21 (600); **dichte Gruppe (30 %)**: 0.96 → 0.37 (18 Epochen) → 0.49. "
               "Nur wo die Struktur erst gelernt werden muss, hilft das Training: **Korrelationsbruch** 0.62 → 0.97 (33) → 1.00 (105), **Lücke** 0.03 / 0.00 → 0.52 / 0.49 (33 Epochen) - nie über Raten hinaus.")

st.markdown("---")

st.subheader("🔬 Der Engpass: kleiner als die latente Dimension?")
if st.button("Engpass 1-8 für drei Arten der Anomalien berechnen (dauert etwa 30 Sekunden)", key="bottleneck_start"):
    st.session_state["bottleneck_on"] = True
if st.session_state.get("bottleneck_on"):
    with st.spinner("Trainiere 3 Arten × 6 Engpässe × 5 Aufnahmen..."):
        bt = _bottleneck(tuple(kv for kv in base_data if kv[0] not in ("kind", "curvature")), settings)
    st.plotly_chart(build_bottleneck(bt), width="stretch", key="bottleneck_chart")
    st.caption("AUC bei Engpass 1 / 2 / 3 / 4 / 6 / 8 (Mittel über 5 feste Datensätze; das Szenario hat 2 latente Faktoren). **Verstreute Anomalien** (in der Ebene): Autoencoder 0.88 / 0.68 / 0.64 / 0.78 / 0.81 / 0.88, PCA-Residuum 0.91 / 0.49 / 0.49 / 0.50 / 0.50 / 0.52 - "
               "das PCA-Residuum sieht sie erst bei einem Engpass **unter** 2; der Autoencoder sieht sie bei Engpass 1 und bei Engpass 8 (0.88), dazwischen schwächer (0.64-0.81). **Korrelationsbruch** (verlässt die Ebene): 0.88 / 1.00 / 1.00 / 1.00 / 1.00 / 0.98 (PCA-Residuum 0.82 / 1.00 / 1.00 / 1.00 / 1.00 / 0.97). **Gekrümmt**: 1.00 / 0.98 / 0.97 / 0.94 / 0.97 / 0.98 (PCA-Residuum 1.00 / 0.99 / 0.97 / 0.86 / 0.72 / 0.60). "
               "Die beste Wahl des Engpasses hängt von der Art der Anomalien ab - und die kennt man vorher nicht.")

st.markdown("---")

st.subheader("🔬 Der lineare Autoencoder lernt die PCA")
if st.button("Tiefe 0, Engpass 2, 1000 Epochen berechnen (dauert etwa 15 Sekunden)", key="linear_start"):
    st.session_state["linear_on"] = True
if st.session_state.get("linear_on"):
    with st.spinner("Trainiere 5 lineare Autoencoder über 1000 Epochen..."):
        li = _linear(base_data, settings)
    st.plotly_chart(build_linear(li), width="stretch", key="linear_chart")
    st.caption("Der größte Hauptwinkel zwischen dem Decoder-Unterraum und dem PCA-Unterraum fällt von 83° (Epoche 0) über 27° (81 Epochen) und 1° (152) auf 0.03° (1000): der lineare Autoencoder findet **denselben Unterraum** wie die PCA (nicht dieselben Achsen). "
               "Die AUC des Rekonstruktionsfehlers folgt: 1.00 bis 23 Epochen (der Unterraum liegt noch beliebig), 0.97 nach 81, 0.52 nach 152 und 0.49 danach - das PCA-Residuum hat 0.49. Die AUC verschwindet, sobald das Netz die Struktur gelernt hat.")

st.markdown("---")

st.subheader("🔬 Der Zufall der Initialisierung")
if st.button("10 Netz-Seeds vergleichen (dauert etwa 20 Sekunden)", key="seeds_start"):
    st.session_state["seeds_on"] = True
if st.session_state.get("seeds_on"):
    with st.spinner("Trainiere 4 Epochenzahlen × 5 Aufnahmen × 10 Netze..."):
        sd = _seeds(base_data, settings)
    st.plotly_chart(build_seeds(sd), width="stretch", key="seeds_chart")
    st.caption("AUC über alle 50 Paare aus Datensatz und Netz-Seed (Einstellungen der Seitenleiste bis auf die Epochen). Nach 0 / 30 / 100 / 300 Epochen: Mittel 1.00 / 0.98 / 0.88 / 0.73, Streuung 0.00 / 0.01 / 0.04 / 0.07, schlechtestes Netz 1.00 / 0.95 / 0.77 / 0.61, bestes 1.00 / 1.00 / 0.94 / 0.92; "
               "F1 an der Schwelle 0.97 ± 0.02 / 0.93 ± 0.04 / 0.63 ± 0.07 / 0.34 ± 0.11. Je länger trainiert wird, desto stärker hängt das Ergebnis an den Anfangsgewichten.")

st.markdown("---")

st.subheader("🔬 Wo der Autoencoder trifft und wo nicht: acht Szenarien")
if st.button("Die Detektoren in acht Szenarien vergleichen (dauert etwa 40 Sekunden)", key="scenarios_start"):
    st.session_state["scenarios_on"] = True
if st.session_state.get("scenarios_on"):
    with st.spinner("Rechne 8 Szenarien × 5 Aufnahmen..."):
        sc_rows = _scenarios(settings)
    st.plotly_chart(build_scenarios(sc_rows), width="stretch", key="scenarios_chart")
    st.caption("Mittel über 5 feste Datensätze, Autoencoder mit den Reglern der Seitenleiste (Standard: Engpass 2, eine verdeckte Schicht der Breite 16, 300 Epochen), dazu das untrainierte Netz, das PCA-Residuum (gleicher Engpass), Deep SVDD, Isolation Forest und Wurzel. "
               "**Standardfall**: 0.68 (untrainiert 1.00, PCA-Residuum 0.49). **Gekrümmt**: 0.98 (PCA-Residuum 0.99). **Dichte Gruppe (30 %)**: 0.50 (untrainiert 0.95, Deep SVDD 0.18). **Lücke**: 0.49 und 0.49. "
               "**Korrelationsbruch**: **1.00** (untrainiert 0.70, PCA-Residuum 1.00, robust 1.00, Deep SVDD 0.92, Isolation Forest 0.80) - hier trifft er. **45 % verstreut**: 0.63 (untrainiert 1.00). **40 Rauschmerkmale**: 0.25 (untrainiert 0.995, Isolation Forest 0.99, robust 0.88). "
               "Rechts der F1 mit bekanntem Anteil: im Standardfall 0.27 (untrainiert 0.98, PCA-Residuum 0.11), beim Korrelationsbruch 0.99.")

st.markdown("---")

st.subheader("🔬 Die Schwelle")
if st.button("ν über die Schwelle vergleichen (dauert etwa 20 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 7 ν-Werte × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Autoencoder: F1, Recall, Fehlalarmrate und AUC je ν (Schwelle = (1 − ν)-Quantil)**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.caption("Links (Standardfall, Voreinstellung): F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gleich 0.08 / 0.17 / 0.25 / 0.26 / 0.26 / 0.25 / 0.24, die AUC bleibt bei 0.68: die Schwelle markiert **so viele Touren, wie ν vorgibt**; wo die Rangfolge schlecht ist, hilft auch das richtige ν nicht. "
               "Rechts: bei einem angenommenen Anteil von 5 / 10 / 20 % ist der F1 des Autoencoders 0.25 / 0.27 / 0.25, des Isolation Forest 0.67 / 0.97 / 0.67, von Deep SVDD 0.59 / 0.63 / 0.55.")

st.markdown("---")

st.subheader("🔬 Kosten")
if st.button("Rechenzeit berechnen (dauert etwa 30 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(tuple(kv for kv in base_data if kv[0] not in ("n", "n_noise")), settings)
    st.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    st.caption("Mittel über 3 feste Datensätze (Zeiten rechnerabhängig, nur die Größenordnungen zählen). Das Training mit 300 Epochen braucht bei 100 / 300 / 600 Touren und 12 Merkmalen 36 / 58 / 89 ms, mit 1000 Epochen 0.12 / 0.19 / 0.30 s - "
               "das Eineinhalb- bis Zweieinhalbfache von Deep SVDD (14 / 29 / 54 ms) und weit mehr als das PCA-Residuum (0.1-0.2 ms, geschlossene Form). Der Isolation Forest braucht 67 / 130 / 133 ms.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Anomalien verlassen die gelernte Struktur** | Liegen sie **auf** ihr, werden sie mitrekonstruiert: im Standardfall (Sonderfahrten in der Faktorebene) AUC **0.68**, das PCA-Residuum bei Engpass 2 **0.49**; der lineare Autoencoder ist nach 300 Epochen das PCA-Residuum (0.49). Beim Korrelationsbruch, der die Ebene verlässt, AUC 1.00. | Isolation Forest, One-Class SVM, Wurzel |
| **Das Training verbessert den Detektor** | Es verschlechtert ihn: AUC **1.00 vor dem Training**, 0.87 nach 100 und 0.56-0.67 nach 1000 Epochen, je nach Rechner (F1 0.98 → 0.58 → 0.14-0.33); bei 45 % Anomalien 1.00 → 0.56, bei dichter Gruppe 0.96 → 0.37. Nur beim Korrelationsbruch und in der Lücke hilft es (0.62 → 1.00; 0.03 → 0.52). | Deep SVDD (dasselbe Muster) |
| **Man weiß, wann man aufhören muss** | Die beste Epochenzahl hängt vom Szenario ab (Standardfall 0, Korrelationsbruch etwa 100, Lücke etwa 30) und ist ohne Etiketten nicht zu erkennen; über 10 Netz-Seeds streut die AUC nach 300 Epochen mit 0.07 (schlechtestes Netz 0.61). | (keiner) |
| **Der Engpass passt** | Engpass 1 (unter der latenten Dimension 2) sieht Anomalien in der Ebene (AUC 0.88), verliert beim Korrelationsbruch (0.88 statt 1.00); Engpass 2 ist umgekehrt. Beim Engpass 8 werden die Anomalien wieder erkannt (0.88; PCA-Residuum 0.52). | Feature Bagging, Wurzel |
| **Alle Merkmale sind rekonstruierbar** | 40 Rauschmerkmale: AUC **0.25** (unter Raten) - 96 % des Fehlers kommen aus den nicht rekonstruierbaren Rauschspalten. Mit Engpass 1: 0.68; das untrainierte Netz 0.995. | Isolation Forest, ECOD |
| **Genug Touren für das Netz** | Bei 20 / 30 / 50 Touren AUC 0.00 / 0.15 / 0.29 (**unter Raten**): die Sonderfahrten werden besser rekonstruiert als die normalen Touren (mittlerer Fehler bei 20 Touren 0.001 gegenüber 0.020). | Isolation Forest, LOF |
| **Wenig Anomalien im Training** | Das Training lernt sie mit: AUC 0.85 bei 2 %, 0.68 bei 10 %, 0.63 bei 45 % (Engpass 1: 0.92 / 0.88 / 0.87). | ECOD, Isolation Forest |
| **Die Schwelle ist ein Anteil** | Sie markiert ν der Touren (F1 0.08 bei ν = 0.01, 0.26 bei 0.1, 0.24 bei 0.5); mit schlechter Rangfolge hilft auch das richtige ν nicht. | ECOD (Fisher-Quantil) |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie - alle gebaut: die Wurzel Elliptic Envelope, LOF, Feature Bagging, Isolation Forest, Extended IF, ECOD, One-Class SVM und Deep SVDD. "
    "Keiner ist überlegen: der Autoencoder ist das Verfahren für Anomalien, die die gelernte Struktur **verlassen** (Korrelationsbruch: 1.00) - und blind für die, die auf ihr liegen (Standardfall: 0.68, PCA-Residuum 0.49)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Netz.** Encoder $z = f_{\text{enc}}(x)$ mit $x \in \mathbb R^p$ (standardisiert), $z \in \mathbb R^k$ (Engpass, linear), Decoder $\hat x = f_{\text{dec}}(z)$ mit linearem Ausgang; dazwischen $\sigma$ = tanh oder ReLU. Mit Tiefe 0: $\hat x = W_2 W_1 x + b$.

**Ziel.** $\min_W\ \tfrac{1}{np}\sum_{i=1}^n \lVert \hat x_i - x_i\rVert^2$, Backpropagation von Hand (gegen Finite Differenzen geprüft), Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, Lernrate $10^{-2}$, Vollbatch).

**Wert und Schwelle.** $s(x) = \tfrac1p\lVert \hat x - x\rVert^2$; Anomalie, wenn $s(x)$ über dem $(1-\nu)$-Quantil der Trainingsfehler liegt (höchstens ein Anteil ν der Trainingstouren liegt strikt darüber).

**PCA-Residuum.** Mit den ersten $k$ Hauptachsen $V_k$ der zentrierten $x$: $s_{\text{PCA}}(x) = \tfrac1p\lVert x - V_k^\top V_k x\rVert^2$; sein Mittel ist $\tfrac{1}{np}\sum_{j>k}\sigma_j^2$ (Eckart-Young).
Ein **lineares** Autoencoder mit MSE-Verlust hat im Optimum denselben Bildraum des Decoders wie $V_k$ (Baldi und Hornik): nicht dieselben Achsen, sondern denselben Unterraum; die Demo misst ihn über die Hauptwinkel.

**Grenzen.** (1) Anomalien auf der gelernten Struktur werden mitrekonstruiert. (2) Das Ziel ist Rekonstruktion, nicht Anomalie-Erkennung: je besser das Netz, desto besser rekonstruiert es auch die Anomalien. (3) Der Engpass und die Epochen sind Regler, ohne Etiketten nicht zu wählen. (4) Die Schwelle markiert ν der Touren.

Implementiert in `aeano_algorithm.py` (Netz, Backpropagation, Adam, PCA-Residuum, Hauptwinkel), `aeano_dsvdd.py` (Deep SVDD des Vergleichs), `aeano_ocsvm.py`, `aeano_lof.py`, `aeano_isolation_forest.py` und `aeano_ee_algorithm.py` (One-Class SVM, LOF, Isolation Forest, χ², MCD; wortgleich aus den Vorgängern),
`aeano_ecod.py` (ECOD als Kontrast), `aeano_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen, Korrelationsbruch), `aeano_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Anomalie-Erkennung: Ellipse bis Autoencoder](https://sebastianhanisch.net/konzepte-anomalie-erkennung.html)."
)
