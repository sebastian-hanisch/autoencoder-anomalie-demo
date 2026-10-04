"""Defaults, Regler-Grenzen und Presets für die Autoencoder-Anomalie-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario, Deep SVDD, One-Class SVM, LOF, Isolation Forest und Vergleichsschätzer aus den Vorgänger-Demos)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap", "decorrelated")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten", "decorrelated": "Korrelationsbruch (Randverteilungen normal)"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- Autoencoder und Vergleichsdetektoren --------------------------------------------------------------------------------------
DEFAULT_CODE, CODE_MIN, CODE_MAX = 2, 1, 8               # Engpass (Zahl der Kode-Werte); das Szenario hat zwei latente Faktoren
DEFAULT_DEPTH, DEPTH_MIN, DEPTH_MAX = 1, 0, 3
DEFAULT_WIDTH, WIDTH_OPTIONS = 16, (4, 8, 16, 32, 64)
DEFAULT_EPOCHS, EPOCHS_MIN, EPOCHS_MAX = 300, 0, 1000
DEFAULT_LR = 0.01
DEFAULT_ACTIVATION = "tanh"
ACTIVATIONS = ("tanh", "relu")
DEFAULT_NU = 0.10                                       # Schwelle = (1 - nu)-Quantil der Trainingsfehler (markiert höchstens den Anteil nu)
NU_MIN, NU_MAX = 0.01, 0.50
DEFAULT_NET_SEED = 0
# Vergleichsdetektoren
DSVDD_WIDTH, DSVDD_DEPTH, DSVDD_N_OUT, DSVDD_EPOCHS, DSVDD_LR = 32, 2, 8, 100, 0.001   # Deep SVDD wie in der Autoencoder-Anomalie-Demo (Voreinstellung)
OCSVM_NU = 0.10
OCSVM_GAMMA_FACTOR = 1.0
DEFAULT_TREES = 100                                     # Isolation Forest im Vergleich
DEFAULT_PSI = 256
DEFAULT_CUTOFF_IF = 0.5                                 # nominelle Score-Schwelle des Isolation Forest
DEFAULT_CUTOFF_LOF = 1.5                                # LOF-Schwelle im Vergleich
DEFAULT_LOF_K = 20                                      # LOF im Vergleich: k = 20 (höchstens n / 2)
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard ((1 − ν)-Quantil bzw. Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle vier)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45
SMO_TOL = 1e-6
FLAG_EPS = 1e-6
SMO_MAX_ITER = 100_000

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, n_code=DEFAULT_CODE, depth=DEFAULT_DEPTH, width=DEFAULT_WIDTH, epochs=DEFAULT_EPOCHS, nu=DEFAULT_NU, activation=DEFAULT_ACTIVATION,
                net_seed=DEFAULT_NET_SEED, threshold_kind=DEFAULT_THRESHOLD_KIND, quantile=DEFAULT_QUANTILE, share=DEFAULT_SHARE, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Ohne Training (0 Epochen)": _preset(epochs=0),
    "Engpass 1": _preset(n_code=1),
    "Linearer Autoencoder = PCA": _preset(depth=0, epochs=1000),
    "Korrelationsbruch": _preset(kind="decorrelated"),
    "40 Rauschmerkmale": _preset(n_noise=40),
    "Kleine Stichprobe (50 Touren)": _preset(n=50),
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien, ein Autoencoder mit einer verdeckten Schicht der Breite 16 und Engpass 2 (so viele Werte wie latente Faktoren im Szenario), 300 Epochen. "
                                    "AUC 0.68, F1 0.26 - das PCA-Residuum bei Engpass 2 hat 0.49 (Raten): die Anomalien liegen in derselben Ebene wie die Normalen und werden mitrekonstruiert. Das **untrainierte** Netz hat AUC 1.00 und F1 0.98, "
                                    "Isolation Forest 1.00 / 0.96, Deep SVDD 0.92 / 0.62.",
    "Ohne Training (0 Epochen)": "Im Mittel über fünf Aufnahmen: dasselbe Netz mit 0 Epochen - der Rekonstruktionsfehler des zufällig initialisierten Netzes. AUC 1.00 und F1 0.98 (Fehlalarmrate 0.1 %): das Netz gibt fast nichts wieder, der Fehler ist "
                                 "im Wesentlichen die Länge der Tour, also der Abstand vom Zentrum - so gut wie der Isolation Forest. Jede Epoche Training verschlechtert die Rangfolge (10 Epochen 1.00, 30 Epochen 0.97, 100 Epochen 0.87, 300 Epochen 0.68).",
    "Engpass 1": "Im Mittel über fünf Aufnahmen: Engpass 1 - weniger Werte als latente Faktoren (2). Das Netz kann die Ebene nicht abbilden, extreme Touren werden schlecht rekonstruiert: AUC 0.88, F1 0.69 (Engpass 2: 0.68 / 0.26); "
                 "das PCA-Residuum bei Engpass 1 hat 0.91. Beim Korrelationsbruch verliert der kleinere Engpass dagegen (0.88 statt 1.00).",
    "Linearer Autoencoder = PCA": "Im Mittel über fünf Aufnahmen: Tiefe 0 (linear), Engpass 2, 1000 Epochen. Der Decoder-Unterraum ist dann mit dem PCA-Unterraum identisch (Winkel 0.03° nach 1000 Epochen; nach 152 Epochen 1°), und der Rekonstruktionsfehler ist das PCA-Residuum: "
                                  "AUC 0.49, wie das PCA-Residuum (0.49) - blind. Nach 12 Epochen, wo der Unterraum noch beliebig liegt (Winkel 87°), war die AUC 1.00.",
    "Korrelationsbruch": "Im Mittel über fünf Aufnahmen: die Merkmale der Anomalien sind je Spalte aus den Normalen neu gezogen - sie verlassen die Ebene. Der Autoencoder hat AUC 1.00 und F1 0.98 (Fehlalarmrate 0 %): hier trifft er. "
                         "Robust 1.00 / 0.84, LOF 0.99, Deep SVDD 0.92, One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56. **Hier hilft das Training**: untrainiert 0.70, nach 100 Epochen 1.00.",
    "40 Rauschmerkmale": "Im Mittel über fünf Aufnahmen: 40 unabhängige Rauschmerkmale. AUC 0.25 - **unter Raten**, F1 0.03: 96 % des Fehlers der normalen Touren kommen aus den Rauschspalten (95 % bei den Sonderfahrten), der Fehler der 12 Kennzahlen ist bei beiden gleich (0.108 gegenüber 0.112). "
                         "Isolation Forest 0.99, robust 0.88, Deep SVDD 0.71; mit Engpass 1 immerhin 0.68.",
    "Kleine Stichprobe (50 Touren)": "Im Mittel über fünf Aufnahmen: 50 Touren, 5 Sonderfahrten. AUC 0.29, F1 0.04 - unter Raten: der mittlere Fehler der Sonderfahrten ist 0.019, der der normalen Touren 0.027 (bei 20 Touren 0.001 gegenüber 0.020). "
                                     "Das untrainierte Netz hat 1.00, der Isolation Forest 1.00, Deep SVDD 0.91.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (ae_*, iforest_*, robust_*, dsvdd_*, untrained_*, pca_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"ae_auc": (0.45, 0.9), "untrained_auc": (0.97, 1.0), "pca_auc": (0.3, 0.65), "verdict": ("training_hurts",)},
    "Ohne Training (0 Epochen)": {"ae_auc": (0.97, 1.0), "ae_f1": (0.85, 1.0), "verdict": ("comparable",)},
    "Engpass 1": {"ae_auc": (0.7, 0.98), "untrained_auc": (0.97, 1.0), "verdict": ("training_hurts",)},
    "Linearer Autoencoder = PCA": {"ae_auc": (0.3, 0.65), "pca_auc": (0.3, 0.65), "iforest_auc": (0.95, 1.0), "verdict": ("blind",)},
    "Korrelationsbruch": {"ae_auc": (0.97, 1.0), "ae_f1": (0.85, 1.0), "robust_auc": (0.95, 1.0), "untrained_auc": (0.4, 0.85), "verdict": ("ae_wins", "comparable")},
    "40 Rauschmerkmale": {"ae_auc": (0.0, 0.5), "untrained_auc": (0.9, 1.0), "iforest_auc": (0.95, 1.0), "verdict": ("blind",)},
    "Kleine Stichprobe (50 Touren)": {"ae_auc": (0.0, 0.6), "untrained_auc": (0.95, 1.0), "iforest_auc": (0.95, 1.0), "verdict": ("blind",)},
}
