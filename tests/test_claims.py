"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel). Das untrainierte Netz, das PCA-Residuum und die anderen Detektoren sind fast deterministisch
(Toleranz ±0.02); trainierte Netze werden über viele Epochen gerechnet und tragen Rundungsunterschiede zwischen numpy-/BLAS-Versionen mit: dort sind die Bänder weiter (±0.06 AUC, ±0.09 F1).
Positive UND negative Aussagen: wo der Autoencoder gegen das untrainierte Netz, das PCA-Residuum, den Isolation Forest, die Wurzel oder Deep SVDD verliert, steht das hier ebenso als Test wie dort, wo er gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import aeano_algorithm as ae
import aeano_constants as C
import aeano_evaluation as ev

TOL = 0.02
TR = 0.06                          # AUC trainierter Netze
TR_F1 = 0.09
STANDARD = ev.Settings()


def S(**kw):
    return ev.Settings(**kw)


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@lru_cache(maxsize=None)
def curves():
    return {r["scenario"]: dict(zip(r["epochs"], r["auc"])) for r in ev.epoch_curves()}


@lru_cache(maxsize=None)
def seeds():
    return {r["epochs"]: r for r in ev.seed_table()}


@lru_cache(maxsize=None)
def bottleneck():
    return {(r["kind"], r["n_code"]): r for r in ev.bottleneck_table()}


@lru_cache(maxsize=None)
def linear():
    return ev.linear_table()


# --- Das untrainierte Netz und das PCA-Residuum als Bezug ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("ep,auc,f1", [(0, 1.00, 0.98), (10, 1.00, 0.97), (30, 0.97, 0.86), (100, 0.87, 0.58), (300, 0.68, 0.26), (1000, 0.56, 0.14)])
def test_epochs_sweep_the_ranking_gets_worse_with_training(ep, auc, f1):
    s = S(epochs=ep)
    near(m("ae", "auc", s), auc, 0.012 if ep <= 10 else (0.04 if ep <= 30 else TR))
    near(m("ae", "f1", s), f1, 0.03 if ep <= 10 else TR_F1)


def test_standard_case_and_the_reference_detectors():
    near(m("ae", "auc"), 0.68, TR)
    near(m("ae", "f1"), 0.26, TR_F1)
    near(m("untrained", "auc"), 1.00, 0.01)
    near(m("untrained", "f1"), 0.98, 0.03)
    near(m("untrained", "false_alarm"), 0.001, 0.01)
    near(m("pca", "auc"), 0.49, 0.03)
    near(oracle("pca"), 0.11, 0.04)
    near(m("iforest", "auc"), 1.00, 0.01)
    near(m("iforest", "f1"), 0.96)
    near(m("dsvdd", "auc"), 0.92, 0.03)
    near(m("dsvdd", "f1"), 0.62, 0.08)
    near(oracle("ae"), 0.27, 0.10)


def test_training_hurts_at_every_length():
    for ep in (30, 100, 300, 1000):
        assert m("untrained", "auc") - m("ae", "auc", S(epochs=ep)) >= 0.01
    assert m("untrained", "auc") - m("ae", "auc", S(epochs=300)) >= 0.2


# --- Seitenleiste: Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil -------------------------------------------------


@pytest.mark.parametrize("n,auc", [(20, 0.00), (30, 0.15), (50, 0.29), (100, 0.43), (200, 0.57), (400, 0.70), (600, 0.72)])
def test_tours_sweep_the_autoencoder_is_below_chance_for_few_tours(n, auc):
    near(m("ae", "auc", n=n), auc, 0.12)
    near(m("untrained", "auc", n=n), 1.00, 0.015)


def test_few_tours_the_anomalies_are_reconstructed_better():
    def mean_errors(n):
        en, ea = [], []
        for a in runs(STANDARD, n=n):
            en.append(a.net.errors[~a.ds.anomaly].mean())
            ea.append(a.net.errors[a.ds.anomaly].mean())
        return float(np.mean(en)), float(np.mean(ea))
    en, ea = mean_errors(20)
    near(ea, 0.001, 0.01)
    near(en, 0.020, 0.01)
    assert ea < 0.2 * en
    en, ea = mean_errors(50)
    near(ea, 0.019, 0.015)
    near(en, 0.027, 0.015)
    assert ea < en


@pytest.mark.parametrize("p,auc,pca", [(2, 0.94, 0.80), (5, 0.67, 0.51), (8, 0.68, 0.47), (12, 0.68, 0.49), (20, 0.79, 0.47), (30, 0.86, 0.46)])
def test_features_sweep(p, auc, pca):
    near(m("ae", "auc", p=p), auc, 0.09)
    near(m("pca", "auc", p=p), pca, 0.03)


@pytest.mark.parametrize("nn,auc,k1", [(0, 0.68, 0.88), (5, 0.48, 0.86), (10, 0.45, 0.82), (20, 0.34, 0.75), (30, 0.26, 0.71), (40, 0.25, 0.68)])
def test_noise_features_sweep(nn, auc, k1):
    near(m("ae", "auc", n_noise=nn), auc, 0.09)
    near(m("ae", "auc", S(n_code=1), n_noise=nn), k1, 0.06)
    assert m("untrained", "auc", n_noise=nn) >= 0.99


def test_noise_features_the_error_comes_from_the_noise_columns():
    frac_n, frac_a, e_n, e_a = [], [], [], []
    for a in runs(STANDARD, n_noise=40):
        err = ae.reconstruct(a.net)[2]
        share = err[:, 12:].sum(axis=1) / err.sum(axis=1)
        frac_n.append(share[~a.ds.anomaly].mean())
        frac_a.append(share[a.ds.anomaly].mean())
        e_n.append(err[~a.ds.anomaly][:, :12].mean())
        e_a.append(err[a.ds.anomaly][:, :12].mean())
    near(float(np.mean(frac_n)), 0.96, 0.03)
    near(float(np.mean(frac_a)), 0.95, 0.03)
    near(float(np.mean(e_n)), 0.112, 0.02)
    near(float(np.mean(e_a)), 0.108, 0.02)
    near(m("iforest", "auc", n_noise=40), 0.99, 0.015)
    near(m("robust", "auc", n_noise=40), 0.88, 0.015)
    near(m("untrained", "auc", n_noise=40), 0.995, 0.01)


@pytest.mark.parametrize("modes,auc", [(1, 0.68), (2, 0.72), (3, 0.73)])
def test_modes_sweep(modes, auc):
    near(m("ae", "auc", n_modes=modes), auc, 0.09)
    near(m("untrained", "auc", n_modes=modes), 1.00, 0.02)


def test_modes_robust():
    for modes, v in ((1, 1.00), (2, 0.95), (3, 0.85)):
        near(m("robust", "auc", n_modes=modes), v, 0.015)


@pytest.mark.parametrize("curv,auc,f1,pca", [(0.0, 0.68, 0.26, 0.49), (0.25, 0.96, 0.77, 0.99), (0.5, 0.99, 0.82, 0.98), (0.75, 0.99, 0.84, 0.99), (1.0, 0.98, 0.83, 0.99)])
def test_curvature_sweep_the_anomalies_leave_the_straight_plane(curv, auc, f1, pca):
    near(m("ae", "auc", curvature=curv), auc, 0.06)
    near(m("ae", "f1", curvature=curv), f1, 0.10)
    near(m("pca", "auc", curvature=curv), pca, 0.03)


@pytest.mark.parametrize("noise,auc,f1", [(0.0, 0.95, 0.73), (0.25, 0.68, 0.26), (0.5, 0.55, 0.14), (0.75, 0.50, 0.10), (1.0, 0.48, 0.11)])
def test_measurement_noise_sweep(noise, auc, f1):
    near(m("ae", "auc", noise=noise), auc, 0.09)
    near(m("ae", "f1", noise=noise), f1, 0.12)


@pytest.mark.parametrize("c,auc,k1", [(2, 0.85, 0.92), (5, 0.80, 0.92), (10, 0.68, 0.88), (20, 0.66, 0.89), (30, 0.68, 0.89), (40, 0.60, 0.86), (45, 0.63, 0.87)])
def test_contamination_sweep_the_training_learns_the_anomalies(c, auc, k1):
    near(m("ae", "auc", contamination=c), auc, 0.09)
    near(m("ae", "auc", S(n_code=1), contamination=c), k1, 0.06)
    assert m("untrained", "auc", contamination=c) >= 0.99


# --- Art der Anomalien ---------------------------------------------------------------------------------------------------------------------


def test_dense_group_and_gap_are_missed():
    kw = dict(kind="cluster", contamination=30)
    near(m("ae", "auc", **kw), 0.50, 0.10)
    near(m("untrained", "auc", **kw), 0.95, 0.03)
    near(m("pca", "auc", **kw), 0.48, 0.03)
    near(m("dsvdd", "auc", **kw), 0.18, 0.10)
    near(m("iforest", "auc", **kw), 0.70, 0.02)
    near(m("ocsvm", "auc", **kw), 0.74, 0.02)
    near(m("ecod", "auc", **kw), 0.84, 0.02)
    for nm, un in ((2, 0.06), (3, 0.01)):
        near(m("ae", "auc", kind="gap", n_modes=nm), 0.49, 0.10)
        near(m("untrained", "auc", kind="gap", n_modes=nm), un, 0.03)
        assert 0.4 <= m("pca", "auc", kind="gap", n_modes=nm) <= 0.6


def test_correlation_break_is_the_domain_of_the_autoencoder():
    kw = dict(kind="decorrelated")
    near(m("ae", "auc", **kw), 1.00, 0.02)
    near(m("ae", "f1", **kw), 0.98, 0.04)
    near(oracle("ae", **kw), 0.99, 0.03)
    near(m("pca", "auc", **kw), 1.00, 0.01)
    near(m("robust", "auc", **kw), 1.00, 0.01)
    near(m("robust", "f1", **kw), 0.84, 0.02)
    near(m("lof", "auc", **kw), 0.99, 0.02)
    near(m("dsvdd", "auc", **kw), 0.92, 0.03)
    near(m("ocsvm", "auc", **kw), 0.88, 0.02)
    near(m("iforest", "auc", **kw), 0.80, 0.02)
    near(m("ecod", "auc", **kw), 0.56, 0.03)
    near(m("untrained", "auc", **kw), 0.70, 0.03)


# --- Regler des Netzes ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,auc,f1,pca,decor", [(1, 0.88, 0.69, 0.91, 0.88), (2, 0.68, 0.26, 0.49, 1.00), (3, 0.64, 0.22, 0.49, 1.00), (4, 0.78, 0.38, 0.50, 1.00), (6, 0.81, 0.43, 0.50, 1.00), (8, 0.88, 0.56, 0.52, 0.98)])
def test_bottleneck_sweep(k, auc, f1, pca, decor):
    near(m("ae", "auc", S(n_code=k)), auc, 0.09)
    near(m("ae", "f1", S(n_code=k)), f1, 0.12)
    near(m("pca", "auc", S(n_code=k)), pca, 0.03)
    near(m("ae", "auc", S(n_code=k), kind="decorrelated"), decor, 0.05)


@pytest.mark.parametrize("d,auc,f1", [(0, 0.49, 0.12), (1, 0.68, 0.26), (2, 0.71, 0.27), (3, 0.76, 0.37)])
def test_depth_sweep_the_linear_network_is_the_pca_residual(d, auc, f1):
    near(m("ae", "auc", S(depth=d)), auc, 0.09)
    near(m("ae", "f1", S(depth=d)), f1, 0.12)
    if d == 0:
        near(m("ae", "auc", S(depth=0)), m("pca", "auc"), 0.05)


@pytest.mark.parametrize("w,auc", [(4, 0.89), (8, 0.85), (16, 0.68), (32, 0.73), (64, 0.52)])
def test_width_sweep(w, auc):
    near(m("ae", "auc", S(width=w)), auc, 0.10)


@pytest.mark.parametrize("ep,relu", [(0, 0.96), (10, 1.00), (30, 1.00), (100, 0.89), (300, 0.77), (1000, 0.59)])
def test_relu_sweep(ep, relu):
    near(m("ae", "auc", S(activation="relu", epochs=ep)), relu, 0.03 if ep <= 30 else 0.10)


@pytest.mark.parametrize("nu,f1,recall,fa", [(0.01, 0.08, 0.04, 0.003), (0.02, 0.17, 0.10, 0.007), (0.05, 0.25, 0.18, 0.032), (0.10, 0.26, 0.26, 0.079), (0.20, 0.26, 0.38, 0.176), (0.30, 0.25, 0.49, 0.275), (0.50, 0.24, 0.70, 0.474)])
def test_nu_sweep_the_threshold_marks_about_nu_of_the_tours(nu, f1, recall, fa):
    s = S(nu=nu)
    near(m("ae", "f1", s), f1, 0.10)
    near(m("ae", "recall", s), recall, 0.10)
    near(m("ae", "false_alarm", s), fa, 0.04)
    near(m("ae", "auc", s), 0.68, TR)


def test_nu_flags_at_most_nu_of_the_tours():
    for nu in (0.05, 0.1, 0.3):
        assert all(a.scores["ae"]["n_flagged"] / a.ds.n <= nu + 1e-9 for a in runs(S(nu=nu)))


@pytest.mark.parametrize("q,f1", [(0.9, 0.67), (0.95, 0.77), (0.975, 0.84), (0.99, 0.89), (0.999, 0.90)])
def test_chi2_quantile_help_numbers(q, f1):
    near(m("robust", "f1", S(quantile=q)), f1)


@pytest.mark.parametrize("share,ae_f1,iforest,robust", [(2, 0.18, 0.33, 0.33), (5, 0.25, 0.67, 0.67), (10, 0.27, 0.97, 0.90), (20, 0.25, 0.67, 0.66), (40, 0.25, 0.40, 0.40)])
def test_assumed_share_costs_all_detectors(share, ae_f1, iforest, robust):
    s = S(threshold_kind="share", share=share)
    near(m("ae", "f1", s), ae_f1, 0.10)
    near(m("iforest", "f1", s), iforest)
    near(m("robust", "f1", s), robust)


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_presets_untrained_bottleneck_one_and_linear():
    z = S(epochs=0)
    near(m("ae", "auc", z), 1.00, 0.01)
    near(m("ae", "f1", z), 0.98, 0.03)
    near(m("ae", "false_alarm", z), 0.001, 0.01)
    k1 = S(n_code=1)
    near(m("ae", "auc", k1), 0.88, 0.09)
    near(m("ae", "f1", k1), 0.69, 0.12)
    near(m("pca", "auc", k1), 0.91, 0.03)
    near(m("ae", "auc", S(n_code=1), kind="decorrelated"), 0.88, 0.05)
    lin = S(depth=0, epochs=1000)
    near(m("ae", "auc", lin), 0.49, 0.06)
    near(m("pca", "auc", lin), 0.49, 0.03)


def test_presets_noise_and_small_sample():
    near(m("ae", "auc", n_noise=40), 0.25, 0.09)
    near(m("ae", "f1", n_noise=40), 0.03, 0.05)
    near(m("dsvdd", "auc", n_noise=40), 0.71, 0.03)
    near(m("ae", "auc", S(n_code=1), n_noise=40), 0.68, 0.06)
    near(m("ae", "auc", n=50), 0.29, 0.12)
    near(m("ae", "f1", n=50), 0.04, 0.06)
    near(m("untrained", "auc", n=50), 1.00, 0.015)
    near(m("iforest", "auc", n=50), 1.00, 0.01)
    near(m("dsvdd", "auc", n=50), 0.91, 0.03)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario,ae_auc,un,pca,dsvdd,iforest,robust", [("Standardfall", 0.68, 1.00, 0.49, 0.92, 1.00, 1.00), ("gekrümmter Normalbereich", 0.98, 1.00, 0.99, 0.98, 1.00, 1.00), ("dichte Gruppe 30 %", 0.50, 0.95, 0.48, 0.18, 0.70, 0.49),
                                                                          ("Lücke, 2 Betriebsarten", 0.49, 0.06, 0.50, 0.45, 0.54, 0.40), ("Lücke, 3 Betriebsarten", 0.49, 0.01, 0.51, 0.53, 0.27, 0.38),
                                                                          ("Korrelationsbruch", 1.00, 0.70, 1.00, 0.92, 0.80, 1.00), ("45 % verstreut", 0.63, 1.00, 0.47, 0.83, 1.00, 0.87), ("40 Rauschmerkmale", 0.25, 0.995, 0.46, 0.71, 0.99, 0.88)])
def test_scenario_table_auc(scenario, ae_auc, un, pca, dsvdd, iforest, robust):
    kw = dict(dict(ev.SCENARIOS)[scenario])
    near(m("ae", "auc", **kw), ae_auc, 0.10)
    near(m("untrained", "auc", **kw), un, 0.03)
    near(m("pca", "auc", **kw), pca, 0.03)
    near(m("dsvdd", "auc", **kw), dsvdd, 0.10 if dsvdd < 0.6 else 0.04)
    near(m("iforest", "auc", **kw), iforest, 0.02)
    near(m("robust", "auc", **kw), robust, 0.02)


def test_scenario_f1_statements():
    near(oracle("ae"), 0.27, 0.10)
    near(oracle("untrained"), 0.98, 0.03)
    near(oracle("pca"), 0.11, 0.04)
    near(oracle("ae", kind="decorrelated"), 0.99, 0.03)


@pytest.mark.parametrize("scenario,points", [("Standardfall", {0: 1.00, 33: 0.97, 105: 0.87, 600: 0.64}), ("gekrümmter Normalbereich", {0: 1.00, 600: 0.96}), ("45 % verstreut", {0: 1.00, 18: 0.79, 33: 0.65, 600: 0.56}),
                                             ("40 Rauschmerkmale", {0: 0.99, 33: 0.84, 59: 0.59, 600: 0.21}), ("dichte Gruppe 30 %", {0: 0.96, 18: 0.37, 600: 0.49}),
                                             ("Korrelationsbruch", {0: 0.62, 33: 0.97, 105: 1.00}), ("Lücke, 2 Betriebsarten", {0: 0.03, 33: 0.52}), ("Lücke, 3 Betriebsarten", {0: 0.00, 33: 0.49})])
def test_epoch_curves(scenario, points):
    c = curves()[scenario]
    for e, v in points.items():
        near(c[e], v, 0.05 if e == 0 else 0.13)


def test_epoch_curves_training_helps_only_where_the_structure_must_be_learned():
    c = curves()
    assert c["Korrelationsbruch"][105] > c["Korrelationsbruch"][0] + 0.3
    for s in ("Lücke, 2 Betriebsarten", "Lücke, 3 Betriebsarten"):
        assert c[s][33] > c[s][0] + 0.3 and max(c[s].values()) < 0.7
    for s in ("Standardfall", "45 % verstreut", "40 Rauschmerkmale", "dichte Gruppe 30 %"):
        assert c[s][600] < c[s][0] - 0.2


@pytest.mark.parametrize("ep,mean,std,mn,mx,f1", [(0, 1.00, 0.00, 1.00, 1.00, 0.97), (30, 0.98, 0.01, 0.95, 1.00, 0.93), (100, 0.88, 0.04, 0.77, 0.94, 0.63), (300, 0.73, 0.07, 0.61, 0.92, 0.34)])
def test_seed_table(ep, mean, std, mn, mx, f1):
    r = seeds()[ep]
    near(r["auc"], mean, 0.05)
    near(r["auc_std"], std, 0.04)
    near(r["auc_min"], mn, 0.09)
    near(r["auc_max"], mx, 0.05)
    near(r["f1"], f1, 0.09)


def test_seed_spread_grows_with_training():
    r = seeds()
    assert r[0]["auc_std"] < r[300]["auc_std"] and r[0]["auc_min"] > r[300]["auc_min"] + 0.2


@pytest.mark.parametrize("kind,vals,pcas", [("verstreut", (0.88, 0.68, 0.64, 0.78, 0.81, 0.88), (0.91, 0.49, 0.49, 0.50, 0.50, 0.52)), ("Korrelationsbruch", (0.88, 1.00, 1.00, 1.00, 1.00, 0.98), (0.82, 1.00, 1.00, 1.00, 1.00, 0.97)),
                                            ("gekrümmt", (1.00, 0.98, 0.97, 0.94, 0.97, 0.98), (1.00, 0.99, 0.97, 0.86, 0.72, 0.60))])
def test_bottleneck_table(kind, vals, pcas):
    for k, v, pv in zip((1, 2, 3, 4, 6, 8), vals, pcas):
        near(bottleneck()[(kind, k)]["auc"], v, 0.09)
        near(bottleneck()[(kind, k)]["pca_auc"], pv, 0.04)


def test_linear_autoencoder_converges_to_the_pca_subspace():
    li = linear()
    ang = dict(zip(li["epochs"], li["angle"]))
    auc = dict(zip(li["epochs"], li["auc"]))
    near(ang[0], 83.0, 8.0)
    near(ang[12], 87.0, 6.0)
    near(ang[81], 27.0, 15.0)
    assert ang[152] < 3.0 and ang[1000] < 0.3
    assert min(auc[e] for e in li["epochs"] if e <= 23) > 0.97
    near(auc[81], 0.97, 0.06)
    near(auc[152], 0.52, 0.08)
    near(auc[1000], 0.49, 0.06)
    near(li["pca_auc"], 0.486, 0.03)
    near(auc[1000], li["pca_auc"], 0.05)


def test_wrong_share_and_cutoff_table():
    tt = ev.threshold_table(STANDARD)
    by_factor = {r["factor"]: r for r in tt["wrong_share"]}
    near(by_factor[0.5]["ae_f1"], 0.25, 0.10)
    near(by_factor[1.0]["ae_f1"], 0.27, 0.10)
    near(by_factor[2.0]["ae_f1"], 0.25, 0.10)
    near(by_factor[0.5]["iforest_f1"], 0.67)
    near(by_factor[1.0]["iforest_f1"], 0.97)
    near(by_factor[2.0]["iforest_f1"], 0.67)
    near(by_factor[0.5]["dsvdd_f1"], 0.59, 0.08)
    near(by_factor[1.0]["dsvdd_f1"], 0.63, 0.08)
    near(by_factor[2.0]["dsvdd_f1"], 0.55, 0.08)


def test_cost_table():
    times = {(t["n"], t["p"]): t for t in ev.cost_table()["times"]}
    for t in times.values():
        assert t["ae"] < 1.0 and t["ae_1000"] > 2 * t["ae"] and t["ae_1000"] < 4.0 and t["pca"] < 0.05 and t["ae"] > 3 * t["pca"] and t["ae"] > t["dsvdd"] * 0.9 and t["ae"] < t["dsvdd"] * 6
    assert times[(600, 12)]["ae"] > times[(100, 12)]["ae"]


# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------


def test_limits_table_numbers():
    near(m("ae", "auc"), 0.68, TR)
    near(m("pca", "auc"), 0.49, 0.03)
    near(m("ae", "auc", S(depth=0)), 0.49, 0.06)
    near(m("ae", "auc", S(epochs=0)), 1.00, 0.01)
    near(m("ae", "auc", S(epochs=100)), 0.87, TR)
    near(m("ae", "auc", S(epochs=1000)), 0.56, 0.10)
    near(m("ae", "f1", S(epochs=100)), 0.58, TR_F1)
    near(m("ae", "f1", S(epochs=1000)), 0.14, TR_F1)
    near(m("ae", "auc", contamination=45), 0.63, 0.10)
    near(m("ae", "auc", contamination=2), 0.85, 0.09)
    near(m("ae", "auc", S(n_code=1)), 0.88, 0.09)
    near(m("ae", "auc", S(n_code=8)), 0.88, 0.09)
    near(m("pca", "auc", S(n_code=8)), 0.52, 0.03)
    near(m("ae", "f1", S(nu=0.01)), 0.08, 0.06)
    near(m("ae", "f1", S(nu=0.5)), 0.24, 0.10)
    for n, v in ((20, 0.00), (30, 0.15), (50, 0.29)):
        near(m("ae", "auc", n=n), v, 0.12)
