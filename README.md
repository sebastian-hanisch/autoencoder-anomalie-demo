# Autoencoder – der Rekonstruktionsfehler als Anomalie-Wert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-autoencoder-anomalie-demo.streamlit.app/)**

Neuntes und **letztes** Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Autoencoder als Anomalie-Detektor** – an einem wachsenden Beispiel, gegen den [Isolation Forest](../isolation-forest-demo), die robuste Schätzung der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo))
und [Deep SVDD](../deepsvdd-demo); als Bezug laufen das **untrainierte** Netz, das **PCA-Residuum**, die [One-Class SVM](../ocsvm-demo), der [LOF](../lof-demo) und [ECOD](../ecod-demo) mit. Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo)
(Szenario, One-Class SVM, Deep SVDD, LOF, Isolation Forest, Wurzel-Schätzer und ECOD-Kern wortgleich übernommen, per Test gegen eingefrorene Werte geprüft; die klassische Schätzung entfällt, damit die Balken lesbar bleiben). Das Netz baut auf dem numpy-MLP der [autoencoder-demo](../autoencoder-demo) der Dimensionsreduktion auf.

**Einordnung in die Reihe (die Kanten des Graphen):** der Autoencoder ist der **unabhängige Rekonstruktions-Ast direkt nach der Wurzel**. Die anderen Äste fragen, wie weit eine Tour von den Normalen entfernt ist oder wie leicht sie sich isolieren lässt; der Autoencoder fragt, wie gut ein Netz sie **durch einen Engpass hindurch wieder aufbauen** kann.
Ergebnis in Kürze: **er trifft, was die gelernte Struktur verlässt – und ist blind für das, was auf ihr liegt.** Beim **Korrelationsbruch** (die Anomalien verlassen die Ebene der Normalen) AUC **1.00** und F1 0.98; im Standardfall, wo die Sonderfahrten Verschiebungen der zwei latenten Faktoren sind und **in derselben Ebene** liegen, AUC 0.68 und F1 0.26 –
das PCA-Residuum hat 0.49 (Raten), und das lineare Autoencoder ist nach 300 Epochen genau das PCA-Residuum. Und wie bei Deep SVDD **schadet das Training**: das untrainierte Netz hat AUC 1.00 und F1 0.98.
Die Linie hat **keinen Konvergenzpunkt**; dieses Stück schließt sie ab – alle neun Stücke sind gebaut.
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ecod-demo                  (Kontrast: verteilungsfrei)                              [gebaut]
  ├─ lof-demo → feature-bagging-demo (lokale Dichte; Ensembles gegen viele Merkmale)     [beide gebaut]
  ├─ ocsvm-demo → deepsvdd-demo (gelernte Grenze; gelernte Abbildung)                    [beide gebaut]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)               [beide gebaut]
  └─ autoencoder-anomalie-demo  (Rekonstruktionsfehler)                                  [dieses Stück – Linie komplett]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; Autoencoder mit einer verdeckten Schicht je Seite der Breite 16, Engpass 2, tanh, 300 Epochen (Adam, Lernrate 0.01, Vollbatch), ν = 0.1 (Schwelle = 0.9-Quantil der Trainingsfehler), Netz-Seed 0; Isolation Forest 100 Bäume × ψ = 256 und Schwelle 0.5, robust χ²-Quantil 0.975, Deep SVDD Voreinstellung; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall, Voreinstellung | ❌ Autoencoder AUC **0.68**, F1 **0.26** (Fehlalarmrate 7.9 %). **PCA-Residuum bei Engpass 2: 0.49** – die Sonderfahrten liegen in der Faktorebene der Normalen und werden mitrekonstruiert. Das **untrainierte** Netz: AUC 1.00, F1 0.98. Isolation Forest 1.00 / 0.96, robust 1.00 / 0.84, Deep SVDD 0.92 / 0.62 |
| **Epochen** | ❌ **Training schadet:** AUC bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen **1.00 / 1.00 / 0.97 / 0.87 / 0.68 / 0.56** (nach 1000 Epochen je nach Rechner bis 0.67), F1 0.98 / 0.97 / 0.86 / 0.58 / 0.26 / 0.14 (nach 1000 Epochen je nach Rechner bis 0.33). Das Netz gibt vor dem Training fast nichts wieder; sein Fehler ist im Wesentlichen die Länge der Tour, also ihr Abstand von der Mitte. Bei ReLU 0.96 / 1.00 / 1.00 / 0.89 / 0.77 / 0.59 |
| **Lineares Autoencoder = PCA** | ✅ Der größte Hauptwinkel zwischen Decoder-Unterraum und PCA-Unterraum fällt von 83° (Epoche 0) über 27° (81) und 1° (152) auf **0.03° (1000)**: **derselbe Unterraum** (Baldi und Hornik), nicht dieselben Achsen. Die AUC folgt: 1.00 bis 23 Epochen (der Unterraum liegt noch beliebig), 0.97 nach 81, 0.52 nach 152, **0.49** danach – gleich dem PCA-Residuum (0.49) |
| **Korrelationsbruch** | ✅ **Hier trifft er:** AUC **1.00**, F1 **0.98** (Fehlalarmrate 0 %), mit bekanntem Anteil 0.99. Die Anomalien haben in jedem Merkmal normale Werte, aber die Abhängigkeit stimmt nicht – sie **verlassen die Ebene**. PCA-Residuum 1.00, robust 1.00 (F1 0.84), LOF 0.99, Deep SVDD 0.92, One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56. Hier **hilft das Training**: untrainiert 0.70, Verlauf 0.62 → 0.97 (33 Epochen) → 1.00 (105) |
| **Engpass** | ⚠️ AUC bei Engpass 1 / 2 / 3 / 4 / 6 / 8 (verstreute Anomalien): **0.88** / 0.68 / 0.64 / 0.78 / 0.81 / 0.88, F1 0.69 / 0.26 / 0.22 / 0.38 / 0.43 / 0.56; PCA-Residuum 0.91 / 0.49 / 0.49 / 0.50 / 0.50 / 0.52. Beim Korrelationsbruch **0.88** / 1.00 / 1.00 / 1.00 / 1.00 / 0.98. Ein Engpass **unter** der latenten Dimension (2) sieht Anomalien in der Ebene und verliert beim Korrelationsbruch |
| Tiefe, Breite | ➖ Tiefe 0 / 1 / 2 / 3: AUC 0.49 / 0.68 / 0.71 / 0.76 (F1 0.12 / 0.26 / 0.27 / 0.37); Breite 4 / 8 / 16 / 32 / 64: 0.89 / 0.85 / 0.68 / 0.73 / 0.52 – je mehr Spielraum, desto besser rekonstruiert das Netz auch die Anomalien |
| Krümmung | ✅ Bei Krümmung 0 / 0.25 / 0.5 / 0.75 / 1 steigt die AUC von 0.68 auf 0.96 / 0.99 / 0.99 / 0.98 (F1 0.26 → 0.77 / 0.82 / 0.84 / 0.83): die Anomalien verlassen jetzt die gerade Ebene. PCA-Residuum 0.49 / 0.99 / 0.98 / 0.99 / 0.99 |
| Messrauschen | ❌ AUC bei 0 / 0.25 / 0.5 / 0.75 / 1.0: 0.95 / 0.68 / 0.55 / 0.50 / 0.48 – ohne Rauschen liegen die Normalen genau in der Ebene und jede Abweichung fällt auf; mit Rauschen ist die Ebene „dick“ |
| Rauschmerkmale | ❌ AUC bei 0 / 5 / 10 / 20 / 30 / 40 Rauschmerkmalen 0.68 / 0.48 / 0.45 / 0.34 / 0.26 / **0.25 – unter Raten**: bei 40 Rauschmerkmalen kommen 96 % des Fehlers der normalen Touren aus den Rauschspalten (95 % bei den Sonderfahrten), der Fehler der 12 Kennzahlen ist bei beiden gleich (0.108 gegenüber 0.112). Mit Engpass 1: 0.88 → 0.68. Isolation Forest 0.99, robust 0.88, untrainiert 0.995 |
| Kleine Stichproben | ❌ AUC bei 20 / 30 / 50 / 100 / 200 / 400 / 600 Touren **0.00 / 0.15 / 0.29** / 0.43 / 0.57 / 0.70 / 0.72: die Sonderfahrten werden besser rekonstruiert als die normalen Touren (mittlerer Fehler bei 20 Touren 0.001 gegenüber 0.020, bei 50 Touren 0.019 gegenüber 0.027). Untrainiert 1.00 |
| Viele Anomalien / dichte Gruppe / Lücke | ❌ Das Training lernt die Anomalien mit: AUC bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % 0.85 / 0.80 / 0.68 / 0.66 / 0.68 / 0.60 / 0.63 (Engpass 1: 0.92 … 0.87; untrainiert 1.00). Dichte Gruppe (30 %): 0.50 (untrainiert 0.95, Deep SVDD 0.18); Lücke (2 / 3 Betriebsarten): 0.49 / 0.49 – nur das Training hebt sie von 0.03 / 0.00 auf 0.52 / 0.49, nie über Raten |
| Zufall der Initialisierung | ⚠️ Über 10 Netz-Seeds und 5 Datensätze (50 Paare): AUC nach 0 / 30 / 100 / 300 Epochen **1.00 ± 0.00 / 0.98 ± 0.01 / 0.88 ± 0.04 / 0.73 ± 0.07**, schlechtestes Netz 1.00 / 0.95 / 0.77 / 0.61, bestes 1.00 / 1.00 / 0.94 / 0.92 |
| Schwelle | ➖ Die Schwelle markiert **so viele Touren, wie ν vorgibt**: F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gleich 0.08 / 0.17 / 0.25 / 0.26 / 0.26 / 0.25 / 0.24 (AUC bleibt 0.68); bei schlechter Rangfolge hilft auch das richtige ν nicht. Falsch angenommener Anteil (½× / 1× / 2×): Autoencoder 0.25 / 0.27 / 0.25, Isolation Forest 0.67 / 0.97 / 0.67 |
| Rechenzeit | ➖ Training mit 300 Epochen bei 100 / 300 / 600 Touren und 12 Merkmalen 36 / 58 / 89 ms, mit 1000 Epochen 0.12 / 0.19 / 0.30 s – das Eineinhalb- bis Zweieinhalbfache von Deep SVDD (14 / 29 / 54 ms), das PCA-Residuum braucht 0.1–0.2 ms (geschlossene Form) |

## Was die Demo zeigt

1. **Autoencoder in Aktion** (Schritt-Slider + Abspielen): **Touren** (Ebene der größten Streuung und zwei Rohmerkmale) → **Netz** (das zufällig initialisierte Netz rekonstruiert eine normale Tour und eine Sonderfahrt, Merkmal für Merkmal; schon hier AUC 1.00) →
   **Training** (Verlust und AUC über die Epochen; der Rekonstruktionsfehler in der Ebene zweier Rohmerkmale, dafür mit Engpass 1 neu trainiert) → **Fehler** (dieselben Touren nach dem Training, Fehler je Merkmal, Verteilung mit Schwelle, Rang gegen die Wurzel) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Detektoren gefunden haben:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → **blind** (AUC unter 0.6, ein anderer Detektor über 0.8) → **Training schadet** (das untrainierte Netz ist um mindestens 0.05 AUC besser) → anderer Detektor besser → Autoencoder besser (Rangfolge oder Schwelle) → falsche Schwelle → ebenbürtig), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, **Engpass, Tiefe, Breite, Epochen, ν**, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** AUC über die Epochen in acht Szenarien, Engpass 1–8 für drei Arten der Anomalien (gegen das PCA-Residuum), lineares Autoencoder gegen den PCA-Unterraum (Hauptwinkel und AUC über die Epochen), 10 Netz-Seeds, acht Szenarien × Detektoren (mit untrainiertem Netz und PCA-Residuum), Schwelle, Kosten.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Anomalien verlassen die gelernte Struktur, das Training verbessert den Detektor, man weiß, wann man aufhören muss, der Engpass passt, alle Merkmale sind rekonstruierbar, genug Touren, wenig Anomalien im Training, die Schwelle ist ein Anteil).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten / Korrelationsbruch), Abstand (bei "Lücke" und "Korrelationsbruch" ausgeblendet, Wert bleibt erhalten),
**Engpass** (1–8), **Tiefe** (0–3; 0 = linear), **Breite** (4–64), **Epochen** (0–1000; 0 = das untrainierte Netz), **ν**, **Aktivierung** (tanh / ReLU), **Netz-Seed** (+ 🎲, getrennt vom Seed der Touren),
**Schwelle** (Standard: (1 − ν)-Quantil der Trainingsfehler und χ²-Quantil der Wurzel, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC AE | F1 AE | AUC unt. | F1 unt. | AUC PCA | AUC IF | F1 IF | AUC robust | F1 robust | AUC DSVDD | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Standardfall (Voreinstellung) | 0.74 | 0.34 | 1.00 | 0.98 | 0.43 | 1.00 | 0.98 | 1.00 | 0.85 | 0.94 | Training schadet |
| Ohne Training (0 Epochen) | 1.00 | 0.98 | 1.00 | 0.98 | 0.43 | 1.00 | 0.98 | 1.00 | 0.85 | 0.94 | ebenbürtig |
| Engpass 1 | 0.76 | 0.51 | 1.00 | 0.98 | 0.83 | 1.00 | 0.98 | 1.00 | 0.85 | 0.94 | Training schadet |
| Lineares Autoencoder = PCA | 0.43 | 0.00 | 1.00 | 0.98 | 0.43 | 1.00 | 0.98 | 1.00 | 0.85 | 0.94 | blind |
| Korrelationsbruch | 1.00 | 0.98 | 0.66 | 0.03 | 1.00 | 0.78 | 0.23 | 1.00 | 0.84 | 0.93 | Autoencoder besser |
| 40 Rauschmerkmale | 0.17 | 0.03 | 0.99 | 0.92 | 0.42 | 0.99 | 0.42 | 0.90 | 0.36 | 0.72 | blind |
| Kleine Stichprobe (50 Touren) | 0.14 | 0.00 | 1.00 | 0.89 | 0.38 | 1.00 | 0.83 | 0.87 | 0.42 | 0.76 | blind |

## Modell und Verfahren

- **Szenario** (`aeano_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke / Korrelationsbruch mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale). Die Sonderfahrten der Art "verstreut" sind Verschiebungen der **latenten Faktoren** – sie liegen in der Ebene der Normalen.
- **Autoencoder** (`aeano_algorithm.py`, numpy von Grund auf): Encoder p → Breite^Tiefe → Engpass k (linear), Decoder k → Breite^Tiefe → p (linearer Ausgang), tanh oder ReLU, mit Bias, Xavier bzw. He; mittlerer quadrierter Rekonstruktionsfehler, **Backpropagation von Hand** (gegen Finite Differenzen geprüft), Adam, Vollbatch, Momentaufnahmen inkl. Epoche 0.
  Wert = mittleres Fehlerquadrat je Tour (z-Einheiten), Schwelle = (1 − ν)-Quantil der Trainingsfehler. **PCA-Residuum** in geschlossener Form (Singulärwertzerlegung); **Hauptwinkel** zwischen Decoder-Unterraum und PCA-Unterraum.
- **Deep SVDD** (`aeano_dsvdd.py`), **One-Class SVM** (`aeano_ocsvm.py`), **Isolation Forest**, **LOF**, **ECOD** und **Wurzel** (`aeano_isolation_forest.py`, `aeano_lof.py`, `aeano_ecod.py`, `aeano_ee_algorithm.py`): wortgleich aus den Vorgängern.
- **Auswertung** (`aeano_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren und fünf Bezugsdetektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; Sweeps, Experiment-Tabellen (Epochenverlauf, Netz-Seeds, Engpass, lineares Autoencoder gegen PCA, Kosten), Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "Blind für Anomalien im Faktorraum" – **bestätigt, in der Ebene** (Standardfall AUC 0.68, PCA-Residuum 0.49, lineares Autoencoder 0.49); nichtlineare Netze sind nicht ganz so blind (0.68 statt 0.49), weil sie nach 300 Epochen noch nicht alles rekonstruieren.
  (2) "Stark beim Korrelationsbruch" – **bestätigt** (1.00, F1 0.98), gleichauf mit dem PCA-Residuum und der Wurzel. (3) "PCA-Residuum = lineares Autoencoder" – **bestätigt** (Winkel 0.03°, AUC 0.49 gegen 0.49); "nichtlineare Netze bringen bei gekrümmtem Normalbereich mehr" – **nicht bestätigt**: PCA-Residuum 0.99, Autoencoder 0.98.
  (4) "Training: die AUC kann fallen" – **bestätigt und stärker als bei Deep SVDD** (1.00 → 0.56 nach 1000 Epochen, je nach Rechner bis 0.67); das Ziel „gut rekonstruieren“ ist nicht das Ziel „Anomalien finden“. (5) "Kontamination: das Training lernt Anomalien mit" – bestätigt (0.85 bei 2 %, 0.63 bei 45 %). (6) "Dichte Gruppe mitgelernt, Lücke blind" – bestätigt (0.50 und 0.49).
  (7) "Rauschmerkmale: der Fehler wird von Rauschen dominiert" – **bestätigt, schärfer als gedacht**: **unter Raten** (0.25), 96 % des Fehlers aus den Rauschspalten. (8) "Netz-Seed-Streuung" – bestätigt (± 0.07 nach 300 Epochen). (9) "Kosten wie Deep SVDD" – bestätigt (das Eineinhalb- bis Zweieinhalbfache).
  Nicht vorhergesagt: die **AUC unter Raten bei wenigen Touren** (0.00 bei 20 Touren: die Sonderfahrten werden besser rekonstruiert als die normalen Touren) und dass ein **Engpass unter der latenten Dimension** (1) den Standardfall auf 0.88 hebt.
- **Das untrainierte Netz ist der ehrliche Bezugspunkt:** sein Fehler ist im Wesentlichen die Länge der standardisierten Tour, also der Abstand von der Mitte – gut für Anomalien weit außen (auch das Deep-SVDD-Ergebnis), nutzlos für den Korrelationsbruch (0.70) und die Lücke (0.06).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Rauschmerkmale sind unabhängige Spalten. Der Standardfall ist der ungünstigste Fall für den Autoencoder (Anomalien in der gelernten Ebene); reale Anomalien liegen meist dazwischen. Kleine Netze, Vollbatch, feste Lernrate 0.01; das Ergebnis gilt für dieses Szenario und diese Netzgrößen.
  Literatur nur mit Namen: Baldi und Hornik (lineare Autoencoder und PCA); Hinton und Salakhutdinov (Autoencoder).

## Verifikation

- Autoencoder: **Gradienten gegen Finite Differenzen** (tanh und ReLU × Tiefe 0–2 × Engpass 1 und 3: 12 Konfigurationen, Gewichte und Bias), Aktivierungsableitungen, Aufbau (Schichtgrößen, Parameterzahl, Engpass-Kode), Wert = mittleres Fehlerquadrat und Fehler je Merkmal summieren zum Wert, Schwelle = (1 − ν)-Quantil (markiert höchstens ν), Verlust fällt und 0 Epochen = untrainiert, Determinismus und Netz-Seed,
  ein trainiertes Netz findet Touren, die die Ebene verlassen; **PCA-Residuum** (Handinstanz mit Rang 1, gegen `numpy.linalg.svd`, Eckart-Young, monoton im Engpass); **lineares Autoencoder → PCA-Unterraum** (Hauptwinkel unter 8° nach 2000 Epochen, Fehler gleich dem PCA-Residuum); Hauptwinkel-Handinstanz.
- Übernommene Bausteine: One-Class SVM (gegen `sklearn.svm.OneClassSVM`), Deep SVDD (bias-frei, deterministisch, Schwelle), LOF, Isolation Forest, Wurzel-Schätzer (χ² gegen scipy), ECOD-Schwanzwahrscheinlichkeiten. Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten, Korrelationsbruch.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Epochen-, Netz-Seed-, Engpass-, Winkel-, Szenarien-, Schwellen- und Kostentabellen, dazu die Fehleranteile der Rauschspalten und der Sonderfahrten bei wenigen Touren; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; **weite Bänder für trainierte Netze** wegen numpy-/BLAS-Rundung, enge für das untrainierte Netz und das PCA-Residuum; Rechenzeiten nur als Größenordnung);
  alle 7 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, lineare und tiefe Netze × jede Art der Anomalien, Extremwerte aller Netzregler, Abspielen ohne doppelte Schlüssel, Netz-Seed-Knopf, ausgeblendete Regler behalten ihre Werte, Sweep-Optionen, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Epochenverlauf, Engpass, lineares Autoencoder gegen PCA, Netz-Seeds, Szenarien, Schwelle, Kosten), 🚧 Grenzen, Mathe |
| `aeano_algorithm.py` | Autoencoder (Netz, Backpropagation von Hand, Adam, Momentaufnahmen), PCA-Residuum, Hauptwinkel |
| `aeano_dsvdd.py`, `aeano_ocsvm.py`, `aeano_lof.py`, `aeano_isolation_forest.py`, `aeano_ee_algorithm.py`, `aeano_ecod.py` | Deep SVDD, One-Class SVM, LOF, Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD, ECOD (wortgleich aus den Vorgängern) |
| `aeano_scenario.py`, `aeano_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien (auch Korrelationsbruch) und Rauschmerkmalen; Konstanten, Presets |
| `aeano_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Urteil |
| `aeano_presets.py`, `aeano_visualization.py` | Permalink/Presets (ausgeblendete Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | Autoencoder (Finite-Differenzen-Gradienten, PCA-Residuum, lineares Autoencoder gegen PCA), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Anomalie-Erkennung: Ellipse bis Autoencoder](https://sebastianhanisch.net/konzepte-anomalie-erkennung.html).
