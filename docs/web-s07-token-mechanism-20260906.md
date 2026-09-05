# S07: Tokenkosten entlang der öffentlichen Werkzeugfolge

Die zehn eingefrorenen S07-Traces wurden erneut ausschließlich nach öffentlichen
Tool-Aufrufen und Provider-Zählern ausgewertet. Alle zehn Summen stimmen exakt
mit den bisherigen trial.json-Werten überein. Wiederholte kumulative Zähler
werden nicht doppelt gezählt; fehlende, zurückgesetzte oder inkonsistente Zähler
verweigern die Zuordnung. Sechs gezielte Tests bestehen.

Evidenz: `/Users/michaelwelsch/.local/state/greppy-web-study/s07-usage-timelines-20260906-01`.
Reproduktion: `usage_timeline.py TRACE OUTPUT` mit einem begrenzten Turn-Export.
Das Skript liest keine privaten Modellgedanken. Mehrere öffentliche Aufrufe in
einer Modellantwort erhalten gemeinsam deren Kosten; eine Aufteilung wird nicht erfunden.

## Konkreter Mechanismus

A1 braucht 14 Modellantworten, C1 24. Beide beginnen mit nahezu gleichem Input:
36.298 bzw. 36.324 Tokens. A1 verarbeitet pro Antwort bis zu 58.240 Input-Tokens,
C1 bis zu 46.324. Trotzdem kostet C1 insgesamt mehr: 1.007.883 statt 705.573
Input-Tokens und 2.728 statt 1.987 Output-Tokens. Die größere Zahl von
Modellantworten überwiegt hier den kleineren Kontext je Antwort.

Die lückenlos zugeordneten Abschnitte von C1:

| Öffentliche Folge | Modellantworten | Input-Tokens | Output-Tokens |
|---|---:|---:|---:|
| Öffnen und erste Auswahlkette | 2 | 73.487 | 333 |
| Erneuter Select, Inspect, DOM, Hilfe, korrigierter Select | 7 | 275.220 | 666 |
| Neuladen, Dialog öffnen, Menge und Absenden | 3 | 125.133 | 494 |
| Wiederholtes Confirm, Events, Diagnose und Sitzung neu öffnen | 9 | 396.059 | 901 |
| Abschließendes Neuladen | 1 | 45.518 | 92 |
| Nicht verlangter Statusbericht an Fix-Task und Abschluss | 2 | 92.466 | 242 |

Die Zahlen sind tatsächlich verbrauchte Tokens während dieser Folgen, keine
kausalen Fehleranteile und keine vorhergesagten Einsparungen. Insbesondere
enthalten die Folgen sowohl Produktfehler als auch Agenten-/Transportabweichungen.
Der letzte Abschnitt enthält die regulär notwendige Abschlussantwort; seine
Kosten sind daher ebenfalls nicht vollständig vermeidbar.

## Konsequenz für die nächste Messung

Die kompaktere Darstellung allein genügt nicht. Der nächste Kandidat muss die
zusätzlichen Entscheidungen vermeiden helfen: ausführbare Auswahlwerte direkt
in der Beobachtung, strikte Select-Ergebnisse, verlässliche Ergebnisbedingungen
nach dem Absenden und erhaltene Sitzungen nach Fehlern. In der Wiederholung wird
gezählt, ob diese konkreten Umwege verschwinden und ob die gesamten Provider-
Input- UND Output-Tokens sinken. Zwischengespeicherter Input bleibt Input und
wird nicht aus dem Abnahmekriterium herausgerechnet.

S07 bleibt wegen des bereits dokumentierten Isolationsfehlers und der zwei
unvollständigen Standard-Läufe ungeeignet als Abnahmenachweis. Die neuen
Auswertungen ändern keine alten Läufe und behaupten keinen Effizienzgewinn.
