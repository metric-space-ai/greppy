# S05: Warum Greppy trotz kleinerer Antworten mehr Tokens benötigt

Stand: 2026-09-05. Ergebnis: **Effizienzabnahme fehlgeschlagen.**

## Messung

Fünf gepaarte Wiederholungen derselben Checkbox-Aufgabe; beide Arme verwenden
gpt-5.6-luna mit medium. A: Codex mit Standard-CUA. C: Codex mit Greppy Web.
Alle zehn unabhängigen Ergebnisprüfungen bestanden. Dies ist ein Entwicklungsfall,
keine vollständige Abnahme. Der tatsächliche Greppy-Agent (B) fehlt hier.
Kontrollierte Gesamtzeit ist nicht nachgewiesen.

| Metrik | A: Median | C: Median | Median der gepaarten Veränderungen |
|---|---:|---:|---:|
| Provider-Input-Tokens | 365858 | 442651 | +19,56 % |
| Provider-Output-Tokens | 620 | 1115 | +118,39 % |
| Tool-Aufrufe | 7 | 10 | — |

Der Median gepaarter Veränderungen ist nicht das Verhältnis der Arm-Mediane.
Alle fünf C-Läufe verbrauchten mehr Output-Tokens als ihr A-Gegenstück.
Input enthält erneut verarbeiteten, auch gecachten Kontext. Cacheeinsparungen
heben das verfehlte Input-/Output-Ziel nicht auf.

Quelle: [unveränderte Zusammenfassung](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/summary.json).
Plan-SHA256: c6080f24a25d470d5187ca29e3eb4037476e0fbdb6d9e88b1ec3f8d4285095ff.
Gemessenes CLI: 2572b96ff6bd9339e2ff96cc099a2bee10f07058;
Binär-SHA256: 555cd58f17c857607d3852183280dbba790a307b5562d7a9025777f271908137.
Spätere Quelländerungen sind kein Bestandteil dieser Messung.

## Nachgewiesene Schleifen und Ursachen

Ausgewertet wurden öffentliche Tool-Aufrufe/-Resultate sowie Providerzähler.
Es wird kein privates Reasoning rekonstruiert. Wiederholungen und wechselnde
Aufrufversuche sind beobachtbar; ein innerer Vertrauenszustand ist daraus nicht
direkt bewiesen. Die prozentualen Beiträge einzelner Ursachen sind ohne Ablation
noch nicht bestimmt.

1. **Unzureichende CLI-Einweisung und teure Fehlererholung.**
   Die von mir vorbereitete Alias-Einweisung erklärte „greppy web syntax“,
   zeigte aber keinen konkreten Aufruf. C1/C2/C4 lassen zeitweise den
   Web-Namensraum weg; C1/C2 quoten ganze Befehle als einzelnes Argument.
   C2 wechselt durch Hilfe, Status-, Session- und Syntaxversuche und benötigt
   27 Aufrufe. Die vollständige Web-Hilfe enthält die benötigten Befehle;
   die Behauptung, open fehle dort, wäre falsch.
   C3 startet dagegen direkt mit korrektem open und verliert dennoch beim
   Output. Onboarding erklärt somit nicht den gesamten Nachteil.

2. **Post-action-Zustand erlaubt den nächsten Arbeitsschritt noch nicht.**
   Nach Aktivierung zeigt Greppy checked=true, aber quantity disabled=true
   und revision 0. Die Anwendung bestätigt den Zustand asynchron und ersetzt
   anschließend die Bedienelemente. Zusätzliche wait/observe-Aufrufe folgen.
   Das ist nicht exklusiv ein Greppy-Phänomen: A2/A3/A4 sehen ebenfalls den
   Zwischenzustand. Greppy muss diese Aufgaben trotzdem effizienter lösen;
   eine zugestellte Eingabe darf nicht als abgeschlossene fachliche Wirkung
   ausgegeben werden.

3. **Ungültige Bedingung wird als langer erfolgloser Wait behandelt.**
   C3: `web wait 'time=500ms'`, call_IRW1SJq73DJYDEFlUdHKobWO,
   läuft etwa 10039 ms und endet mit Exit 13. Der nicht unterstützte Query-Typ
   sollte vor Session/RPC/Polling mit konkreter Syntaxdiagnose scheitern.
   Im geprüften Quellstand fehlte die Query-Validierung auf dem wait/assert-Pfad.
   C1s `text=Quantity:` ist dagegen ein legitimer Nichttreffer: text= vergleicht
   den normalisierten gesamten Elementtext; Quantity: ist hier ein einzelner
   Textknoten neben Eingaben. Das bisher unklare Match-Verhalten ist ein
   Erklärungs-/Diagnostikproblem, kein nachgewiesener verlorener DOM-Text.

4. **Erfolgreicher Wait liefert keine aktualisierten Referenzen.**
   C2: call_IVrnV6TKgRTMRVLmxDsPNiPa meldet held=true, aber keinen neuen
   Seitenzustand. Das anschließende fill auf @2 scheitert mit STALE_REF
   (call_6JFJHgePLFYEJg3In5VF4EII); observe muss neue Referenzen beschaffen.
   Die Ablehnung des ersetzten Knotens ist korrekt. Der Optimierungsbedarf
   besteht in nutzbarer aktueller Rückmeldung, nicht im stillen Umbinden
   alter Referenzen auf andere Elemente.

5. **Bündelung allein beseitigt Abhängigkeiten nicht.**
   C4 quotiert zunächst die gesamte do-Kette und bekommt eine Syntaxablehnung.
   Die anschließend korrekt geschriebene Kette
   `web do check @1001 :: fill @1002 3`
   (call_g6tVrwjnpBKWrJMPhFLpX8FC) führt check aus und stoppt bei fill mit
   STALE_REF, Exit 34. Der vorherige Erfolg wird korrekt nicht zurückgerollt.
   Die Kette kennt die asynchrone Bereitschaft des abhängigen Feldes nicht.

6. **Zusätzliche Protokollarbeit im generierten Output.**
   C5 schreibt bei jedem Shell-Aufruf eigenen Code für output, exit_code
   und session_id, statt das komplette Ergebnisobjekt direkt weiterzugeben.
   Die wesentlichen Transportzustände bleiben erhalten; die generierten
   Aufrufe werden aber länger. Dies ist ein weiterer konkreter Ansatzpunkt
   für eine kurze, eindeutige Werkzeug-Einweisung.

## Warum kompakte Antworten nicht genügen

C-Antworten sind in allen fünf Paaren wesentlich kleiner als A-Antworten,
gemessen als UTF-8-JSON-Bytes. Trotzdem entstehen mehr Provider-Tokens.
Diese Bytewerte sind keine Tokenmessung.

A verwendet 6/8/9/8/7 Modellantworten, C 12/28/10/11/9. Jede weitere Antwort
verarbeitet in dieser Umgebung erneut etwa 37000–51000 Input-Tokens;
die ersten Requests liegen in beiden Armen bei etwa 36900 Tokens.
Mehr Entscheidungen erzeugen außerdem weitere Befehle und sonstigen
Modelloutput. Die öffentlichen Traces belegen damit einen konkreten
Schleifenmechanismus. Sie liefern noch keine saubere kausale Aufteilung
des Tokenunterschieds in Engine, CLI, Anleitung und Hostumgebung.

Auch A ist nicht idealisiert: Alle fünf A-Läufe versuchen zunächst
visible:true und bekommen „IAB visibility is not supported in a subagent
thread“. A4/A5 haben zusätzlich einen undefinierten Tab-Bezeichner.
Diese Fehler bleiben in der Messung; zukünftige Einweisungsverbesserungen
müssen für beide Arme prospektiv und unverändert eingefroren werden.

## Konsequenz und aktueller Arbeitsstand

Priorität haben verständliche Aufrufe, sofortige präzise Diagnosen,
eine Aktion plus relevante erfüllte Bedingung und aktuelle handlungsfähige
Referenzen. Ein zweites autonomes LLM oder semantische Suche sind durch
diese Fälle nicht als Lösung begründet.

Die Findings wurden an die bestehende Aufgabe „greppy bug fixing“
(01a02118-0d61-7e10-a9d4-be496fa34879) gemeldet.
Root bearbeitet nach expliziter Abstimmung CLI-Validierung und Erklärung
in expect.rs/see.rs; der Worker bearbeitet native Zustands-/Fehlerpfade.
Die CLI-Validierung ist inzwischen in Commit d00c675f implementiert und mit
13 Unit- sowie drei Integrationstests und separaten nativen Abfragen geprüft.
Sie wurde nicht in die unveränderten S05-/S06-Messkandidaten aufgenommen.
Diese Funktionsprüfung belegt noch keinen geringeren Tokenverbrauch;
ein nativer Wait mit aktuellen Referenzen ist weiterhin separat abzusichern.

Die nächste Untersuchung muss Verbesserungen getrennt prüfen und erneut
tatsächliche Provider-Tokens einschließlich aller Fehlversuche erfassen.
Weder kleinere Ausgaben noch bestandene Funktionstests ergeben bereits
eine Effizienzfreigabe.

## Öffentliche Trace-Belege

- [C1: Syntax, Zwischenzustand, Wait, zusätzliche Beobachtungen](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/trials/02-checkbox-1-C/turn-01a07290-06ca-7e70-be35-7ef1e3d6104e.metadata.json)
- [C2: 27 Aufrufe und erfolgreiche Bedingung ohne neue Referenzen](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/trials/03-checkbox-2-C/turn-01a07295-886f-7762-914a-f49f97347dbf.metadata.json)
- [C3: ungültiger time-Query mit zehn Sekunden Polling](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/trials/06-checkbox-3-C/turn-01a0729b-6954-71d3-bfef-1c239e02fb56.metadata.json)
- [C4: Kette stoppt an ersetztem Knoten](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/trials/07-checkbox-4-C/turn-01a0729d-abdf-72c2-a300-58f56dd78524.metadata.json)
- [C5: wiederholter Transportcode](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-05/trials/10-checkbox-5-C/turn-01a072a3-9cf9-78f3-a8a2-98ef6afc06e0.metadata.json)

Alle fünf A-Traces sind ebenfalls über summary.json verknüpft und wurden
bei dieser Auswertung auf ihre öffentlichen Aufrufe und Resultate geprüft.

