# S07: Kosten komplexerer Interaktionen und Grenzen des Versuchsaufbaus

Stand 5. September 2026: Alle zehn Teilnehmer sind beendet. Greppy erfüllt das
Effizienzziel weiterhin nicht. S07 ist wegen eines nachgewiesenen Isolationsfehlers
im Studienaufbau kein valider Abnahmevergleich. Die konkreten Produktfehler und
Wiederherstellungsfolgen bleiben verwertbare Entwicklungsbefunde.

## Eingefrorene Bedingungen und Ergebnisse

Aufgabe: EU und mindestens drei verfügbare Einheiten filtern, aufsteigend nach
Stückpreis sortieren, drei Einheiten des günstigsten passenden Artikels im
Dialog reservieren und nach Neuladen die Persistenz prüfen. Fünf gepaarte Seeds,
wechselnde Reihenfolge, jeweils Codex Luna/medium ohne übernommene Gesprächshistorie.
A verwendet den aktuellen Browser-Pluginpfad; C verwendet Greppy Web als Tool.
Der eigentliche Greppy-Agent B ist nicht Bestandteil dieser Serie.

| Paar | A Input / Output / Aufrufe | C Input / Output / Aufrufe | Unabhängige Ergebnisprüfung A / C |
|---|---:|---:|---|
| 1 | 705573 / 1987 / 13 | 1007883 / 2728 / 23 | bestanden / bestanden |
| 2 | 722612 / 1881 / 12 | 644641 / 2156 / 15 | bestanden / bestanden |
| 3 | 485458 / 1703 / 9 | 613539 / 2645 / 14 | unvollständig / bestanden |
| 4 | 702802 / 1692 / 13 | 1448612 / 3482 / 32 | bestanden / bestanden |
| 5 | 479494 / 1542 / 9 | 742112 / 1488 / 17 | unvollständig / bestanden |

Deskriptiver Median der fünf gepaarten Änderungen: **+42,846 % Input-Tokens und
+37,292 % Output-Tokens** für C. Diese Zahlen enthalten ausdrücklich auch die
unvollständigen A-Läufe und den kontaminierten C4-Lauf; sie belegen keinen kausalen
Werkzeugvergleich. Kein Fehlerlauf wird als schneller Erfolg gewertet oder aus
der Tabelle entfernt. Provider-Tokens einschließlich Fehlerbehandlung und
Dokumentationsabrufen werden verwendet; Bytes werden nicht als Tokens ausgegeben.

Die Mediane pro Arm sind A 702802 Input / 1703 Output / 12 Aufrufe und C 742112 /
2645 / 17. Der Quotient dieser Arm-Mediane ist nicht der Median der Paarquotienten.
Die unabhängige Prüfung bestätigt Reservierungsartikel, Menge, Filter, Sortierung
und Bestandswirkung; die öffentlichen Werkzeugspuren enthalten das Neuladen.
Kontrollierte End-to-End-Zeiten fehlen. Ein separat koordinierter Android-Build
überlappte Teile der Serie. Keine p95-, Geschwindigkeits- oder vollständige
Zwölf-Fälle-Abnahme wird daraus abgeleitet.

Evidence: /Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260905-07
mit plan.json, prepared-dispatches, spawn-receipts, trials, summary.json,
terminal.json, external-load-notice.json und routing-audit-10.json.
Plan-SHA256: ade801e57de2fd628599d52a286f9c899564b291b2d09acfb2639188f8327e24.

CLI: /Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-inspect-154d1a775f41/greppy
SHA256 154d1a775f4156c6d33742ac309e063a49563a57e16cfb6a92c97a20a8082471.
Runtime: /Volumes/tmp/dev-artifacts/greppy/e1-label-native-candidate.THEPk0/preserved-bin/web-runtime
SHA256 3ee1f2e17a697b29fac40791028228d085b819af9c4d2a2042b34e877b3cdfb8.
Die ausführbaren Dateien wurden nach jedem C-Lauf gegengeprüft und nicht verändert.
Die CLI enthält die funktional geprüfte Inspect-/Tab-/Condition-Integration;
sie enthält weder den neuen nativen Wait noch den jetzt gefundenen Select-Fix.

## Was die öffentlichen Traces konkret zeigen

1. **Unbekannte Auswahlwerte werden als erfolgreiche Aktion angenommen.** C1–C4
   versuchen zunächst low/asc bzw. ein sichtbares Label als Wert. Greppy zeigt
   danach value leer und selected_options leer, aber receipt.ok=true und eine
   ausgeführte Chain. C1: call_bIyyQLX7v1Cz494sswpUgv4n und
   call_GLNK2Hmfy1kLSZ5OJpsP989f. Das ist durch eine separate native Negativprobe
   des Fix-Workers bestätigt: ungültiger Wert führt zu Exit 0, geleerter Auswahl,
   input/change-Ereignissen und ausgeführtem Folgeklick. Native Belege:
   /Volumes/tmp/dev-artifacts/greppy/native-select-repro.OxqBgA/receipt.json.
   Die Capture-Probe selbst endet erfolgreich, weil sie den Fehler dokumentiert;
   das ist kein bestandener Produkttest. Korrektur gehört in den gemeinsamen
   Content-Worker-Pfad von CLI Select und Playwright selectOption.

2. **Die Beobachtung lässt die nötigen Auswahlwerte weg.** Verklebte Labels und
   nur die bereits ausgewählte Option reichen für den nächsten Select-Aufruf
   nicht aus. Inspect liefert zwar den Knoten, aber keine vollständigen
   label/value-Paare. C2 geht von Inspect über zwei ungültige DOM-Aufrufe zu
   einer JavaScript-Abfrage der Optionen; C1 und C3 nehmen ähnliche Umwege.
   Kleine Auswahlmengen sollen begrenzt und direkt ausführbar ausgegeben werden.
   Vereinbart ist select_choices mit greppy.web.select-choices.v1, höchstens
   acht Einträgen, Gesamtzahl, deaktiviertem Zustand und ausdrücklichen
   Trunkierungsmarkierungen. Ein gekürzter Wert darf niemals als gültiger Wert
   erscheinen. Der gemeinsame Projektionshelfer eaf4768f ist vorbereitet und
   mit sieben Node-Tests geprüft, aber noch nicht nativ angebunden.

3. **Nach dem Absenden erscheint noch der vorherige Dialogzustand.** C1, C2
   und C4 wiederholen daraufhin den Confirm-Klick. Spätere Antworten zeigen
   TIMEOUT/unsichtbares Element; die Reservierung war inzwischen gespeichert.
   C4 wechselt zusätzlich zu Inspect, Screenshot, Find und Sitzungsdiagnosen.
   Dies spricht für die Priorität von Aktion plus ausdrücklicher Ergebnisbedingung
   und ereignisgestütztem Warten. Aus einer zugestellten Eingabe darf kein
   fachlicher Erfolg abgeleitet werden. Einzelnen Ursachen wird hier kein
   erfundener Anteil der Token-Kosten zugerechnet.

4. **Transportfehler verschärfen die Wiederherstellung.** C1/C2 und zunächst C4
   geben entgegen der vorbereiteten Anweisung nur r.output weiter. Bei länger
   laufenden Shell-Aufrufen geht damit der session_id-Handle verloren. C4 versucht
   einen nicht belegten Handle und korrigiert die Weitergabe erst später. C3
   prüft den Handle, pollt aber nur einmal. Das sind Agenten-/Transportabweichungen
   trotz vorhandener konkreter Anleitung; keine Belege für verschwundene
   Runtime-Antworten. Ihre tatsächlichen Kosten bleiben vollständig enthalten.

5. **Der Human-Renderer dupliziert die Inspect-Nutzdaten.** Die Antwort enthält
   sowohl serialized mit k/v/o/s/n-Transportstruktur als auch denselben Knoten
   unter value. C1 call_XAWl5s65acn7hFcVw5ZlrarI belegt das. Die kompakte Darstellung
   gehört Root; JSON-Vertrag, Fehler, Zustände und unbekannte Felder müssen
   erhalten bleiben. Diese Formatkorrektur allein löst die zusätzlichen
   Modellschritte aus den vorherigen Punkten nicht.

6. **Befehlsform und Recovery bleiben Stolperstellen.** DOM ohne Unterbefehl,
   Verkettung ohne web do, bare CSS bei einer Aktion und session close ohne
   positional ID erzeugen weitere Aufrufe. Teilweise ist die ausgegebene
   Korrekturanweisung bereits klar. Solche Fälle werden nicht pauschal als
   funktionale Produktfehler klassifiziert; inkonsistente Befehlsformen und
   fehlende konkrete Recovery-Kommandos sind gesonderte Usability-Kandidaten.

A ist ebenfalls nicht fehlerfrei: A1 rät asc als Optionswert und muss nachsehen.
A2 lädt die vollständige Browser-Dokumentation nach einer gekürzten Ausgabe
noch einmal und schickt eine unnötige Bestätigungsfrage an den Bugfix-Task,
beendet den Ablauf anschließend aber selbst. A3 und A5 stoppen vor der
synthetischen Reservierung aufgrund einer fälschlich angenommenen
Bestätigungspflicht. Diese Abweichungen werden nicht Greppy zugerechnet.
Es werden öffentliche Aufrufe, Antworten und Provider-Zähler ausgewertet;
private Modellgedanken oder ein innerer Vertrauenszustand werden nicht behauptet.

## Fehler des Studienaufbaus und Korrektur

Die C-Aliase hatten eigene Arbeitsverzeichnisse, teilten aber den Runtime-Owner
study-table-s07-integrated. C4 listet Sitzungen und versucht observe/close auf
wrs_fe718d28746901d93101, der bereits C3 gehörte. Das ist kein bloß zufällig
wiederholter String: Der öffentliche Request nennt die vorherige Sitzung.
Die konservative ID-Prüfung meldet entsprechend fresh_vs_prior=false.
Getrennte Testdaten und ein erfolgreicher Oracle-Check heilen diesen Verstoß
gegen die geplante Isolation nicht.

Neue Kontextvorbereitungen erzeugen deshalb einen eigenen Runtime-Owner aus
Gruppen- und Teilnehmeridentität. Der geerbte Owner wird im ausführbaren Alias
überschrieben. Das Auswertungsgate verweigert einen Token-Pass bei fehlenden
oder wiederverwendeten Sitzungsbelegen. Die relevante Python-Prüfung besteht
38 Tests; darunter tatsächlich ausgeführte Alias-Prozesse mit gleichem
Gruppennamen und unterschiedlichem Owner sowie verweigerte vermeintliche
Token-Siege bei fehlender/negativer Isolation. Das ist noch kein vollständiger
nativer Isolationsnachweis. Neue Owner verändern Kalt-/Warmstartbedingungen;
eine neue Serie muss das ausdrücklich registrieren.

Für künftige synthetische Serien benennt browser_plugin_synthetic_v3 in beiden
Armen ausdrücklich den Testcharakter ohne Bestellung, Zahlung oder Vertrag.
S07-Prompts, Wrapper und Fehlerläufe wurden nicht rückwirkend verändert.

Der Host speichert Task-Payloads sowohl im Parent als auch im Child verschlüsselt.
Die Routing-Prüfung bestätigt identische verschlüsselte Payloads, Empfänger,
Modell/Effort, Plan-Hashes und die vorbereiteten Klartext-Hashes. Sie behauptet
keine unabhängige Klartext-zu-Ciphertext-Prüfung oder Entschlüsselung. Frühere
fehlgeschlagene Prüfversuche mit einer falschen Klartext-Annahme bleiben erhalten.

Alle Produktbefunde wurden ausschließlich an den festen Bugfix-Task
01a02118-0d61-7e10-a9d4-be496fa34879 gemeldet. Gefordert sind konkrete
Regressionstests und Rückmeldung mit geprüften Kandidaten. Kein weiteres
unverändertes S07-Experiment wird als Lösung präsentiert; zuerst müssen die
gefundenen Aktions- und Informationsfehler behoben und die neue Isolation
nativ geprüft werden. Die Gesamtaufgabe bleibt offen.
