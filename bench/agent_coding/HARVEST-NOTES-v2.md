# HARVEST-NOTES v2

Geerntet wurden **90** echte, nicht gemergte Commits aus den sechs Ziel-Repositories. Es wurde nichts installiert oder gebaut; geprüft wurden Historie, Diff, Dateistatistik und Testberührung.

## Ausbeute pro Kategorie und Repository

| Kategorie | flask | hugo | gson | zod | serde | tokio | Summe |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-mit-callsite | 2 | 4 | 1 | 0 | 1 | 2 | 10 |
| feature-parameter | 4 | 0 | 1 | 2 | 2 | 3 | 12 |
| rename-cleanup | 0 | 5 | 0 | 0 | 6 | 1 | 12 |
| api-migration | 2 | 5 | 1 | 0 | 0 | 4 | 12 |
| guard/validierung | 2 | 5 | 2 | 3 | 0 | 0 | 12 |
| config-und-code | 0 | 10 | 0 | 0 | 0 | 0 | 10 |
| import-und-nutzung | 5 | 0 | 3 | 0 | 1 | 3 | 12 |
| test-ergänzen | 0 | 4 | 2 | 1 | 3 | 0 | 10 |

## Einordnung

- **config-und-code ist die dünnste und am stärksten konzentrierte Kategorie.** Die harten Kriterien „JSON/TOML/YAML plus lesender Code plus Tests im selben kleinen Commit“ traten in den anderen fünf Repositories kaum sauber auf. Die 10 Kandidaten stammen deshalb aus Hugo; mehrere koppeln die kanonischen Konfigurationsdaten mit Config-Strukturen, Laufzeitverdrahtung und Integrationstests. Diese Kandidaten sind vorsorglich mit `medium` markiert und sollten in Stufe 2 besonders auf echte Parent-Test-Entscheidung geprüft werden.
- **test-ergänzen wird durch die Source-Datei-Pflicht künstlich verengt.** Reine Test-Commits ohne Produktionsdatei wurden trotz guter Eignung verworfen. Die verbliebenen Kandidaten ergänzen Tests und berühren Produktionscode meist nur minimal oder zur Erreichbarkeit des getesteten Pfads.
- **rename-cleanup und api-migration liegen häufiger in älteren Historien.** Dort sind die Diffs klar, klein und testberührt, aber teils auf historische APIs oder frühere Repository-Strukturen bezogen.
- **import-und-nutzung wurde streng am Diff geprüft.** Aufgenommen wurden nur Commits, die in einer Quelldatei einen Import ergänzen und die importierte Abstraktion im selben Commit verwenden; reine Import-Sortierung und test-only Imports wurden verworfen.
- **Vorschlag bei weiterer Ausdünnung von config-und-code:** Entweder Test-Fixture-Konfiguration als zulässige Config-Datei ausdrücklich bestätigen oder diese Kategorie repository-übergreifend um Projekte mit handgepflegten TOML/YAML-Laufzeitkonfigurationen erweitern. Package-Manifeste allein sollten nicht als Ersatz gelten.

## Selbstprüfung

- Alle Einträge erfüllen: kein Merge, 1–5 Quelldateien, mindestens eine berührte Testdatei und höchstens 150 geänderte Zeilen.
- Generated-/Vendor-/reine Format-Commits wurden verworfen; insbesondere wurde ein Kandidat mit gebündeltem JavaScript nach Diff-Sichtung entfernt.
- Alle `intent`-Sätze beschreiben nur Zielverhalten und nennen weder konkrete Lösungsbezeichner noch geänderte Werte.
- Commit und Parent werden aus den lokalen Vollklonen referenziert; die JSONL wird nach dem Schreiben vollständig geparst und alle Objekte werden per Git geprüft.
