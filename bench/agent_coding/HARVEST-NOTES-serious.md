# Harvest-Notizen Klasse S (ernsthafte Aufgaben)

Stand: 2026-07-20. Es wurden ausschließlich Kandidaten für Klasse S gesammelt; keine Klasse-M-Kandidaten sind enthalten.

## Vorgehen und Filter

- Quelle: gemergte, issue-verlinkte PRs aus `pallets/flask`, `gohugoio/hugo`, `google/gson`, `colinhacks/zod`, `serde-rs/serde` und `tokio-rs/tokio`.
- Pro Repository wurden bis zu 800 zuletzt aktualisierte gemergte PRs gesichtet. Berücksichtigt wurden nur PRs mit `closingIssuesReferences`, genau einem PR-Commit und einem testrelevanten Diff.
- Bei Merge-Commit-Repositories (insbesondere Flask und Serde) ist der einzelne PR-Commit gegen dessen Parent eingetragen. Bei Squash-Merge-Repositories ist der Squash-Commit gegen dessen einzigen Parent eingetragen.
- Umfang und Dateiliste wurden lokal mit `git diff --numstat <parent> <commit>` neu bestimmt, nicht aus den PR-Zählern übernommen.
- Harte Filter: 80–800 geänderte Zeilen, 2–15 Dateien, Tests/Test-Fixtures/Test-Harness geändert, kein reiner Merge-, Format-, Vendor-, Generated- oder Dependency-Bump-Diff.
- Alle 40 Einträge besitzen eine echte schließende Issue-Referenz und daher `confidence: high`. Commit und Parent wurden lokal per `git cat-file` geprüft. Es wurden keine Abhängigkeiten installiert und keine Builds oder Tests ausgeführt.

## Ausbeute pro Typ und Repository

| Typ | Flask | Hugo | Gson | Zod | Serde | Tokio | Gesamt | Zeilen min/median/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feature-implementation | 2 | 2 | 2 | 1 | 1 | 2 | **10** | 85 / 140 / 523 |
| reported-bugfix | 2 | 3 | 1 | 3 | 2 | 1 | **12** | 83 / 177 / 430 |
| cross-cutting-change | 2 | 3 | 0 | 1 | 1 | 1 | **8** | 144 / 206 / 701 |
| refactor-mit-verhalten | 3 | 4 | 0 | 1 | 2 | 0 | **10** | 116 / 237 / 596 |
| **Gesamt** | **9** | **12** | **3** | **6** | **6** | **4** | **40** | **83 / 182 / 701** |

Damit liegt jeder S-Typ im Zielkorridor von 8–12 Kandidaten. Alle sechs Repositories sind vertreten; `reported-bugfix` und `feature-implementation` decken jeweils alle sechs beziehungsweise fast alle Repositories breit ab.

## Einordnung der Typen

- **feature-implementation:** neue, nutzerseitig beschriebene Fähigkeiten mit API-/Modul- und Teständerungen, zum Beispiel Form-Limits in Flask, Map-Größenregeln in Zod oder Unix-Pipe-I/O in Tokio.
- **reported-bugfix:** konkrete Reproduktionen oder beobachtete Fehlverhalten; die Issues benennen das erwartete Verhalten, nicht die Patch-Mechanik.
- **cross-cutting-change:** Änderungen, die mehrere öffentliche oder interne Flächen konsistent betreffen, etwa CLI-Ausgabekanäle, Konfigurationsmodelle, Derive-Pfade oder Scheduler-Metriken.
- **refactor-mit-verhalten:** strukturelle Umbauten mit Folgeänderungen und verhaltenssichernden Tests, etwa expliziter Kontextfluss in Flask, skalierbarere Pfadgruppierung in Hugo oder robustere Codegenerierung in Serde.

## Dünne Stellen und Ursachen

### Gson

Nur drei Kandidaten bestanden alle S-Filter. Viele issue-verlinkte Gson-PRs im betrachteten Fenster waren Dokumentations-/Workflow-Arbeit, lagen unter 80 Zeilen, änderten nur eine Datei oder enthielten keinen geänderten Verhaltenstest. Die brauchbaren Kandidaten sind deshalb auf Features und einen Bugfix konzentriert; belastbare Cross-Cutting- oder Refactor-Kandidaten fehlen.

### Zod

Zod liefert gute fokussierte Features und Bugfixes. Nur ein Refactor-Kandidat bestand alle Filter: die Korrektur der Shape-Auswertung bei rekursiven Schemas. Weitere Refactor-PRs waren häufig nicht issue-schließend, unterhalb der Umfangsschwelle oder mit mehreren Commits verteilt.

### Cross-Cutting allgemein

Mit acht Kandidaten ist der Typ ausreichend, aber am unteren Zielrand. Größere API-Migrationen überschritten oft 15 Dateien oder 800 Zeilen; kleinere konsistente Änderungen lagen dagegen unter 80 Zeilen. Diese Klasse ist dadurch stärker von Hugo und Flask getragen.

### Refactor-mit-Verhalten allgemein

Echte Refactors ändern häufig keine Tests, obwohl vorhandene Tests das Verhalten absichern. Solche PRs wurden wegen des harten Kriteriums „Tests geändert“ verworfen. Andere Refactors bestanden überwiegend aus generierten Snapshots oder Formatänderungen. Die verbleibenden zehn Kandidaten haben jeweils echte Teständerungen und einen issue-beschriebenen Verhaltens- oder Performancegrund.

### Hugo-Anteil

Hugo stellt 12 von 40 Kandidaten. Ursache ist die ungewöhnlich gute Kombination aus issue-schließenden Squash-Commits, integrierten Go-Tests und ernsthaften Änderungen im Zielumfang. Bei der späteren Bankauswahl sollte man diesen Anteil gegebenenfalls zugunsten der kleineren Repos reduzieren, ohne die dünnen Typen zu entleeren.

## Beobachtungen zur Testbarkeit

| Repository | Fokussierbarkeit | Beobachtung für Stufe 2 |
|---|---|---|
| Flask | sehr gut | Pytest-Dateien und einzelne Testfunktionen lassen sich direkt adressieren. Kandidaten verteilen sich auf `tests/test_*.py`; Setup benötigt die Python-Testabhängigkeiten des jeweiligen Pins. |
| Hugo | gut bis mittel | Pakettests über `go test ./pfad` sind gut fokussierbar. Viele Verhaltenstests sind Integrationstests innerhalb eines Pakets und per `-run` auswählbar; `hugolib`- und Template-Integration kann trotzdem merklich länger dauern. |
| Gson | sehr gut | Maven/Surefire kann auf Testklasse und häufig einzelne Methode eingeschränkt werden (`-pl gson -Dtest=Klasse#Methode`). Die drei Diffs haben jeweils eine klar zugehörige Testklasse. |
| Zod | gut | Vitest lässt sich auf eine Datei und einen Testnamen begrenzen. Stufe 2 muss den passenden Workspace-/Paketpfad am historischen Parent beachten; Installation über den gepinnten Lockfile-Stand kann den Großteil der Setup-Zeit ausmachen. |
| Serde | mittel | Normale Tests sind per Cargo-Paket, Test-Binary und Testname fokussierbar. UI-/Compile-Fail-Fälle (`.stderr`) laufen über die jeweilige UI-Suite und sind weniger fein isolierbar; historische Toolchain-Kompatibilität muss geprüft werden. |
| Tokio | gut bis mittel | Paket- und Integrationstests sind gut adressierbar. Compile-Fail-Tests in `tests-build`, Loom-Tests und Scheduler-Tests können zusätzliche Features/Cfgs oder längere Laufzeiten brauchen; der Scheduler-Kandidat #8065 sollte früh auf Laufzeit und Diskriminationskraft geprüft werden. |

## Offene Bedenken für Validierung und Bau

1. **Parent-Fail-Nachweis fehlt noch:** Harvest prüft nur Historie, Umfang und Teständerungen. Stufe 2 muss für jeden Kandidaten den neuen/angepassten Test auf dem Parent anwenden und den erwarteten Fehlschlag reproduzieren.
2. **Historische Toolchains:** Ältere Serde-, Zod- und Gson-Parents können andere Rust-/Node-/Java-Versionen benötigen als die aktuellen Pins aus `tasks_v1.json`.
3. **Performance-Issues:** Hugo #14211 sowie Tokio #8065 sind testbar, aber ein funktionaler Test allein beweist möglicherweise nicht die ursprüngliche Performanceaussage. Die geänderten Repository-Tests sind vorhanden; die spätere Auswahl sollte prüfen, ob sie einen falschen Agent-Patch ausreichend diskriminieren.
4. **Issue-Texte mit API-Wünschen:** Einige echte Issues nennen gewünschte öffentliche Namen oder Konfigurationsformen. Das ist legitime Anforderung, nicht Patch-Leak; die in JSONL gespeicherten Excerpts und Intents vermeiden dennoch interne Implementierungsdetails.
5. **Hugo QR und Abhängigkeit:** Hugo #13205 fügt für die neue Funktion eine Bibliothek samt `go.mod`/`go.sum`-Änderung hinzu. Es ist kein Dependency-Bump-Task, sondern eine echte Feature-Implementierung; Stufe 2 sollte trotzdem sicherstellen, dass der Bench-Setup die historische Modulauflösung reproduzierbar erlaubt.
