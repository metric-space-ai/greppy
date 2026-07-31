# Plan: V3 als Benchmark für normale Coding-Aufgaben

Status: Planungsentwurf, noch kein freigegebener Corpus
Zielmaschine: `gpu3`
Aktive Daten: `/mnt/nvme1/greppy-bench-v3`
Versiegelte Artefakte: `/mnt/asustor/LLM-Store/greppy-bench-v3`

## Ziel

Der Release-Benchmark soll messen, ob ein Coding-Agent mit der ausgelieferten
Greppy-0.3.0-Anleitung normale Aufgaben in unbekannten mittelgroßen bis sehr
großen Repositories bei gleicher Correctness mit weniger Gesamtkosten löst.

Die Aufgaben werden nicht für einzelne Greppy-Kommandos, Graphmuster,
Dateizahlen oder Patchgrößen ausgewählt. Greppy-Nutzung und Task-Eigenschaften
werden erst nach Auswahl und Versiegelung ausgewertet.

## Nicht-Ziele

- Kein Microbenchmark für `brief`, `impact`, `path`, Read- oder Edit-Verben.
- Keine absichtlich kleinen, großen oder Cross-File-Aufgaben.
- Keine Auswahl von Aufgaben, weil Greppy sie vermutlich gut löst.
- Keine künstlich geschwächte Baseline ohne `rg`, Read- oder Edit-Werkzeuge.
- Keine Aussage über allgemeine Agenten-Correctness ohne unabhängige Tests.
- Der bestehende 41er-Bestand bleibt ein Dev-/Regressionstest, kein Release-Set.

## Feststehendes Benchmark-Design

- 144 Aufgaben aus 24 Repositories und acht Sprachen.
- Drei Repositories pro Sprache, sechs Aufgaben pro Repository.
- Repositories müssen am Parent-Snapshot mindestens mittelgroß sein.
- Quelle sind natürliche gemergte Pull Requests mit verknüpften echten Issues.
- Prompt ist ausschließlich Titel und Body des Issues, unverändert.
- Agent sieht einen history-freien Parent-Snapshot mit genau einem Commit.
- Gold-Patch, Folge-Commits und Hidden Tests bleiben außerhalb des Agenten.
- Beide Arme erhalten dasselbe Modell, Budget, Hardware und dieselben normalen
  Werkzeuge. Einziger beabsichtigter Unterschied ist die zur Laufzeit gelesene
  und gehashte ausgelieferte `AGENTS.md` im Treatment-Arm.

Die Repo-/Sprachbalance verhindert, dass ein einzelnes Projekt wie Hugo das
Gesamtergebnis dominiert. Sie ist keine Optimierung auf Greppy-Fähigkeiten.

## Auswahlregeln: Was entfernt wird

Folgende Regeln aus dem bisherigen Contract werden aus dem Admission- und
Selection-Gate entfernt:

- Mindestzahl geänderter Produktionsdateien.
- Mindest- oder Zielzahl geänderter Codezeilen.
- Auswahl nach Cross-File-, Migration-, Refactor- oder Bugfix-Quote.
- Feste Task-Klassenquoten.
- Mindestzahl von Kandidaten pro Repo-und-Klasse-Slot.
- Navigation Pressure als Eigenschaft des Gold-Patches.
- Greppy-Adoption als Release-Gate.

Auch ein normaler Ein-Datei-Fix kann in einem sehr großen unbekannten Repo
erhebliche Discovery-Arbeit verlangen. Umgekehrt beweist ein erzwungener
20-Dateien-Patch nicht automatisch realistische Agenteneffizienz.

## Auswahlregeln: Was bestehen bleibt

Ein Kandidat ist nur aus technischen und wissenschaftlichen Gründen
unzulässig:

- PR ist nicht gemergt oder schließt kein eindeutig verknüpftes echtes Issue.
- Parent- und Merge-Tree lassen sich nicht eindeutig rekonstruieren.
- Reine Docs-, Formatting-, Generated-, Vendor- oder Dependency-Bump-Änderung.
- Kein beobachtbarer Code- oder Konfigurations-Fix.
- Keine aus dem PR ableitbaren unabhängigen Behavior-Tests.
- Parent+Hidden-Test schlägt nicht aus dem beabsichtigten Grund fehl.
- Gold+Hidden-Test oder PASS_TO_PASS ist nicht reproduzierbar grün.
- Aufgabe braucht Internet, bezahlte Dienste, privilegierte Infrastruktur oder
  nicht versiegelbare mutable Daten.
- Security-Embargo, Credentials, vollständige Lösung im Issue oder anderes Leak.
- Aufgabe befindet sich bereits in SWE-bench, V2 oder einem anderen Denylist-Set.
- Aufgabe ist unter dem registrierten Modell-/Agentenbudget technisch nicht
  ausführbar; die konkrete Ursache wird protokolliert und darf keine
  Greppy-spezifische Begründung sein.

Es gibt keine Admission-Regel für Patchzeilen, Zahl der Module, Symbolgraph-
Tiefe oder erwartete Greppy-Kommandos.

## Natürliche Stichprobe

1. Registry, Zeitfenster, Modell, Agent, Budgets und Auswahlalgorithmus werden
   vor dem Harvest eingefroren.
2. Für jedes Repo werden alle gemergten PRs im Zeitfenster erfasst, nicht nur
   eine handverlesene Kandidatenzahl.
3. Die rein technischen Ausschlussgründe werden auf alle Kandidaten angewandt.
4. Vor der Validierung werden alle verbleibenden Kandidaten pro Repo mit einem
   versiegelten HMAC-Secret deterministisch zufällig sortiert.
5. Validiert wird in dieser Reihenfolge, bis sechs Aufgaben plus mindestens
   zwei versiegelte Reserven pro Repo reproduzierbar bestehen.
6. Ausfälle dürfen nur durch den nächsten Kandidaten desselben Repos ersetzt
   werden. Kein Backfill aus einem anderen Repo oder einer gewünschten Klasse.
7. Weder Greppy noch ein Benchmark-Arm wird vor dem Corpus-Seal ausgeführt.

Die Candidate-Ledger enthält jeden Kandidaten und jeden Ausschlussgrund. Damit
lässt sich später prüfen, ob die Validierung ungewollt nur eine bestimmte
Aufgabenart übrig gelassen hat.

## Post-hoc-Stratifizierung

Erst nach der Auswahl werden diagnostische Eigenschaften berechnet:

- Gold-Patch-Zeilen und geänderte Produktionsdateien.
- Anzahl betroffener Module/Packages und Sprachen.
- Issue-Länge, Testlaufzeit und Repository-Größe.
- Bugfix, Feature, Refactor, Migration, Robustness oder Mischform.
- Ein-Datei versus Cross-File und lokal versus architekturübergreifend.
- Tatsächlich verwendete Greppy-Fähigkeiten.

Diese Tags dienen ausschließlich der Erklärung der Ergebnisse. Sie dürfen
weder Tasks aufnehmen/entfernen noch das primäre Gesamtergebnis gewichten.

## Hidden-Test- und Grader-Vertrag

- Der Agent arbeitet nur auf dem Parent-Snapshot.
- Der Agent-Diff wird nach Agentenende in einen frischen Grader-Workspace
  übernommen.
- Erst danach wird der versiegelte Hidden-Test-Patch angewandt.
- Versiegelte `post_patch_commands` laufen anschließend vor FAIL_TO_PASS und
  PASS_TO_PASS, damit C++- und Node-Artefakte nicht veraltet sind.
- Setup-, Rebuild- und Testkommandos sind argv-Arrays, keine Shell-Strings.
- Der gesamte Evaluation-Spec wird gehasht und beim Laden fail-closed geprüft.
- Parent-fail/Gold-pass und PASS_TO_PASS werden zweimal in sauberen Workspaces
  im gleichen digest-gepinnten Image reproduziert.

## Experiment und Metriken

Primär:

1. Correctness: gepaarte Treatment-minus-Control-Nichtunterlegenheit mit einer
   vorregistrierten Untergrenze von minus fünf Prozentpunkten.
2. Brutto-Providerkosten als Intention-to-treat über alle Aufgaben, Fehler,
   Retries und zensierten Läufe; Prompt-Overhead bleibt enthalten.

Sekundär:

- Kosten pro gelöster Aufgabe.
- Input-/Outputtokens, Wall-Time, Median, Mittelwert, p95 und gepaarte CIs.
- Treatment-Prompt-Overhead separat in Bytes und geschätzten Tokens.
- Cold-, Warm- und inkrementelle Indexkosten getrennt.

Diagnostisch, nicht als Release-Gate:

- Greppy-Adoption und verwendete Verben.
- Tool Calls und Source Opens aller Werkzeuge.
- Re-reads, Editversuche, Refusals und Fallbacks.
- Ergebnis nach Repo-Größe und post-hoc Task-Eigenschaften.
- Transactionality erst nach echter Per-Tool-Before/After-Instrumentierung.

Jede Rate mit Nenner null ist `N/A` und kann kein Gate bestehen.

## Faire Arme

Control und Treatment erhalten identisch:

- `bash`, Read-, Edit- und Write-Werkzeuge.
- Ein funktionierendes `rg`.
- Modell, Sampling, Token-/Turn-/Zeitbudget und Retry-Policy.
- Parent-Snapshot, Dependency-Caches, Image und Hardware.
- Test- und Graderlogik.

Das Treatment erhält zusätzlich exakt die ausgelieferte `AGENTS.md`. Es gibt
keine eingebettete Kopie des Greppy-Vokabulars im Harness.

## gpu3-Ausführungsplan

### Phase 1: Contract und Registry überarbeiten

- Feste Task-Klassenquoten und Patchgrößen-Gates entfernen.
- Natürlichen HMAC-Sampling-Algorithmus implementieren.
- Candidate-Ledger um alle Ausschlussgründe und Rangpositionen erweitern.
- Adoption vom Release-Gate in Diagnostics verschieben.
- Contract-, Pipeline- und Runner-Tests aktualisieren.

Abnahme: Unit-Suite grün; unabhängiger Audit bestätigt, dass kein
Greppy-spezifisches Selektionssignal mehr existiert.

### Phase 2: 24 Adapter real verifizieren

- Digest-gepinnte Images und Offline-Dependency-Caches bauen.
- Alle repository-spezifischen Setup-, Rebuild- und Testkommandos manuell
  verifizieren.
- Pro Repo mindestens einen echten Kandidaten zweimal clean-room validieren.
- Erst danach Adapterstatus von `pending` auf `ready` setzen.

Abnahme: 24/24 Adapter-Smoke-Ledger reproduzierbar grün. Kein stilles Droppen
von Java, C++, Ruby oder service-lastigen Repositories.

### Phase 3: Vollständiger Metadata-Harvest

- Alle PRs des Holdout-Zeitfensters je Repo erfassen.
- Issue/PR-Texte, Merge-Provenienz und Zeitstempel kanonisch einfrieren.
- Denylists und exakte/nahe Duplikate anwenden.
- HMAC-Ranking erzeugen, ohne Agenten oder Greppy auszuführen.

Abnahme: Vollständiges, gehashtes Candidate-Ledger auf NAS; Counts und
Ausschlussgründe pro Repo veröffentlichbar auditierbar.

### Phase 4: Offline-Validierung und Seal

- Kandidaten pro Repo in HMAC-Reihenfolge auf NVMe validieren.
- Sechs Aufgaben plus mindestens zwei Reserven pro Repo erzeugen.
- Parent-Snapshots mit genau einem Commit exportieren.
- Hidden Tests, Gold, Evaluation-Spec und Provenienz getrennt versiegeln.
- Matrix, Denylists, Auswahlsecret-Commitment und Artefakthashes auditieren.

Abnahme: exakt 144 ausgewählte Aufgaben, 24 Repos, acht Sprachen, keine
fehlenden Slots und kein Gold-/History-/Task-ID-Leak.

### Phase 5: Drei vollständige Smoke-Trajektorien

- Drei Aufgaben aus unterschiedlichen Sprachen und natürlichen
  Schwierigkeitsbereichen auswählen.
- Beide Arme vollständig laufen lassen und alle sechs Traces lesen.
- Command-not-found-, Refusal-Loop-, Output-Bomb-, Source-Open-, Hidden-Test-
  und Kosten-Reconciliation-Probleme ausschließen.
- Jede Runtime-Änderung invalidiert den Smoke und erzwingt eine Wiederholung.

Abnahme: signierte Smoke-Evidenz ohne offene Findings.

### Phase 6: Produktiver 144er-Lauf

- Provider-Credential-Broker implementieren, sodass Agent/Shell den Key nicht
  lesen und keine ungemessenen Modellaufrufe ausführen kann.
- Netzwerk-, Preflight- und Broker-Evidenz signieren.
- Exakt balancierte Arm-Reihenfolge ausführen und Checkpoints auf NVMe halten.
- Redigierte Ergebnisse und unveränderliche Evidenz atomar auf NAS publizieren.

Abnahme: kein Full Run ohne Credential-Isolation, signierten Preflight und
gültigen Drei-Trajektorien-Smoke; ein fehlgeschlagenes Release-Gate endet
nonzero und bewahrt trotzdem das Ergebnisarchiv.

## Reihenfolge und Parallelisierung

Phase 1 muss vor neuem Harvest abgeschlossen sein. Danach können Adapter-
Images/Caches und der Metadata-Harvest parallel vorbereitet werden. Die
Offline-Validierung beginnt erst mit geprüften Adaptern. Der Credential-Broker
kann parallel zur Corpus-Erzeugung entstehen, blockiert aber Smoke und Full
Run mit Release-Evidenz.

## Fertig-Definition

Der Bench ist erst fertig, wenn alle folgenden Aussagen wahr sind:

- Die 144 echten Aufgaben existieren und sind versiegelt; nicht nur der Harness.
- Keine Aufgabe wurde nach Greppy-Fähigkeit, Patchform oder Arm-Ergebnis gewählt.
- 24/24 Repositories sind real validiert und keines dominiert das Ergebnis.
- Agenten sehen weder Gold, Hidden Tests, Folge-History noch aussagekräftige IDs.
- Beide Arme besitzen dieselben normalen Werkzeuge und Budgets.
- Alle Providerkosten inklusive Prompt-Overhead, Fehlern und Retries sind erfasst.
- Drei vollständige gepaarte Traces wurden vor dem Full Run gelesen.
- Der Full Run ist credential-isoliert, signiert, reproduzierbar und fail-closed.
