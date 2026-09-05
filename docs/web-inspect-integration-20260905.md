# Integration vorhandener Web-Fixes nach S06

S06 ist abgeschlossen und scheitert weiterhin an beiden Token-Gates. Diese
Integration ist keine neue Effizienzmessung und verändert keine S06-Artefakte.

## Übernommener Quellstand

- Aus 475b8152: präzise verschachtelte Web-Befehlsdiagnosen in lib.rs und
  resolving.rs, Ref-Diagnosen in see.rs und vorhandene CLI-Regressionstests.
  Die chain.rs-Änderung war im Effizienz-Worktree bereits wortgleich vorhanden.
- Aus dd108d54: CLI inspect @REF über web.inspect, ausdrückliche Tab-Zuordnung,
  gemeinsamer describe-node.js-Serializer samt Web-Client-Protokollkennung,
  Tests und inspect-refs-v1-Vertrag. Native Quellen dieses Commits wurden
  hier nicht kopiert; die native Implementierung bleibt beim Fix-Worker.
- Die eigene, gerade vorbereitete gleichartige Inspect-Implementierung wurde
  vor der Übernahme verworfen. Es gibt keinen parallelen zweiten Resolver.
- Die bereits getestete Root-Query-Validierung d00c675f bleibt bestehen.

Die Runtime muss web.inspect unterstützen. Es gibt keinen automatischen
JS-Fallback für abgelehnte oder veraltete Referenzen. Bestehende Nicht-Ref-
Queries behalten ihren bisherigen Ausgabevertrag und berücksichtigen den Tab.

## Verifikation

Am 5. September wurden am integrierten Root-Kandidaten abgeschlossen:

- Vier Unit-Tests für strikt boolesche Condition-Antworten bestanden. Fehlende,
  nicht boolesche oder fehlerhafte Runtime-Antworten dürfen insbesondere mit
  --absent keinen Erfolg ergeben. Bestehende typisierte Fehler bleiben erhalten.
- 49 CLI-Tests bestanden: web_cli 39, web_chain_compact 6,
  web_condition_diagnostics 3, web_inspect_diagnostics 1. Optional übersprungene
  native Prüfungen in web_cli sind dabei kein Nachweis der Runtime-Funktion.
- Die tatsächliche CLI/Runtime-Probe beendete sich mit Exit 0, allen 13 Checks
  und erfolgreichem Stop ihres eigenen Runtime-Kontexts. Sie prüfte native
  Inspect-Ausgabe, deaktivierte Formularwerte, ausdrückliche Tab-Zuordnung bei
  Inspect/Assert/Wait, abgelehnte falsche Tab-Identität, abgelehnte ersetzte
  Knoten, den aktuellen Ersatzwert und unveränderte Kandidatenbytes.

Build- und Testbelege: /Users/michaelwelsch/.local/state/greppy-web-study/cli-inspect-integration-20260905-01.
Native Einzelaufrufe, Kontext und Terminalbeleg: /Users/michaelwelsch/.local/state/greppy-web-study/inspect-proof-20260905-01.
Die Build-Test-Zusammenfassung ist kein Ersatz für einen vollständigen Rohlog;
ursprüngliche Tool-Ausgaben bleiben im Root-Trace erhalten.

Geprüfte Tests:


- crates/cli/tests/web_inspect_diagnostics.rs: ungültige @-Identitäten vor
  Session/RPC ablehnen, ohne verschluckte --tab-Argumente oder JS-Fehler.
- bench/web_study/basic_fixture/inspect_ref_probe.py: echter CLI/Runtime-Test
  mit deaktivierter Eingabe, ausgewähltem nicht aktivem Tab, falschem Tab,
  ersetztem Knoten, aktuellem Ersatzknoten und unveränderten Kandidatenbytes.
  Jeder Aufruf wird vor Assertions gespeichert. Fehlende Runtime ist kein Skip;
  kein fehlgeschlagener Browseraufruf wird automatisch wiederholt. Der Test
  ist eine Funktionsprüfung, keine agentische Effizienzserie.

Vorgesehener bestehender Runtimekandidat:
/Volumes/tmp/dev-artifacts/greppy/e1-label-native-candidate.THEPk0/preserved-bin/web-runtime
SHA256 3ee1f2e17a697b29fac40791028228d085b819af9c4d2a2042b34e877b3cdfb8.
Der Hash wurde in dieser Runde erneut am tatsächlichen Artefakt bestätigt.
Der geprüfte Root-CLI-Kandidat ist separat eingefroren:
/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-inspect-154d1a775f41/greppy
SHA256 154d1a775f4156c6d33742ac309e063a49563a57e16cfb6a92c97a20a8082471.
Die dortige provenance.json enthält Quell-HEAD 89f4abeb und den beim Build
zusätzlich enthaltenen Condition-/Probe-Patch. Die native Probe bestätigt die
Binäridentität vor und nach dem Lauf. Build und Browserprobe liefen seriell
zur nativen Worker-Lane und erst nach dem vereinbarten Archivfenster.

Die Condition-Prüfung enthält weiterhin den vorhandenen Polling-Weg; dieser
Funktionstest ist weder der neue native Wait noch ein Token-/Zeitnachweis.


## Ausstehende Arbeit

Nativer Boolean-Wait: Quellentwurf beim Fix-Worker, keine CLI-Anbindung vor
nativer Deadline-/Tab-/Ref-Prüfung. Gezieltes observe QUERY/within ist im
vorhandenen nativen page.observe nicht implementiert; es wird nicht als
vorhandene Funktion dargestellt oder durch stilles Ignorieren ersetzt.

Erst nach funktional geprüftem zusammengehörigem Kandidaten folgen weitere
unbekannte Luna-Aufgaben, anschließend komplexere Abläufe und CTOX-Office.
Die vollständigen Dreiarm-, Zeit-, Token- und Korrektheitskriterien bleiben offen.

Der beim Patch-Import getroffene Git-Diff-Metadatenfehler wurde an den festen
Fix-Task gemeldet und als bereits durch 7b45519a behoben zugeordnet. Die sichere
Importalternative entfernt nur diff --git/index-Metadaten; neue Dateien wurden
mit greppy write übernommen. Ein gesonderter Import dieses Parserfixes hatte
einen echten Kontextkonflikt und schrieb nichts. Es wurde kein Parsercode
im Rahmen dieser Web-Integration improvisiert.
