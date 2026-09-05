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

Nach der Übernahme: git diff --check erfolgreich. Die neue Python-Probe ist
syntaktisch geprüft, ihr Browserlauf steht aus. Der neue Rust-Diagnosetest
und die integrierten CLI-Tests sind in dieser Kombination noch UNGEPRÜFT.
Alte Testergebnisse des Fix-Workers ersetzen diese Integrationsprüfung nicht.

Vorbereitete Tests:

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
Der neue Root-CLI-Kandidat muss erst kompiliert, separat eingefroren und geprüft
werden. Der aktive native Wait-Build des Workers wird nicht durch einen
parallelen Root-Cargo-/Browserlauf gestört.

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
