# S06: Zusätzliche Agentenrunden bleiben der Effizienzfehler

Stand: 2026-09-05. **Effizienzabnahme erneut fehlgeschlagen.**

Fünf neue gepaarte Checkbox-Aufgaben, beide Arme gpt-5.6-luna/medium:
A = Codex mit Standard-CUA, C = Codex mit Greppy Web. Zehn unverändert
vorbereitete Prompts, zehn tatsächliche Dispatches, zehn abgeschlossene Exporte
und zehn unabhängige Ergebnisprüfungen. Alle zehn Aufgaben korrekt.
Der native Greppy-Agent B ist weiterhin nicht enthalten.

Diese Serie prüft ausschließlich eine explizitere Tool-/Transport-Einweisung
für beide Arme. CLI, Runtime, Aufgabe und kompakte Ansichten entsprechen S05.
Die späteren CLI-Validierungsänderungen sind nicht in diesen Kandidaten.

## Gemessene Tokens

| Kennzahl | A: Median | C: Median | Median der gepaarten Veränderungen |
|---|---:|---:|---:|
| Provider-Input | 227467 | 228067 | +26,47 % |
| Provider-Output | 331 | 570 | +57,36 % |
| Tool-Aufrufe | 4 | 5 | — |

Gepaarte Prozentänderungen sind nicht das Verhältnis der Arm-Mediane.
Alle Fehlversuche und Wiederholungen bleiben in den gemeldeten Providerzählern.
Cachewerte oder kleinere Antwortbytes ersetzen keines dieser Gates.

| Paar | A Input / Output / Calls | C Input / Output / Calls |
|---|---|---|
| 1 | 179689 / 260 / 3 | 228067 / 584 / 5 |
| 2 | 443989 / 891 / 9 | 228520 / 570 / 5 |
| 3 | 356126 / 649 / 7 | 188542 / 320 / 4 |
| 4 | 227467 / 331 / 4 | 346939 / 738 / 8 |
| 5 | 179538 / 258 / 3 | 227061 / 406 / 5 |

Quelle: [S06 summary.json](/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-06/summary.json).
Plan-SHA256: 2d261e1a02ce57d59acf24ce78d7a51ab84a274f53d246713aa3971d9ac7b894.
CLI 0.4.0, Quelle 2572b96f, SHA256
555cd58f17c857607d3852183280dbba790a307b5562d7a9025777f271908137.
Runtime SHA256 57318ead7505fdf2aa7e62a89c511bc207a9c9c9848e50e11421fc208678d399.

## Was die öffentlichen Traces belegen

1. **Bündelung mit alten Referenzen scheitert weiter.** C1 und C5 führen
   click/ref → fill/ref aus. Nach click meldet die Ansicht checked=true,
   quantity aber noch disabled und Revision 0. Die Anwendung ersetzt die
   Eingaben; fill endet mit STALE_REF. Danach folgen observe und fill mit
   neuer Referenz. Die Verweigerung der alten Referenz ist korrekt. Nötig
   sind eine ausdrückliche Bereitschaftsbedingung und aktuelle, eindeutig
   aufgelöste Ziele im Ablauf; keine stille Umbindung alter Identitäten.

2. **Erfolgreiches Warten liefert keine handlungsfähige Folgeansicht.** C4:
   click → wait text~/Quantity/ → inspect @602 → Fehler → observe → fill →
   press TAB → observe. Der Wait liefert held=true, matched=4, waited_ms=719,
   aber keine aktuellen Referenzen. Der gesuchte Text war schon vorhanden;
   diese Bedingung belegt ausdrücklich nicht, dass die Eingabe freigegeben
   wurde. Tool-Verständlichkeit muss die relevante Zustandsprüfung erleichtern.

3. **Inspect führt in technische Fehlerdetails statt gezielter Recovery.**
   C4 inspect @602 endet mit exit34/web.evaluate, einer internen
   EvaluationFailure/JavaScriptErrorInfo-Struktur und SyntaxError. Gleichzeitig
   heißt es retryable=false und „retry the operation or inspect web.doctor“.
   Bei ersetzter Referenz wäre STALE_REF mit konkreter Folgemaßnahme nötig;
   bei nicht unterstützter Ref-Syntax eine präzise Syntaxdiagnose. Eine erneute
   vollständige Beobachtung beschafft anschließend @1002. Das ist ein belegter
   Diagnose-/Resolververdacht, keine bereits nachgewiesene native Grundursache.

4. **Gezielte Beobachtungen werden ignoriert.** C1 observe body, C2 observe
   @1002 und C5 observe input[type=number] liefern exit0 und „web takes no
   argument; ignoring …“, anschließend die ganze Ansicht. Die aktuelle
   Repository-Anleitung dokumentiert observe QUERY. Mindestens der Vertrag
   zwischen Anleitung, Argumentbehandlung und Diagnose ist inkonsistent.

5. **Shell-Transport bleibt eine zusätzliche Fehlerfläche.** Trotz exakt
   vorgegebenem Erstaufruf geben C1/C2/C5 nur r.output aus. C1 erhält beim
   ersten open leeren Output und startet observe; die sichtbare Spur enthält
   damit keine weitergereichte session_id. Daraus lässt sich nicht behaupten,
   dass Greppy selbst keine Sitzungskennung geliefert hat. C5 lässt zunächst
   ein schließendes Anführungszeichen weg; zsh meldet eindeutig unmatched '.
   Das ist ein Generierungsfehler trotz konkreter Anleitung, kein Greppy-
   Parserfehler. C3/C4 verwenden das vollständig weitergereichte Ergebnis.

6. **Der effiziente Weg ist möglich, aber nicht zuverlässig.** C3 verwendet
   open, Poll desselben laufenden Shell-Prozesses, click und fill. C2 benötigt
   dagegen observe nach click und nach fill. C4 kommt auf acht Aufrufe. Das
   begründet die Priorität für zuverlässige Aktions-/Bedingungsrückmeldungen;
   es beweist noch nicht die Einsparung einer bestimmten Implementierung.

Ausgewertet wurden öffentliche Tool-Aufrufe, Ergebnisse und Providerzähler.
Ein innerer Vertrauensverlust wird nicht behauptet und privates Reasoning
nicht rekonstruiert. Beobachtbar sind zusätzliche Kontroll-, Fehler- und
Reparaturschritte. Jede zusätzliche Modellrunde liest hier erneut einen großen
Kontext und erzeugt weitere Befehle; kleinere einzelne Antworten reichen nicht.

## Standardarm und Grenzen

A1/A5 erledigen die Aufgabe in drei Aufrufen. A4 verwendet eine Kette mit
alter AX-Identität, erhält „Input is no longer connected“, aktualisiert die
Ansicht und setzt den Wert: vier Aufrufe. Das Ersetzungsproblem betrifft
auch Standard-CUA und darf nicht exklusiv Greppy zugeschrieben werden.

A2/A3 erleben beim ersten createBrowserTab einen Timeout mit Kernel-Reset.
Der wiederholte Aufruf mit visible:false meldet anschließend trotzdem
„IAB visibility is not supported in a subagent thread“. Zusätzliche
Tab-Bindungsschritte folgen, in A3 außerdem „tab is not defined“. Derselbe
vorbereitete Erstaufruf funktioniert bei A1/A4/A5. Diese Host-/Toolfehler
bleiben vollständig enthalten; sie sind kein durch Greppy erzielter Vorteil.
Damit ist S06 keine saubere kausale Quantifizierung allein des Onboardings.

Ein Entwicklungsfall, fünf Paare und unkontrollierte Abschluss-zu-Oracle-Zeit
belegen weder die Gesamtzeitziele noch die zwölf Aufgaben der vollständigen
Abnahme. Keine Überlegenheit und keine Standardaktivierung.

## Maßnahmen und Zuständigkeit

Die konkreten observe-/inspect-/Referenzbefunde wurden mit Befehlen, Version,
Hashes, Kontexten, Exitcodes und Recovery an die bestehende Aufgabe
01a02118-0d61-7e10-a9d4-be496fa34879 gemeldet; Rückmeldung nach Validierung
angefordert. Nach allen zehn terminalen Exporten wurde die Browser-/Build-Lane
explizit an den Fix-Worker zurückgegeben.

Root: CLI-Verständlichkeit, gezielte Beobachtungen und Einbindung geprüfter
Aktions-/Bedingungsrückmeldungen. Worker: native Fehler-/Wait-Pfade. Der neue
interne Boolean-Wait ist nur ein Quellentwurf mit Teiltests; native Deadline-,
Tab- und Referenzprüfung stehen aus. Keine CLI-Anbindung vor dieser Prüfung.
Die bereits separat getestete Query-Validierung d00c675f wurde nicht als
Tokenverbesserung ausgegeben. Die bisherigen kompakten Ansichten bleiben ein
Experiment ohne Effizienzfreigabe.

Öffentliche Metadaten und unveränderte Rohartefakte sind pro Lauf über
summary.json und trials/*/trial.json verknüpft. Frühere Fehlserien bleiben erhalten.
