# RAM-Budget für headless Ubuntu (1–4 GiB)

`--max-ram` (Umgebung: `GREPPY_MAX_RAM`) ist die gesamte Host-RAM-Größe des
Servers, zum Beispiel `2g` auf einer Maschine mit 2 GiB. Es ist nicht das
Budget, das nach dem Betriebssystem übrig bleibt. Es gibt kein einzelnes
Minimal-Flag. Ist nichts gesetzt, bleibt das Produktverhalten: `--device auto`
wählt die GPU, beide Modelle dürfen laden, der Web-Daemon hat kein globales
Sitzungs- oder Tab-Limit.

Der Webstack (Servo + deno_core) wird nie abgeschaltet. Er bleibt bei jedem
Budget benutzbar. Die harten Hebel sitzen bei EmbeddingGemma und dem
Qwen3.5-Summarizer. Am Web sind genau zwei Maßnahmen umgesetzt: ein globales
Limit für gleichzeitig lebende Tabs, und das Auslagern inaktiver Tabs auf
Disk mit späterem Wiederherstellen.

## Formel

Alles in GiB, 1024-basiert. `--max-ram` ist der ganze Rechner. `Rest` ist das,
was die Modelle bekommen dürfen, nachdem Ubuntu, die Browser-Reserve und der
Browser-Slack abgezogen sind:

```
Rest = Host-RAM − Ubuntu − Browser-Reserve − Browser-Slack
Host-RAM        = --max-ram
Ubuntu          = 250 MiB
Browser-Reserve = 1,25 GiB
Browser-Slack   = 256 MiB
```

Ubuntu und Browser-Slack sind zwei verschiedene Posten. Der Slack ist Puffer
neben der Browser-Reserve (greppy-Prozess, Seitenrauschen). Er ist nicht das
Betriebssystem. Vorher steckte das OS nur in diesem ungemessenen 256-MiB-Slack,
und der Rest bei 2 GiB wurde in dieser Datei noch einmal als OS-Marge
beschrieben. Das ist die alte Rechnung, nicht mehr die Semantik.

| Konstante | Wert | Herkunft |
| --- | --- | --- |
| Browser-Reserve | 1,25 GiB | **Im Code belegt** als Kommentar in `crates/web-runtime/runtime/src/limits.rs`: eine gespiegelte Nachrichtenseite brauchte 1,25 GB Content-RSS. Die Zahl ist die Prozess-RSS, deshalb wird das Runtime-Image nicht noch einmal addiert. Plattform und Messprotokoll stehen im Kommentar nicht. |
| Runtime-Image | ~400 MiB | **Im Code belegt** als Kommentar in `crates/web-runtime/runtime/Cargo.toml` und in `daemon.rs` („400MB image“). Das ist die Image-Größe beim Start, nicht die RSS einer Seite. |
| Ubuntu minimal | 250 MiB | **Gemessen von Michael** auf einem headless Ubuntu, idle, etwa 250 MB. Steht nicht im greppy-Quelltext. Wird als `OS_RESERVE_BYTES` vor Browser-Reserve und Browser-Slack abgezogen. |
| Browser-Slack | 256 MiB | **Planungswert, nicht gemessen.** Puffer neben der Browser-Reserve für den greppy-Prozess und Seitenrauschen. Nicht das Betriebssystem. |
| Embedding-Budget | 768 MiB | **Im Code belegt:** `estimated_gpu_memory` addiert für EmbeddingGemma einen Workspace von 768 MiB (`crates/embed-native/src/backend.rs`). |
| Summary-Budget | 1024 MiB | **Im Code belegt:** derselbe Helper, Workspace 1024 MiB für Qwen3.5. |
| Embedding-RSS, CUDA | 702 910 464 B (670 MiB) | **Gemessen**, nicht CPU: `bench/evidence/runtime-footprint-linux-x86_64-cuda-a4500.json`, Daemon `embedding`, `state: ready`, während der semantischen Suche. Gewichte lagen auf der GPU. |
| Summary-RSS, CUDA | 541 036 544 B (516 MiB) | **Gemessen**, dieselbe Datei, Daemon `summary`, `state: ready`. |
| `required_gpu_memory` | 2 647 337 024 B | **Gemessen und im Code hergeleitet:** Summe aus beiden Workspaces plus den GGUF-Dateigrößen. Dateien zusammen 768 288 832 B. |
| Payload-Kommentar | 827 MB | **Im Code belegt** in `crates/cli/Cargo.toml` (ci-test-assets ersetzt „827 MB payloads“). Liegt über der reinen GGUF-Summe, weil Tokenizer und Binary mitzählen. |

CPU-Inferenz kopiert die quantisierten Gewichte aus dem mmap in den Prozess
(`CpuEmbeddingModel::from_gguf`, `CpuQwen35Model::load`). Das mmap selbst wird
danach fallengelassen. Die Planung benutzt weiterhin die Workspace-Konstanten.
Die Debug-CPU-RSS steht in „Messung“, nicht in dieser Tabelle.

Ein Modell wird nur eingeplant, wenn `Rest` sein Budget deckt. Summaries kommen
erst dazu, wenn `Rest` Embedding und Summary zusammen deckt. Eine zweite
lebende Seite kommt erst dazu, wenn nach den eingeplanten Modellen noch einmal
die Browser-Reserve übrig ist. Das ist bei 1–4 GiB nicht der Fall: eine zweite
schwere Seite und beide Modelle passen in dieser Rechnung nicht gleichzeitig.

## Stufen

| Budget | Rest nach Ubuntu, Browser und Slack | Embedding | Summary | Lebende Tabs (global) | Sessions (global) | Tab-Swap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GiB (`1g`, alles unter 2 GiB, z. B. `1536m`) | 0 | aus | aus | 1 | 1 | ja |
| 2 GiB | 262 MiB | aus | aus | 1 | 1 | ja |
| 3 GiB | 1286 MiB | lazy, CPU, Modell-TTL 60 s, Prozess-TTL 180 s | aus | 1 | 1 | ja |
| 4 GiB und darüber | 2310 MiB | lazy, CPU, TTL 30 s / 120 s | lazy, CPU, TTL 30 s / 120 s | 1 | 2 | ja |

1 GiB und 2 GiB treffen dieselbe Modellentscheidung. 262 MiB Rest liegen unter
dem 768-MiB-Embedding-Workspace. Ubuntu ist in dieser Zahl schon abgezogen.
Der Unterschied zur 1-GiB-Stufe ist die verbleibende Marge für Seitenrauschen,
nicht ein zweites Modell und nicht noch einmal das Betriebssystem.

Geschätzte Spitze, Planung, nicht gemessen:

| Stufe | Ubuntu | Browser | Browser-Slack | Embedding | Summary | Summe der Planung |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GiB | 250 MiB | 1,25 GiB, real oft der ganze Rest | 256 MiB | 0 | 0 | über dem Rechner, siehe offene Punkte |
| 2 GiB | 250 MiB | 1,25 GiB | 256 MiB | 0 | 0 | 1786 MiB belegt, 262 MiB Rest |
| 3 GiB | 250 MiB | 1,25 GiB | 256 MiB | 768 MiB | 0 | 2554 MiB belegt, 518 MiB Rest |
| 4 GiB | 250 MiB | 1,25 GiB | 256 MiB | 768 MiB | 1024 MiB | 3578 MiB belegt, 518 MiB Rest, wenn beide Modelle gleichzeitig resident sind |

Die Default-TTLs ohne Budget bleiben 300 s Modell und 1800 s Prozess
(`GREPPY_EMBED_DAEMON_MODEL_TTL_S`, `GREPPY_SUMMARIZE_DAEMON_*`). Das Budget
setzt die kürzeren Fenster über dieselben Variablen. Entladen bei Idle ist
bereits implementiert: `inference_daemon.rs` setzt `model = None`, sobald
`model_ttl` ohne Abschluss verstrichen ist, und beendet den Prozess nach
`exit_ttl`.

## Was die Modelle im Code tun, wenn sie aus sind

| Funktion | Verhalten ohne Modell | Beleg |
| --- | --- | --- |
| `greppy index` | Graph wird veröffentlicht, Embedding-Pass entfällt (`embedding_config == None` schreibt keinen Vektorblock). | `index_atomic_snapshot_attempt` |
| `greppy search` | Exit 1, Status „embeddings are off“, plus die bestehenden lexikalischen Folgekommandos (`search-symbol`, `search-pattern`/`grep`). Es wird kein Embedding-Daemon gestartet. | neu, neben `emit_semantic_backend_unavailable` |
| `brief` | Doc-Kommentar, sonst kein generierter Satz. Die Skizze bleibt Graph und Parser. | `BriefRender::sentence` |
| Purpose-Zeilen der Suche | entfallen, weil `qwen_summary_config_optional` `None` liefert | `semantic_vector_purposes` |
| Triage | Fail-open: jeder Span `read: true`. Außerhalb von `summarize_daemon.rs` gibt es keinen Aufrufer. Im Agent-Prompt kommt Triage nicht vor. | `triage_spans_via_daemon` |
| Gecachte Summaries | werden bei abgeschaltetem Summarizer nicht gelesen, damit kein Modellschlüssel und kein Daemon entsteht | bewusste Grenze |
| Agent-Prompt | Die signierten Dateien `AGENTS.md` und `assets/prompts/agent-adapter.md` bleiben byte-identisch. `apply_model_prompt_edits` streicht modellabhängige Stellen erst beim Rendern, analog zum Browser-Block. Modelle an ist der signierte Prompt. | `crates/agent/src/prompt.rs` |

Lazy heißt: der Vordergrund-Index bleibt strukturell, die Vektoren entstehen
beim nächsten `search`. `prewarm_summary_daemon` startet unter einem Budget
nicht. Der erste `brief` oder die erste Purpose-Zeile lädt den Summarizer,
wenn die Stufe ihn erlaubt. `--device` bleibt unangetastet, wenn der Aufruf
eines nennt. Sonst setzt ein Budget mit Modell die Inferenz auf CPU. `auto`
fällt im Code nicht auf CPU zurück (`load_auto_backend`).

Ein explizites `--device cuda` zusammen mit `--max-ram` wird nicht umgebogen.
Die Planung nimmt dann Host-RSS an, die für CPU-Gewichte gilt; CUDA liegt
neben dieser Rechnung. Ohne `--max-ram` ändert sich am GPU-Release nichts.

Kleinere Modelldateien gibt es im Baum nicht. `MODEL_ASSETS.json` kennt nur
`embeddinggemma-300M-Q4_K` und `Qwen3.5-0.8B-MTP-Q4_K_M`. Quantisierung ist
bereits Q4. Eine weitere Stufe „kleineres Modell“ ist kein Codepfad.


## Agent-Prompt, wenn ein Modell aus ist

Nur der gerenderte System-Prompt ändert sich. Der Contract-Hash prüft die ungekürzten Quellen.

Embeddings aus (`1g`, `2g`):

- `, plus a meaning index over its source` fällt weg.
- Beide Zeilen `greppy search "WHAT IT DOES"` (Routing und SEARCH) fallen weg.
- Der Absatz zur Concept-Discovery (`one search` / `search-pattern fallback`) fällt weg. `Do NOT run grep/find/read loops.` bleibt.
- `greppy plus` verliert das Wort `meaning`.
- `Indexing and embedding preparation are one-time work` wird zu `Indexing is one-time work`.

Summarizer aus (`1g`, `2g`, `3g`):

- `Text after an em dash is a generated hint, not source.` fällt weg.
- `purpose, signature and body sketched` wird zu `signature and body sketched`.

`greppy brief`, `search-symbol` und `search-pattern` bleiben. Triage steht im Prompt nicht. Bei 4 GiB und ohne `--max-ram` ist der Prompt byte-gleich mit dem signierten One-Shot-Text.

## Web

**Im Code belegt, Architektur**

- Ein Controller-Prozess (deno_core) und ein Content-Prozess (Servo). Alle
  Sessions teilen sie sich (`Daemon` hält je ein `WorkerProcess`).
- Jede Session baut ein eigenes `EngineBundle` mit einem weiteren Servo
  (`ContentEngine::build_bundle`).
- Jeder Tab ist ein `WebView` in diesem Bundle (`context.newPage`).
- `SessionLimits` gelten pro Session. Default: 16 Seiten, Content-RSS-Decke
  2 GiB, Controller-Heap 256 MiB. Die RSS-Decke wird in
  `with_session_page_until` geprüft. Sie wird vom Budget nicht gesenkt: der
  Kommentar an der Decke sagt, dass eine niedrigere Grenze echte Seiten
  scheitern lässt.
- Ein globales Session- oder Tab-Limit gab es vorher nicht (`sessions` ist
  eine unbegrenzte `HashMap`).

**Maßnahme 1.** Unter einem Budget gilt im Daemon `max_live_pages` und
`max_sessions` aus der Tabelle, über alle Sessions. Eine neue Session über dem
Limit schließt zuerst eine nicht beschäftigte Session. Sind alle beschäftigt,
wird die neue Session trotzdem angelegt: das Web soll nicht mit einem harten
Fehler aussteigen.

**Maßnahme 2.** Passt ein weiterer lebender Tab nicht ins Limit, wird der
älteste lebende Tab geschlossen, nachdem sein Snapshot auf Disk liegt
(`<store>/web-runtime/<run>/tab-swap/`). Der Tab bleibt in der Session als
ausgelagert sichtbar. `web tab switch` und jede Operation auf diese Id laden
ihn zurück. Die Engine vergibt dabei eine neue Page-Id; die alte Id wird
aliasiert.

### Was sich serialisieren lässt

| Zustand | Lesen | Zurückschreiben | Urteil |
| --- | --- | --- | --- |
| URL | `page.url` → `webview.url()` | `page.goto` | **Belegt.** Wiederherstellen lädt die URL neu. JavaScript-Heap, WebSockets und die History (`goBack`/`goForward` nur am lebenden WebView) gehen verloren. |
| Titel | `page.title` → `document.title` | nicht nötig | **Belegt**, nur Anzeige. |
| DOM | `page.content` → `outerHTML` | `page.setContent` schreibt per `document.write` | **Belegt als HTML-Kopie, unsicher als Seitenersatz.** Listener, Canvas, Iframes und laufende Skripte sind nicht der gespeicherte Baum. HTML über 1 MiB wird weggelassen. `setContent` nur, wenn die URL nicht `http(s)` ist. |
| Cookies | `page.cookies` liest `document.cookie` | `page.addCookies` schreibt `document.cookie` | **Belegt für nicht-HttpOnly.** Der Servo-Cookie-Jar hat keine eigene Export-API. Ob er `page.close` überlebt, ist **unsicher**; der Context-Servo bleibt, der WebView wird verworfen (`dispose_page`). |
| Scroll, localStorage, sessionStorage | kein eigenes Kommando; `page.evaluate` kann sie lesen | dasselbe per Evaluate | **Belegt als Script-Fähigkeit, unsicher wenn die Seite das verbietet.** IndexedDB, Cache API, Service Worker: keine API im Worker gefunden. |
| History | `page.goBack` / `page.goForward` | keine Export-Funktion | **Belegt, dass der Stack am WebView hängt.** Swap stellt ihn nicht wieder her. |
| Profil auf Disk | `config_dir` nur mit `GREPPY_WEB_CONTENT_CONFIG_DIR` oder persistentem Profil | vom Worker, nicht vom Swap | **Belegt.** Ephemere Sessions bekommen das nicht automatisch. |

Der Content-Worker startet mit `env_clear` und einer Allowlist
(`inherited_worker_env`). `GREPPY_MAX_RAM` muss der Supervisor-Prozess sehen.
Den startet die CLI ohne `env_clear`, er erbt die Variable. Ein schon
laufender Supervisor übernimmt ein später gesetztes Budget nicht.

## Messung (Debug-CPU, 3. November 2026)

Gemessen auf dieser Maschine, nicht auf einem 1-GiB-Server. Host: Linux 6.12, x86_64, 4 Kerne, etwa 15 GiB RAM, Swap des Hosts war 0, bis für die Swap-Variante eine 2-GiB-Swapdatei angelegt wurde. Grenze: cgroup v2, `memory.max` hart, `memory.swap.max` 0 außer in der Swap-Zeile. `systemd-run` ging nicht, weil PID 1 kein systemd ist. Docker war nicht da.

Binary: `target/debug/greppy` 0.4.6, `cargo build -p greppy --features cpu-only`. Das ist kein Release. Ein Release mit `cpu-only` verweigert der Build (`crates/cli/src/lib.rs`, `crates/cli/build.rs`): Produkt-Releases verlangen Metal oder CUDA/nvcc. Auf dieser Maschine gibt es kein `nvcc`, deshalb gibt es hier kein GPU-Release und kein CPU-Release. Die RSS unten ist die Debug-RSS. Eine kleinere Release-RSS ist nicht gemessen.

Korpus: lokaler Klon dieses Repos, 938 indexierte Dateien, 26842 Knoten. Jede RAM-Stufe hatte ein eigenes `GREPPY_STORE_DIR`. `--max-ram` und `--device cpu` standen auf der Kommandozeile. Die cgroup enthält nur die greppy-Prozesse, nicht Ubuntu. Eine Extra-Zeile setzt `memory.max` auf 2 GiB minus 250 MiB, damit die Ubuntu-Reserve nicht nur im Flag, sondern auch in der harten Grenze steckt.

„Cgroup“ ist `memory.peak` nach neu angelegter Gruppe. „HWM“ ist `VmHWM` des genannten Prozesses. Wo die beiden auseinanderlaufen, stehen beide Zahlen. Ein OOM ist `memory.events` `oom_kill` oder Exit 137.

### Coding, Swap aus

| Stufe | Aufgabe | Exit | OOM | Dauer | Cgroup-Peak | Prozess-HWM | Ergebnis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 GiB | `index` | 137 | 1 | 30,6 s | 1024 MiB, die Kappe | greppy 1060 MiB, anon 1021 MiB | Abbruch bei `extracting_files` 448/938. Meldung: Embeddings aus, Graph wäre der Index. |
| 1 GiB | `search-symbol` | 73 | 1 | 53,2 s | 1024 MiB | greppy 1070 MiB | Startet den Index neu, der bricht wieder ab. Kein veröffentlichter Graph. |
| 1 GiB | `search` | 1 | 0 | 1,1 s | 7,0 MiB | greppy 16,5 MiB | „embeddings are off“. Kein Embedding-Daemon. |
| 1 GiB | `brief` | 73 | 1 | 52,1 s | 1024 MiB | — | Versucht den fehlenden Index und stirbt dort. |
| 1 GiB | `search-pattern` eine Datei | 0 | 0 | 1,2 s | 9,1 MiB | greppy 34,7 MiB | Treffer auf `OS_RESERVE_BYTES`. Braucht den vollen Index dieses Laufs nicht. |
| 1 GiB | `replace-text` | 0 | 0 | 1,1 s | 6,7 MiB | greppy 29,1 MiB | Eine Doc-Zeile wurde wirklich geändert und danach im Klon zurückgesetzt. |
| 2 GiB | `index` | 0 | 0 | 72,5 s | 1877 MiB | greppy 1744 MiB, anon 1683 MiB | 938 Dateien, 26842 Knoten. Embeddings aus. |
| 2 GiB | `search-symbol` | 0 | 0 | 1,7 s | 51 MiB | greppy 81 MiB | `RamBudget` gefunden. |
| 2 GiB | `search` | 1 | 0 | 1,0 s | 7,0 MiB | — | Wie bei 1 GiB: Ablehnung, kein Daemon. |
| 2 GiB | `brief` | 0 | 0 | 1,5 s | 15,5 MiB | greppy 44 MiB | Struktur, kein Summary-Satz. |
| 2 GiB minus 250 MiB | `index` | 0 | 0 | 70,0 s | 1798 MiB, gleich der Kappe | greppy 1735 MiB, anon 1674 MiB | Fertig, `oom_kill` 0. Die Kappe ist 2 GiB abzüglich der Ubuntu-Reserve. Der Zähler lag auf der Kappe. |
| 3 GiB | `index` | 0 | 0 | 66,3 s | 1703 MiB | greppy 1595 MiB | „embeddings stay unloaded until the next semantic search“. |
| 3 GiB | `search-symbol` | 0 | 0 | 1,7 s | 51 MiB | greppy 81 MiB | Wie bei 2 GiB. |
| 4 GiB | `index` | 0 | 0 | 64,5 s | 1859 MiB | greppy 1726 MiB | Modelle bleiben bis zur Suche ausgeladen. |

Der Index bei 2, 3 und 4 GiB schwankt zwischen etwa 1700 und 1877 MiB Cgroup. Das ist Lauf-zu-Lauf, nicht eine höhere Stufe.

### CPU-Daemons

| Lauf | Exit | OOM | Dauer | Cgroup-Peak | Daemon-HWM | Was passiert ist |
| --- | --- | --- | --- | --- | --- | --- |
| 3 GiB, `search` über das ganze Repo | −9, Abbruch nach 600 s | 0 | 675 s inklusive 70 s Nachlauf | 1466 MiB | Embedding 738 MiB, davon anon bis 578 MiB, file bis 252 MiB. Danach RSS um 570–620 MiB. Kind-`index` 443 MiB. | Log blieb bei 0/15312 Spans. Worker-Threads liefen auf der CPU (`wchan` 0). Kein OOM. |
| 4 GiB, dieselbe Suche | −9 nach 600 s | 0 | 675 s | 1482 MiB | Embedding 738 MiB, anon bis 593 MiB | Dasselbe Bild. |
| 3 GiB, `search --path` auf `max_ram.rs` (26 Spans) | −9 nach 240 s | 0 | 365 s | 1134 MiB | Embedding 738 MiB | Auch 26 Spans kamen in 240 s nicht als Fortschritt im Log an. Die ETA von 26 s war die Schätzung bei 0, nicht eine gemessene Rate. |
| 4 GiB, `brief` auf `RamBudget` | 0 | 0 | 146 s | 1800 MiB, Peak zu Beginn 0 | Summarize 1740 MiB, anon bis 1249 MiB, file bis 533 MiB. greppy 48 MiB. | Der Daemon lief mit `--device cpu`. Der Client meldete `daemon request failed after restart` und erzeugte keinen Satz. `CLIENT_READ_TIMEOUT` im Code ist 60 s. Ein OOM war es nicht. Die Ursache hinter `Failed` ist nicht weiter isoliert. |
| 4 GiB, Suche und `brief` überlappend | 1 | 0 | 188 s | 2441 MiB, Peak zu Beginn 0 | Embedding 738 MiB und Summarize 1740 MiB, beide in diesem Lauf gesehen | Die Summe der beiden HWM ist 2478 MiB und liegt über dem Cgroup-Peak. Die beiden Maxima waren also nicht derselbe Augenblick. 2441 MiB ist die gemessene gemeinsame Ladung. Kein Browser, kein Ubuntu in der Gruppe. Der Summary-Satz fehlte wieder. Die Suche war nicht fertig. |

Nach dem Abbruch der 26-Span-Suche liefen Index-Kind und Embedding-Daemon weiter. Ein direkter Kill nur der Clients ließ den Daemon rechnen: `utime` stieg über 60 s, RSS blieb bei etwa 607–618 MiB, HWM 738 MiB. `model_ttl` von 60 s hat in diesem Fenster nichts freigegeben, weil die Anfrage im Daemon noch lief. RSS nach einem echten Idle-Unload ist nicht gemessen.

### Swap

Host-Swapdatei 2 GiB. Cgroup `memory.max` wie die Stufe, `memory.swap.max` 2 GiB.

| Stufe | Aufgabe | Exit | OOM | Dauer | RAM-Peak | Swap-Peak | HWM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 GiB | `index`, kalt | 0 | 0 | 102,7 s | 1024 MiB, die Kappe | 875 MiB (`memory.swap.peak`) | greppy 1077 MiB |
| 2 GiB | `index` | 0 | 0 | 64,3 s | 1824 MiB | nicht aufgezeichnet, der Lauf passt in RAM | 1690 MiB |
| 3 GiB | `index` | 0 | 0 | 64,3 s | 1381 MiB | nicht aufgezeichnet | 1249 MiB |
| 4 GiB | `index` | 0 | 0 | 63,7 s | 1399 MiB | 0 | 1268 MiB |

Ohne Swap stirbt derselbe Index bei 1 GiB. Mit 2 GiB Swap wird er fertig und benutzt 875 MiB Swap. Die kleineren Peaks bei 3 und 4 GiB sind die Schwankung des Index, nicht ein Effekt des Swaps.

### Was nicht gemessen ist

- Web. `greppy web doctor` endet mit Exit 31, „web-runtime distributable is not installed“. Die Release-Assets von v0.4.5 enthalten kein `web-runtime`. Der Vendored-Servo-Baum ist ein Patch, kein ganzer Browser. `libfontconfig-dev`, `libfreetype-dev` und `libclang-dev` fehlen hier. Es gibt keine Zahl für Content-RSS, für `page.close` und für den Tab-Swap. Die 1,25 GiB im Code bleiben der Kommentar, keine neue Messung.
- Agent-Läufe. In dieser Umgebung liegt kein API-Key. `greppy -p` wurde nicht aufgerufen.
- Release-RSS und GPU-RSS. Nur Debug-CPU.
- Ob `model_ttl` den RSS senkt, nachdem keine Anfrage mehr im Daemon liegt.

### Empfehlung

Nur aus den Zeilen oben, nicht aus der Planungsformel.

- Coding ohne Modelle, dieser Korpus, Debug-Binary: 2 GiB ist die kleinste harte Kappe, unter der `index` ohne Swap fertig wurde. 1 GiB ohne Swap reicht nicht. `search-pattern` auf eine Datei und `replace-text` gehen bei 1 GiB, der Index und alles, was ihn nachzieht, nicht. Zieht man die 250 MiB Ubuntu von der 2-GiB-Kappe ab, wurde der Index in diesem Lauf noch fertig, der Cgroup-Zähler lag aber auf der Kappe. Reserve nach oben ist in dieser Messung nicht da.
- Coding mit Swap: 1 GiB RAM plus 2 GiB Swap hat den Index fertiggestellt und 875 MiB Swap benutzt.
- Semantische Suche: bei 1 und 2 GiB gemessen abgeschaltet, Exit 1, kein Daemon. Bei 3 GiB passt der Embedding-Daemon (HWM 738 MiB) in die Kappe, die Suche über 15312 Spans und auch die über 26 Spans war nach dem Zeitlimit nicht fertig.
- Summaries: der Daemon wurde bei 4 GiB 1740 MiB groß und hat keinen Satz geliefert. OOM war es nicht. 3 GiB lässt Summaries aus, das ist die Stufe, nicht eine neue Messung eines fehlgeschlagenen Starts.
- Beide Modelle ohne Browser und ohne Ubuntu in der Gruppe: Cgroup-Peak 2441 MiB bei 4 GiB Kappe, kein OOM.
- Web, und Web zusammen mit Coding: keine gemessene Mindestgröße. Rechnung, keine Messung: 2441 MiB gemessene Modell-Überlappung plus die unveränderte 1,25-GiB-Browser-Reserve liegen über 4 GiB, bevor Ubuntu dazukommt. Ob eine echte Seite diese Reserve braucht, ist hier nicht geprüft.

## Auf einer frischen Ubuntu-Box

`greppy -p` spricht nur Klartext-HTTP mit einem lokalen Gateway (`crates/agent/src/client.rs`). `https://api.minimax.io` kann nicht `GREPPY_ENDPOINT` sein. MiniMax selbst, laut den API-Docs vom 3. November 2026:

- Anthropic-kompatible Basis: `https://api.minimax.io/anthropic`
- Modell-ID: `MiniMax-M3.1-Flash`
- Schlüssel: `MINIMAX_API_KEY` als Bearer gegen diese Basis

Das Gateway auf der Box muss diese Basis mit TLS aufrufen. greppy sieht nur das Gateway. Standard im Produkt ist CLIProxyAPI auf `http://127.0.0.1:8317`. Der Schlüssel, den greppy schickt, ist `GREPPY_API_KEY` (`x-api-key` und `Authorization: Bearer`), nicht eine eigene MiniMax-Variable. Das Skript kopiert `MINIMAX_API_KEY` nach `GREPPY_API_KEY`, wenn letztere leer ist. Das hilft nur, wenn das lokale Gateway genau diesen Schlüssel erwartet.

Modellfenster: die MiniMax-Docs nennen 1 000 000 Tokens. greppy leitet das aus dem Namen `MiniMax-M3.1-Flash` nicht ab. `--context-tokens 1000000` setzen.

```
export GREPPY_ENDPOINT=http://127.0.0.1:8317
export GREPPY_MODEL=MiniMax-M3.1-Flash
export GREPPY_API_KEY="$MINIMAX_API_KEY"
python3 bench/max_ram_box.py --bin /usr/bin/greppy --repo /pfad/zum/repo --out /tmp/max-ram.jsonl
```

Das Skript fährt 1, 2, 3 und 4 GiB mit `--max-ram` und `--device cpu`. Zuerst `systemd-run --scope` mit `MemoryMax`. Ohne systemd legt es eine cgroup v2 an. Web-Aufgaben laufen nur, wenn `greppy web doctor` den Stack findet. Die Agent-Aufgabe läuft nur, wenn `GREPPY_ENDPOINT` antwortet. Jede Zeile auf stdout ist ein JSON-Objekt mit Exit, OOM, Dauer, Cgroup-Peak und HWM je Prozess.

Ein CPU-only-Release liegt diesem Stand nicht bei. `cargo build --release -p greppy --features cpu-only` bricht ab, solange `GREPPY_ALLOW_CPU_ONLY_RELEASE` nicht existiert: der Schalter ist absichtlich nicht eingebaut, damit ein normales Release GPU bleibt. Debug auf einer Maschine ohne nvcc:

```
cargo build -p greppy --features cpu-only
```

Der Webstack ist ein zweites Binary, `crates/web-runtime`, Feature `content-runtime`. Er ist nicht im deb von v0.4.5. Bauen heißt Servo übersetzen und danach `crates/web-runtime/scripts/package-web-runtime.sh` in ein Verzeichnis legen, das `GREPPY_WEB_RUNTIME` oder `GREPPY_WEB_RUNTIME_DIST` zeigt.
