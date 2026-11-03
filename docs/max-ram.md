# RAM-Budget für headless Ubuntu (1–4 GiB)

`--max-ram` (Umgebung: `GREPPY_MAX_RAM`) setzt das Host-RAM, in dem greppy
bleiben soll. Es gibt kein einzelnes Minimal-Flag. Ist nichts gesetzt, bleibt
das Produktverhalten: `--device auto` wählt die GPU, beide Modelle dürfen
laden, der Web-Daemon hat kein globales Sitzungs- oder Tab-Limit.

Der Webstack (Servo + deno_core) wird nie abgeschaltet. Er bleibt bei jedem
Budget benutzbar. Die harten Hebel sitzen bei EmbeddingGemma und dem
Qwen3.5-Summarizer. Am Web sind genau zwei Maßnahmen umgesetzt: ein globales
Limit für gleichzeitig lebende Tabs, und das Auslagern inaktiver Tabs auf
Disk mit späterem Wiederherstellen.

## Formel

Alles in GiB, 1024-basiert. `Rest` ist das, was die Modelle bekommen dürfen:

```
Rest = Budget − Browser-Reserve − Slack
Browser-Reserve = 1,25 GiB
Slack           = 256 MiB
```

| Konstante | Wert | Herkunft |
| --- | --- | --- |
| Browser-Reserve | 1,25 GiB | **Im Code belegt** als Kommentar in `crates/web-runtime/runtime/src/limits.rs`: eine gespiegelte Nachrichtenseite brauchte 1,25 GB Content-RSS. Die Zahl ist die Prozess-RSS, deshalb wird das Runtime-Image nicht noch einmal addiert. Plattform und Messprotokoll stehen im Kommentar nicht. |
| Runtime-Image | ~400 MiB | **Im Code belegt** als Kommentar in `crates/web-runtime/runtime/Cargo.toml` und in `daemon.rs` („400MB image“). Das ist die Image-Größe beim Start, nicht die RSS einer Seite. |
| Slack | 256 MiB | **Planungswert, nicht gemessen.** Platz für den greppy-Prozess und den Kernel. |
| Embedding-Budget | 768 MiB | **Im Code belegt:** `estimated_gpu_memory` addiert für EmbeddingGemma einen Workspace von 768 MiB (`crates/embed-native/src/backend.rs`). |
| Summary-Budget | 1024 MiB | **Im Code belegt:** derselbe Helper, Workspace 1024 MiB für Qwen3.5. |
| Embedding-RSS, CUDA | 702 910 464 B (670 MiB) | **Gemessen**, nicht CPU: `bench/evidence/runtime-footprint-linux-x86_64-cuda-a4500.json`, Daemon `embedding`, `state: ready`, während der semantischen Suche. Gewichte lagen auf der GPU. |
| Summary-RSS, CUDA | 541 036 544 B (516 MiB) | **Gemessen**, dieselbe Datei, Daemon `summary`, `state: ready`. |
| `required_gpu_memory` | 2 647 337 024 B | **Gemessen und im Code hergeleitet:** Summe aus beiden Workspaces plus den GGUF-Dateigrößen. Dateien zusammen 768 288 832 B. |
| Payload-Kommentar | 827 MB | **Im Code belegt** in `crates/cli/Cargo.toml` (ci-test-assets ersetzt „827 MB payloads“). Liegt über der reinen GGUF-Summe, weil Tokenizer und Binary mitzählen. |

CPU-Inferenz kopiert die quantisierten Gewichte aus dem mmap in den Prozess
(`CpuEmbeddingModel::from_gguf`, `CpuQwen35Model::load`). Das mmap selbst wird
danach fallengelassen. **CPU-RSS ist nicht gemessen.** Die Planung nimmt deshalb
die Workspace-Konstanten, die über der gemessenen CUDA-Host-RSS liegen.

Ein Modell wird nur eingeplant, wenn `Rest` sein Budget deckt. Summaries kommen
erst dazu, wenn `Rest` Embedding und Summary zusammen deckt. Eine zweite
lebende Seite kommt erst dazu, wenn nach den eingeplanten Modellen noch einmal
die Browser-Reserve übrig ist. Das ist bei 1–4 GiB nicht der Fall: eine zweite
schwere Seite und beide Modelle passen in dieser Rechnung nicht gleichzeitig.

## Stufen

| Budget | Rest | Embedding | Summary | Lebende Tabs (global) | Sessions (global) | Tab-Swap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GiB (`1g`, alles unter 2 GiB, z. B. `1536m`) | 0 | aus | aus | 1 | 1 | ja |
| 2 GiB | 512 MiB | aus | aus | 1 | 1 | ja |
| 3 GiB | 1536 MiB | lazy, CPU, Modell-TTL 60 s, Prozess-TTL 180 s | aus | 1 | 1 | ja |
| 4 GiB und darüber | 2560 MiB | lazy, CPU, TTL 30 s / 120 s | lazy, CPU, TTL 30 s / 120 s | 1 | 2 | ja |

1 GiB und 2 GiB treffen dieselbe Modellentscheidung. 512 MiB Rest liegen unter
dem 768-MiB-Embedding-Workspace. Der Unterschied ist die Marge: bei 2 GiB
bleiben 512 MiB für OS und Seitenrauschen, ohne ein Modell zu starten. Eine
eigene „2-GiB-lädt-trotzdem-ein-Modell“-Stufe würde die Reserve unterlaufen.

Geschätzte Spitze, Planung, nicht gemessen:

| Stufe | Browser | Embedding | Summary | Summe der Planung |
| --- | --- | --- | --- | --- |
| 1 GiB | 1,25 GiB Reserve, real oft der ganze Rechner | 0 | 0 | über dem Rechner, siehe offene Punkte |
| 2 GiB | 1,25 GiB | 0 | 0 | 1,25 GiB + Slack 0,25 GiB = 1,5 GiB, etwa 0,5 GiB Puffer |
| 3 GiB | 1,25 GiB | 0,75 GiB | 0 | 2,25 GiB + Slack |
| 4 GiB | 1,25 GiB | 0,75 GiB | 1,0 GiB | 3,25 GiB + Slack, wenn beide Modelle gleichzeitig resident sind |

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
| Triage | Fail-open: jeder Span `read: true`. Außerhalb von `summarize_daemon.rs` gibt es keinen Aufrufer. | `triage_spans_via_daemon` |
| Gecachte Summaries | werden bei abgeschaltetem Summarizer nicht gelesen, damit kein Modellschlüssel und kein Daemon entsteht | bewusste Grenze |

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

## Offene Punkte, die eine Messung auf Ubuntu brauchen

1. CPU-RSS von Embedding-Daemon und Summarize-Daemon auf einer Maschine ohne
   GPU, kalt und warm, plus RSS nachdem `model_ttl` das Modell verworfen hat.
   Die CUDA-Zahlen oben sind eine Untergrenze nur für den Host-Anteil mit
   Gewichten auf der GPU.
2. RSS des Content-Prozesses mit null, einer und zwei echten Seiten, headless,
   Software-Rendering 1280×720. Der 1,25-GiB-Kommentar hat keine
   Linux-Messreihe. Eine schwere Seite kann einen 1-GiB-Rechner auch ohne
   Modelle sprengen. Das Budget kann Servo nicht kleiner machen.
3. Ob `page.close` den Speicher des WebView im Servo-Prozess wirklich
   zurückgibt oder nur den Rust-Handle droppt.
4. Ein Wiederherstellungsversuch auf einer Seite mit HttpOnly-Cookie,
   `localStorage`, Scroll und History: was nach Swap noch stimmt.
5. Ob zwei gleichzeitig residente Daemons auf 4 GiB (Plan 3,25 GiB plus Slack
   und OS) vom OOM-Killer getroffen werden. Ein harter Cross-Daemon-Lock ist
   nicht eingebaut; es gibt nur kurze TTLs und kein Summary-Prewarm während
   der Suche.
6. zram oder Swap auf dem Host. Das ist Betriebssystem, kein greppy-Schalter.
