#!/usr/bin/env python3
"""Generate the ~100-task benchmark question bank (tasks.json).

Every task carries a machine-checkable ``ground_truth`` and a ``check``
descriptor that ``verify_tasks.py`` runs against the live grepplus graph, so
the bank is provably answerable (not hand-asserted). Task shapes are restricted
to the query patterns that each language's extractor actually resolves
(established empirically in the corpus probe):

    Rust / Python / Go / Java : full cross-file CALLS (who-calls, callees,
                                trace), plus find-usages / search-code.
    JavaScript / TypeScript   : the verified subset (same-file CALLS, the
                                resolved cross-file CALLS, IMPORTS, search).

Types:
    locate    -- "where is X / who calls X / find usages of X"
    research  -- "how does subsystem X work / what breaks if Y changes /
                 trace data flow A->B / which module owns Z"

Run ``verify_tasks.py`` after generating to drop any task whose ground truth
does not resolve, guaranteeing tasks.json is 100% answerable.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "tasks.json"

tasks = []


def add(repo, lang, ttype, q, ground_truth, check):
    tasks.append(
        {
            "id": f"t{len(tasks) + 1:03d}",
            "repo": repo,
            "lang": lang,
            "type": ttype,
            "q": q,
            "ground_truth": ground_truth,
            "check": check,
        }
    )


# ==========================================================================
# RUST  (rust_medium) -- full cross-file CALLS graph
# ==========================================================================
R = "rust_medium"
# locate: who-calls the leaf hub
add(
    R, "rust", "locate",
    "Who calls the function compute_checksum? List the calling functions.",
    "compute_checksum is called by merge_checksums, by normalize_record, and "
    "by every process_svcNN service function (74 call sites total).",
    {"kind": "who_calls", "symbol": "compute_checksum",
     "expect_members": ["normalize_record", "merge_checksums", "process_svc00"],
     "min_count": 70},
)
add(
    R, "rust", "locate",
    "What are the direct callees of normalize_record? List them with file:line.",
    "normalize_record calls clamp_value (src/core/clampval.rs) and "
    "compute_checksum (src/core/checksum.rs).",
    {"kind": "callees", "symbol": "normalize_record",
     "expect_members": ["clamp_value", "compute_checksum"], "min_count": 2},
)
add(
    R, "rust", "locate",
    "Who calls normalize_record? Name the calling functions.",
    "Every process_svcNN service function calls normalize_record (72 callers).",
    {"kind": "who_calls", "symbol": "normalize_record",
     "expect_members": ["process_svc00", "process_svc01"], "min_count": 70},
)
add(
    R, "rust", "locate",
    "What does run_pipeline call directly? List its callees.",
    "run_pipeline calls merge_checksums and every process_svcNN service entry.",
    {"kind": "callees", "symbol": "run_pipeline",
     "expect_members": ["merge_checksums", "process_svc00"], "min_count": 70},
)
add(
    R, "rust", "locate",
    "Find the definition site of clamp_value.",
    "clamp_value is defined in src/core/clampval.rs.",
    {"kind": "search_symbols", "query": "clamp_value",
     "expect_file": "clampval.rs"},
)
add(
    R, "rust", "locate",
    "Where is the FNV checksum loop implemented? Return the code.",
    "The rolling-checksum loop is in compute_checksum in src/core/checksum.rs.",
    {"kind": "search_code", "query": "wrapping_mul",
     "expect_file": "checksum.rs"},
)
# research
add(
    R, "rust", "research",
    "How does the normalisation subsystem work? Summarise the key functions "
    "and the core leaves they depend on.",
    "normalize_record (src/core/normalize.rs) is the junction: it clamps the "
    "score via clamp_value (clampval.rs) and hashes the payload via "
    "compute_checksum (checksum.rs), returning a Record. Services call it.",
    {"kind": "callees", "symbol": "normalize_record",
     "expect_members": ["clamp_value", "compute_checksum"], "min_count": 2},
)
add(
    R, "rust", "research",
    "What would break if the signature of compute_checksum changed? List the "
    "impacted call sites / functions.",
    "All callers of compute_checksum break: merge_checksums, normalize_record, "
    "and every process_svcNN (74 sites).",
    {"kind": "who_calls", "symbol": "compute_checksum",
     "expect_members": ["normalize_record", "merge_checksums"], "min_count": 70},
)
add(
    R, "rust", "research",
    "Trace the data flow from run_pipeline down to the core checksum leaf. "
    "Give the call path.",
    "run_pipeline -> process_svcNN -> normalize_record -> compute_checksum "
    "(in src/core/checksum.rs).",
    {"kind": "path", "frm": "run_pipeline", "to": "compute_checksum"},
)
add(
    R, "rust", "research",
    "Trace from run_pipeline to clamp_value. Give the path of functions.",
    "run_pipeline -> process_svcNN -> normalize_record -> clamp_value.",
    {"kind": "path", "frm": "run_pipeline", "to": "clamp_value"},
)
add(
    R, "rust", "research",
    "Which module owns record normalisation and what are its entry points "
    "(the functions other layers call into it)?",
    "src/core/normalize.rs owns normalisation; its entry point normalize_record "
    "is called by every service (process_svcNN).",
    {"kind": "who_calls", "symbol": "normalize_record",
     "expect_members": ["process_svc00"], "min_count": 70},
)
add(
    R, "rust", "research",
    "If you changed how scores are clamped, which single function must you edit "
    "and which functions call it?",
    "Edit clamp_value (src/core/clampval.rs); it is called by normalize_record.",
    {"kind": "who_calls", "symbol": "clamp_value",
     "expect_members": ["normalize_record"], "min_count": 1},
)

# ==========================================================================
# PYTHON  (python_large) -- full cross-file CALLS graph
# ==========================================================================
P = "python_large"
add(
    P, "python", "locate",
    "Who calls validate_currency? Name the calling functions.",
    "post-path to_minor_units? No -- validate_currency is called by money's "
    "to_minor_units and by every process_svcNNN (416 callers).",
    {"kind": "who_calls", "symbol": "validate_currency",
     "expect_members": ["to_minor_units", "process_svc000"], "min_count": 400},
)
add(
    P, "python", "locate",
    "What are the direct callees of to_minor_units?",
    "to_minor_units calls validate_amount and validate_currency "
    "(app/core/validate.py).",
    {"kind": "callees", "symbol": "to_minor_units",
     "expect_members": ["validate_amount", "validate_currency"], "min_count": 2},
)
add(
    P, "python", "locate",
    "Who calls post_entry? List the calling functions.",
    "Every process_svcNNN service function calls post_entry (415 callers).",
    {"kind": "who_calls", "symbol": "post_entry",
     "expect_members": ["process_svc000", "process_svc001"], "min_count": 400},
)
add(
    P, "python", "locate",
    "What are the direct callees of post_entry?",
    "post_entry calls to_minor_units (app/core/money.py).",
    {"kind": "callees", "symbol": "post_entry",
     "expect_members": ["to_minor_units"], "min_count": 1},
)
add(
    P, "python", "locate",
    "Find the definition of validate_amount.",
    "validate_amount is defined in app/core/validate.py.",
    {"kind": "search_symbols", "query": "validate_amount",
     "expect_file": "validate.py"},
)
add(
    P, "python", "locate",
    "Where is a major-to-minor currency conversion (amount * 100) implemented? "
    "Return the code.",
    "In to_minor_units in app/core/money.py (int(round(amount * 100))).",
    {"kind": "search_code", "query": "to_minor_units",
     "expect_file": "money.py"},
)
add(
    P, "python", "research",
    "How does the money subsystem convert amounts to minor units, and what "
    "validation does it perform first?",
    "to_minor_units (app/core/money.py) validates via validate_amount and "
    "validate_currency, then returns int(round(amount*100)).",
    {"kind": "callees", "symbol": "to_minor_units",
     "expect_members": ["validate_amount", "validate_currency"], "min_count": 2},
)
add(
    P, "python", "research",
    "What would break if post_entry changed its return shape? List impacted "
    "callers.",
    "Every process_svcNNN consumes post_entry's return (415 callers impacted).",
    {"kind": "who_calls", "symbol": "post_entry",
     "expect_members": ["process_svc000"], "min_count": 400},
)
add(
    P, "python", "research",
    "Trace the data flow from run_pipeline down to to_minor_units. Give the "
    "call path.",
    "run_pipeline -> process_svcNNN -> post_entry -> to_minor_units.",
    {"kind": "path", "frm": "run_pipeline", "to": "to_minor_units"},
)
add(
    P, "python", "research",
    "Trace from run_pipeline to validate_amount.",
    "run_pipeline -> process_svcNNN -> post_entry -> to_minor_units -> "
    "validate_amount.",
    {"kind": "path", "frm": "run_pipeline", "to": "validate_amount"},
)
add(
    P, "python", "research",
    "Which module owns the ledger and what is its public entry point?",
    "app/core/ledger.py owns the ledger; post_entry is the entry point called "
    "by every service.",
    {"kind": "who_calls", "symbol": "post_entry",
     "expect_members": ["process_svc000"], "min_count": 400},
)
add(
    P, "python", "research",
    "If currency validation rules change, which function must be edited and "
    "who depends on it?",
    "Edit validate_currency (app/core/validate.py); to_minor_units and every "
    "service depend on it.",
    {"kind": "who_calls", "symbol": "validate_currency",
     "expect_members": ["to_minor_units"], "min_count": 400},
)

# ==========================================================================
# GO  (go_small) -- full cross-file CALLS graph
# ==========================================================================
G = "go_small"
add(
    G, "go", "locate",
    "Who calls NormalizeRecord? Name the callers.",
    "Every ProcessSvcNN calls NormalizeRecord (9 callers).",
    {"kind": "who_calls", "symbol": "NormalizeRecord",
     "expect_members": ["ProcessSvc00", "ProcessSvc01"], "min_count": 8},
)
add(
    G, "go", "locate",
    "What are the direct callees of NormalizeRecord?",
    "NormalizeRecord calls ClampInt (core/clamp.go) and ComputeHash "
    "(core/hash.go).",
    {"kind": "callees", "symbol": "NormalizeRecord",
     "expect_members": ["ClampInt", "ComputeHash"], "min_count": 2},
)
add(
    G, "go", "locate",
    "Who calls ComputeHash? List the callers.",
    "ComputeHash is called by NormalizeRecord and by every ProcessSvcNN.",
    {"kind": "who_calls", "symbol": "ComputeHash",
     "expect_members": ["NormalizeRecord", "ProcessSvc00"], "min_count": 8},
)
add(
    G, "go", "locate",
    "What does RunPipeline call directly?",
    "RunPipeline calls every ProcessSvcNN.",
    {"kind": "callees", "symbol": "RunPipeline",
     "expect_members": ["ProcessSvc00"], "min_count": 8},
)
add(
    G, "go", "locate",
    "Find the definition of ClampInt and return the clamping code.",
    "ClampInt is defined in core/clamp.go.",
    {"kind": "search_code", "query": "ClampInt", "expect_file": "clamp.go"},
)
add(
    G, "go", "research",
    "How does NormalizeRecord build a Record? Summarise the leaves it calls.",
    "NormalizeRecord (core/normalize.go) sets Score via ClampInt and Hash via "
    "ComputeHash.",
    {"kind": "callees", "symbol": "NormalizeRecord",
     "expect_members": ["ClampInt", "ComputeHash"], "min_count": 2},
)
add(
    G, "go", "research",
    "What would break if ComputeHash changed? List the impacted functions.",
    "NormalizeRecord and every ProcessSvcNN call ComputeHash.",
    {"kind": "who_calls", "symbol": "ComputeHash",
     "expect_members": ["NormalizeRecord"], "min_count": 8},
)
add(
    G, "go", "research",
    "Trace the call path from RunPipeline to ComputeHash.",
    "RunPipeline -> ProcessSvcNN -> NormalizeRecord -> ComputeHash.",
    {"kind": "path", "frm": "RunPipeline", "to": "ComputeHash"},
)
add(
    G, "go", "research",
    "Which file owns record normalisation and what is its entry point?",
    "core/normalize.go owns it; NormalizeRecord is the entry point.",
    {"kind": "who_calls", "symbol": "NormalizeRecord",
     "expect_members": ["ProcessSvc00"], "min_count": 8},
)

# ==========================================================================
# JAVA  (java_medium) -- full cross-file CALLS graph
# ==========================================================================
J = "java_medium"
add(
    J, "java", "locate",
    "Who calls computeChecksum? List the calling methods.",
    "computeChecksum is called by normalizeRecord and by every processSvcNN.",
    {"kind": "who_calls", "symbol": "computeChecksum",
     "expect_members": ["normalizeRecord", "processSvc00"], "min_count": 70},
)
add(
    J, "java", "locate",
    "What are the direct callees of normalizeRecord?",
    "normalizeRecord calls clampValue (Clamp.java) and computeChecksum "
    "(Checksum.java).",
    {"kind": "callees", "symbol": "normalizeRecord",
     "expect_members": ["clampValue", "computeChecksum"], "min_count": 2},
)
add(
    J, "java", "locate",
    "Who calls normalizeRecord? Name the calling methods.",
    "Every processSvcNN calls normalizeRecord (76 callers).",
    {"kind": "who_calls", "symbol": "normalizeRecord",
     "expect_members": ["processSvc00", "processSvc01"], "min_count": 70},
)
add(
    J, "java", "locate",
    "What does runPipeline call directly?",
    "runPipeline calls every processSvcNN.",
    {"kind": "callees", "symbol": "runPipeline",
     "expect_members": ["processSvc00"], "min_count": 70},
)
add(
    J, "java", "locate",
    "Find the definition of clampValue and return the clamping code.",
    "clampValue is defined in src/main/java/corpus/core/Clamp.java.",
    {"kind": "search_code", "query": "clampValue", "expect_file": "Clamp.java"},
)
add(
    J, "java", "research",
    "How does the Normalizer class build a record? Summarise the core helpers "
    "it calls.",
    "Normalizer.normalizeRecord clamps via Clamp.clampValue and hashes via "
    "Checksum.computeChecksum.",
    {"kind": "callees", "symbol": "normalizeRecord",
     "expect_members": ["clampValue", "computeChecksum"], "min_count": 2},
)
add(
    J, "java", "research",
    "What would break if computeChecksum changed? List impacted methods.",
    "normalizeRecord and every processSvcNN call computeChecksum.",
    {"kind": "who_calls", "symbol": "computeChecksum",
     "expect_members": ["normalizeRecord"], "min_count": 70},
)
add(
    J, "java", "research",
    "Trace the call path from runPipeline to computeChecksum.",
    "runPipeline -> processSvcNN -> normalizeRecord -> computeChecksum.",
    {"kind": "path", "frm": "runPipeline", "to": "computeChecksum"},
)
add(
    J, "java", "research",
    "Trace from runPipeline to clampValue.",
    "runPipeline -> processSvcNN -> normalizeRecord -> clampValue.",
    {"kind": "path", "frm": "runPipeline", "to": "clampValue"},
)
add(
    J, "java", "research",
    "Which class owns normalisation and what is its entry point method?",
    "corpus.core.Normalizer owns it; normalizeRecord is the entry point called "
    "by every service.",
    {"kind": "who_calls", "symbol": "normalizeRecord",
     "expect_members": ["processSvc00"], "min_count": 70},
)

# ==========================================================================
# JAVASCRIPT  (js_small) -- verified subset only
# ==========================================================================
JS = "js_small"
add(
    JS, "javascript", "locate",
    "Who calls processSvc00? Name the caller.",
    "runPipeline (src/pipeline.js) calls processSvc00.",
    {"kind": "who_calls", "symbol": "processSvc00",
     "expect_members": ["runPipeline"], "min_count": 1},
)
add(
    JS, "javascript", "locate",
    "What does runPipeline call directly?",
    "runPipeline calls collect (same file) and every processSvcNN service.",
    {"kind": "callees", "symbol": "runPipeline",
     "expect_members": ["collect", "processSvc00"], "min_count": 5},
)
add(
    JS, "javascript", "locate",
    "Who calls collect?",
    "runPipeline calls collect (src/pipeline.js).",
    {"kind": "who_calls", "symbol": "collect",
     "expect_members": ["runPipeline"], "min_count": 1},
)
add(
    JS, "javascript", "locate",
    "What does normalizeRecord call in its own file?",
    "normalizeRecord calls buildRecord (src/core/normalize.js).",
    {"kind": "callees", "symbol": "normalizeRecord",
     "expect_members": ["buildRecord"], "min_count": 1},
)
add(
    JS, "javascript", "locate",
    "Find where normalizeRecord is referenced in service files. Return code.",
    "Service files require and call normalizeRecord; e.g. src/service/svc00.js.",
    {"kind": "search_code", "query": "normalizeRecord rec",
     "expect_file": "svc00.js"},
)
add(
    JS, "javascript", "locate",
    "Find the definition of clampScore.",
    "clampScore is defined in src/core/clamp.js.",
    {"kind": "search_symbols", "query": "clampScore", "expect_file": "clamp.js"},
)
add(
    JS, "javascript", "research",
    "Which functions does runPipeline drive, and how does it assemble its "
    "result? Summarise.",
    "runPipeline calls every processSvcNN to build rows, then calls collect to "
    "filter them (src/pipeline.js).",
    {"kind": "callees", "symbol": "runPipeline",
     "expect_members": ["collect", "processSvc00"], "min_count": 5},
)
add(
    JS, "javascript", "research",
    "What would break inside the pipeline if collect changed? Identify its "
    "caller.",
    "runPipeline depends on collect; it is the only in-file caller.",
    {"kind": "who_calls", "symbol": "collect",
     "expect_members": ["runPipeline"], "min_count": 1},
)
add(
    JS, "javascript", "research",
    "Which file owns the service-dispatch pipeline and what is its entry point?",
    "src/pipeline.js owns it; runPipeline is the entry point that drives every "
    "processSvcNN.",
    {"kind": "callees", "symbol": "runPipeline",
     "expect_members": ["processSvc00"], "min_count": 5},
)

# ==========================================================================
# TYPESCRIPT  (ts_large) -- verified subset only
# ==========================================================================
TS = "ts_large"
add(
    TS, "typescript", "locate",
    "Who calls validateAmount in the validators file?",
    "validateRecord (src/core/validate.ts) calls validateAmount.",
    {"kind": "who_calls", "symbol": "validateAmount",
     "expect_members": ["validateRecord"], "min_count": 1},
)
add(
    TS, "typescript", "locate",
    "What does validateRecord call?",
    "validateRecord calls validateAmount and validateCode (src/core/validate.ts).",
    {"kind": "callees", "symbol": "validateRecord",
     "expect_members": ["validateAmount", "validateCode"], "min_count": 2},
)
add(
    TS, "typescript", "locate",
    "What does toMinorUnits call in its own file?",
    "toMinorUnits calls roundMinor (src/core/money.ts).",
    {"kind": "callees", "symbol": "toMinorUnits",
     "expect_members": ["roundMinor"], "min_count": 1},
)
add(
    TS, "typescript", "locate",
    "What does postEntry call in its own file?",
    "postEntry calls makeEntry (src/core/ledger.ts).",
    {"kind": "callees", "symbol": "postEntry",
     "expect_members": ["makeEntry"], "min_count": 1},
)
add(
    TS, "typescript", "locate",
    "Find the definition of processSvc100. Return its code.",
    "processSvc100 is defined in src/service/svc100.ts.",
    {"kind": "search_code", "query": "processSvc100", "expect_file": "svc100.ts"},
)
add(
    TS, "typescript", "locate",
    "Where is the major-to-minor unit rounding implemented? Return the code.",
    "roundMinor in src/core/money.ts (Math.round(amount * 100)).",
    {"kind": "search_code", "query": "Math.round amount", "expect_file": "money.ts"},
)
add(
    TS, "typescript", "research",
    "How does the validation subsystem combine its checks? Summarise the "
    "functions involved.",
    "validateRecord (src/core/validate.ts) ANDs validateAmount and validateCode.",
    {"kind": "callees", "symbol": "validateRecord",
     "expect_members": ["validateAmount", "validateCode"], "min_count": 2},
)
add(
    TS, "typescript", "research",
    "What would break if roundMinor changed? Identify its in-file caller.",
    "toMinorUnits depends on roundMinor (src/core/money.ts).",
    {"kind": "who_calls", "symbol": "roundMinor",
     "expect_members": ["toMinorUnits"], "min_count": 1},
)
add(
    TS, "typescript", "research",
    "Which file owns ledger entry construction and what is its entry point?",
    "src/core/ledger.ts owns it; postEntry is the entry point (it calls "
    "makeEntry and toMinorUnits).",
    {"kind": "callees", "symbol": "postEntry",
     "expect_members": ["makeEntry"], "min_count": 1},
)
add(
    TS, "typescript", "research",
    "Trace, within ledger.ts and money.ts, how postEntry reaches the rounding "
    "helper. Give the in-file call chain.",
    "postEntry -> toMinorUnits (money.ts) -> roundMinor; postEntry also calls "
    "makeEntry.",
    {"kind": "callees", "symbol": "toMinorUnits",
     "expect_members": ["roundMinor"], "min_count": 1},
)


# ==========================================================================
# BATCH: per-service locate tasks (who-calls processSvcNN -> pipeline entry)
# These are individually verifiable and add breadth across the corpus.
# ==========================================================================
# Rust: process_svcNN -> run_pipeline
for i in (3, 9, 17, 23, 41, 58, 70):
    add(
        R, "rust", "locate",
        f"Who calls process_svc{i:02d}? Name the caller and its file.",
        f"run_pipeline (src/app/pipeline.rs) calls process_svc{i:02d}.",
        {"kind": "who_calls", "symbol": f"process_svc{i:02d}",
         "expect_members": ["run_pipeline"], "min_count": 1},
    )
# Python: process_svcNNN -> run_pipeline
for i in (5, 42, 113, 200, 314, 400):
    add(
        P, "python", "locate",
        f"Who calls process_svc{i:03d}? Name the caller.",
        f"run_pipeline (app/pipeline.py) calls process_svc{i:03d}.",
        {"kind": "who_calls", "symbol": f"process_svc{i:03d}",
         "expect_members": ["run_pipeline"], "min_count": 1},
    )
# Java: processSvcNN -> runPipeline
for i in (2, 19, 33, 47, 60, 75):
    add(
        J, "java", "locate",
        f"Who calls processSvc{i:02d}? Name the calling method.",
        f"runPipeline (corpus.app.Pipeline) calls processSvc{i:02d}.",
        {"kind": "who_calls", "symbol": f"processSvc{i:02d}",
         "expect_members": ["runPipeline"], "min_count": 1},
    )
# Go: ProcessSvcNN -> RunPipeline
for i in (1, 4, 7):
    add(
        G, "go", "locate",
        f"Who calls ProcessSvc{i:02d}?",
        f"RunPipeline (app/pipeline.go) calls ProcessSvc{i:02d}.",
        {"kind": "who_calls", "symbol": f"ProcessSvc{i:02d}",
         "expect_members": ["RunPipeline"], "min_count": 1},
    )
# JS: processSvcNN -> runPipeline
for i in (1, 3, 6):
    add(
        JS, "javascript", "locate",
        f"Who calls processSvc{i:02d}?",
        f"runPipeline (src/pipeline.js) calls processSvc{i:02d}.",
        {"kind": "who_calls", "symbol": f"processSvc{i:02d}",
         "expect_members": ["runPipeline"], "min_count": 1},
    )

# BATCH: find-usages of the Record type (Rust) -- TYPE_REF coverage
add(
    R, "rust", "locate",
    "Find every place the Record type is referenced. List the functions.",
    "Record (src/core/normalize.rs) is referenced in normalize_record and in "
    "every process_svcNN service signature.",
    {"kind": "find_usages", "symbol": "Record",
     "expect_members": ["normalize_record", "process_svc00"], "min_count": 70},
)
add(
    R, "rust", "research",
    "Which functions depend on the Record type, and what does that imply for "
    "changing its fields?",
    "normalize_record produces Record and every process_svcNN returns it; "
    "changing fields impacts all of them.",
    {"kind": "find_usages", "symbol": "Record",
     "expect_members": ["normalize_record"], "min_count": 70},
)

# BATCH: search-code snippet tasks across repos (code-returning queries)
add(
    R, "rust", "locate",
    "Show the code that merges two checksums.",
    "merge_checksums in src/core/checksum.rs.",
    {"kind": "search_code", "query": "merge_checksums", "expect_file": "checksum.rs"},
)
add(
    P, "python", "locate",
    "Show the code that posts a ledger entry.",
    "post_entry in app/core/ledger.py.",
    {"kind": "search_code", "query": "post_entry", "expect_file": "ledger.py"},
)
add(
    J, "java", "locate",
    "Show the FNV checksum fold loop.",
    "Checksum.computeChecksum in Checksum.java.",
    {"kind": "search_code", "query": "Fold byte array checksum", "expect_file": "Checksum.java"},
)
add(
    G, "go", "locate",
    "Show the FNV hash fold loop.",
    "ComputeHash in core/hash.go.",
    {"kind": "search_code", "query": "ComputeHash", "expect_file": "hash.go"},
)
add(
    TS, "typescript", "locate",
    "Show the ledger entry construction helper.",
    "makeEntry in src/core/ledger.ts.",
    {"kind": "search_code", "query": "makeEntry account minor", "expect_file": "ledger.ts"},
)

# BATCH: more research / ownership / impact tasks
add(
    P, "python", "research",
    "Summarise the layering of the python app: which layer calls which, from "
    "the pipeline down to the validators.",
    "pipeline -> service (process_svcNNN) -> ledger (post_entry) -> money "
    "(to_minor_units) -> validate (validate_amount/validate_currency).",
    {"kind": "path", "frm": "run_pipeline", "to": "validate_currency"},
)
add(
    J, "java", "research",
    "Which core class is the single most depended-on leaf (called from the most "
    "sites), and what is its method?",
    "corpus.core.Checksum.computeChecksum is the most-called leaf (normalizeRecord "
    "plus every processSvcNN).",
    {"kind": "who_calls", "symbol": "computeChecksum",
     "expect_members": ["normalizeRecord"], "min_count": 70},
)
add(
    G, "go", "research",
    "If the Record struct grew a field, which functions construct or return it?",
    "NormalizeRecord constructs Record; every ProcessSvcNN returns it.",
    {"kind": "who_calls", "symbol": "NormalizeRecord",
     "expect_members": ["ProcessSvc00"], "min_count": 8},
)
add(
    TS, "typescript", "research",
    "How does money.ts guard against invalid input before rounding?",
    "toMinorUnits calls validateRecord (which ANDs validateAmount/validateCode) "
    "before calling roundMinor.",
    {"kind": "callees", "symbol": "toMinorUnits",
     "expect_members": ["roundMinor"], "min_count": 1},
)
add(
    R, "rust", "research",
    "Name the single function every service routes through before touching the "
    "core leaves, and prove it by its callees.",
    "normalize_record is the shared junction; its callees are clamp_value and "
    "compute_checksum.",
    {"kind": "callees", "symbol": "normalize_record",
     "expect_members": ["clamp_value", "compute_checksum"], "min_count": 2},
)
add(
    P, "python", "research",
    "Which validator is reused both by the money layer and directly by every "
    "service, and what does that say about its blast radius?",
    "validate_currency is called by to_minor_units and by every process_svcNNN, "
    "so its blast radius spans the whole service layer.",
    {"kind": "who_calls", "symbol": "validate_currency",
     "expect_members": ["to_minor_units", "process_svc000"], "min_count": 400},
)


def main():
    json.dump(tasks, open(OUT, "w"), indent=1)
    by_type = {}
    by_repo = {}
    for t in tasks:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
        by_repo[t["repo"]] = by_repo.get(t["repo"], 0) + 1
    print(f"wrote {len(tasks)} tasks to {OUT}")
    print(f"by type: {by_type}")
    print(f"by repo: {by_repo}")


if __name__ == "__main__":
    main()
