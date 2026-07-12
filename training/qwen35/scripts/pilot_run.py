#!/usr/bin/env python3
"""Pilot: candidate teacher prompts x real functions -> MiniMax M3 -> format validation report.

Usage: MINIMAX_API_KEY=... python3 pilot_run.py pilot_functions.jsonl [prompt_ids...]
Writes pilot_results.jsonl and prints a compliance report.
"""
import json, os, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.minimax.io/v1/text/chatcompletion_v2"
KEY = os.environ.get("MINIMAX_API_KEY")  # required only when calling M3

PROMPTS = {
    # A: strict one-liner, rules only
    "A": """Write a one-line summary of the function below for a code-search index.
A coding agent reads ONLY this line to decide whether the function is worth opening, so be specific and discriminative: what it does, on what input/state, and the key side effect or return value. Never describe implementation steps.

Rules — your entire reply must be exactly one line:
- One English sentence, at most 120 characters.
- Start with a present-tense third-person verb (e.g. "Parses", "Returns", "Registers").
- Name identifiers only if they literally appear in the code.
- No prefix like "Summary:", no bullet, no quotes, no backticks, no code, no markdown, no ellipsis, no question.

Function ({lang}, {path}):
{source}""",
    # C: same rules + few-shots
    "C": """Write a one-line summary of the function below for a code-search index.
A coding agent reads ONLY this line to decide whether the function is worth opening, so be specific and discriminative: what it does, on what input/state, and the key side effect or return value. Never describe implementation steps.

Rules — your entire reply must be exactly one line:
- One English sentence, at most 120 characters.
- Start with a present-tense third-person verb (e.g. "Parses", "Returns", "Registers").
- Name identifiers only if they literally appear in the code.
- No prefix like "Summary:", no bullet, no quotes, no backticks, no code, no markdown, no ellipsis, no question.

Example function (python, app/cache.py):
def invalidate(self, key):
    with self._lock:
        entry = self._entries.pop(key, None)
    if entry and entry.on_evict:
        entry.on_evict(key)

Example reply:
Removes a cache entry by key under the lock and fires its on_evict callback if one is registered.

Example function (rust, src/pool.rs):
pub fn acquire(&self) -> Result<Conn, PoolError> {
    let mut inner = self.inner.lock();
    if let Some(c) = inner.idle.pop() { return Ok(c); }
    if inner.count < self.max { inner.count += 1; return Conn::connect(&self.url); }
    Err(PoolError::Exhausted)
}

Example reply:
Returns an idle pooled connection or opens a new one up to the max limit, failing with PoolError::Exhausted.

Function ({lang}, {path}):
{source}""",
    # D: word cap, hard-compression few-shot, explicit bare identifiers
    "D": """Write a one-line summary of the function below for a code-search index.
A coding agent reads ONLY this line to decide whether the function is worth opening, so be specific and discriminative: what it does, on what input/state, and the key side effect or return value. Never describe implementation steps. When the function does many things, keep only what best distinguishes it from other functions and drop the rest.

Rules — your entire reply must be exactly one line:
- One English sentence, at most 20 words and at most 120 characters. This is a hard limit; compress, do not truncate.
- Start with a present-tense third-person verb (e.g. "Parses", "Returns", "Registers").
- Name identifiers only if they literally appear in the code, written bare: no backticks, no quotes.
- No prefix like "Summary:", no bullet, no code, no markdown, no ellipsis, no question.

Example function (python, app/cache.py):
def invalidate(self, key):
    with self._lock:
        entry = self._entries.pop(key, None)
    if entry and entry.on_evict:
        entry.on_evict(key)

Example reply:
Removes a cache entry by key under the lock and fires its on_evict callback if registered.

Example function (rust, src/pool.rs):
pub fn acquire(&self) -> Result<Conn, PoolError> {
    let mut inner = self.inner.lock();
    if let Some(c) = inner.idle.pop() { return Ok(c); }
    if inner.count < self.max { inner.count += 1; return Conn::connect(&self.url); }
    Err(PoolError::Exhausted)
}

Example reply:
Returns an idle pooled connection or opens a new one up to the max limit, else PoolError::Exhausted.

Example function (java, JsonScanner.java) — a long function doing many things; the reply compresses to the distinguishing core:
private int peekNumber() throws IOException {
    // 60 lines: scans buffered chars, handles sign, mantissa, exponent,
    // detects overflow into BigDecimal, stores parsed value in fields,
    // returns a PEEKED_* constant...
}

Example reply:
Scans the buffered input for a numeric literal, storing the parsed value and returning a PEEKED_* token type.

Function ({lang}, {path}):
{source}""",
}

PROMPTS["E"] = (
    PROMPTS["D"]
    .replace(
        "- One English sentence, at most 20 words and at most 120 characters. This is a hard limit; compress, do not truncate.",
        "- One English sentence, at most 18 words and at most 115 characters. Hard limit: silently rewrite shorter until it fits; compress, never truncate.",
    )
    .replace(
        '- Start with a present-tense third-person verb (e.g. "Parses", "Returns", "Registers").',
        '- The FIRST word is a present-tense third-person verb (e.g. "Parses", "Returns", "Registers") - never an adverb, noun, or article.',
    )
)

PROMPTS["F"] = PROMPTS["E"].replace(
    "Function ({lang}, {path}):",
    """Before answering, verify in your private reasoning, in this order: (1) exactly one line, (2) the first word is a third-person verb, (3) at most 18 words, (4) at most 115 characters — count them. If any check fails, rewrite and re-verify. Only then reply.

Function ({lang}, {path}):""",
)

# Product's postprocess has no verb-first rule; accept an optional leading -ly
# adverb before the verb ("Safely casts...") instead of churning the prompt.
PROMPTS["G"] = PROMPTS["F"].replace(
    "- One English sentence, at most 18 words and at most 115 characters. Hard limit: silently rewrite shorter until it fits; compress, never truncate.",
    "- One English sentence, at most 15 words and at most 110 characters. Hard limit: silently rewrite shorter until it fits; compress, never truncate. Never append parenthetical lists; name at most two concrete details.",
).replace(
    "(3) at most 18 words, (4) at most 115 characters",
    "(3) at most 15 words, (4) at most 110 characters, (5) no parenthetical list",
)

PROMPTS["H"] = PROMPTS["G"].replace(
    "Never describe implementation steps. When the function does many things, keep only what best distinguishes it from other functions and drop the rest.",
    "Never describe implementation steps. When the function does many things, keep only what best distinguishes it from other functions and drop the rest. Judge ONLY what is visible inside this function: if its true purpose depends on code you cannot see, plainly state the visible behavior and stay vague about intent - never guess or invent what it is \"really\" for.",
)

PROMPTS["I"] = """Write a brief for the function below, for a code-search index.
A coding agent reads ONLY this brief to decide whether the function is worth opening, so be specific and discriminative: what it does, on what input/state, and the key side effects or return value. Never describe implementation steps. Judge ONLY what is visible inside this function: if its true purpose depends on code you cannot see, plainly state the visible behavior and stay vague about intent - never guess or invent what it is "really" for.

Output format - 1 to 3 lines, nothing else:
- Each line is ONE English sentence of at most 18 words and 120 characters, stating one distinct fact.
- Line 1 states the primary purpose. Add lines 2 and 3 ONLY for genuinely distinct behavior worth knowing before opening: a side effect, an edge case, an error path. A simple function gets exactly one line.
- Every line's FIRST word is a present-tense third-person verb (e.g. "Parses", "Returns", "Registers") - never an adverb, noun, or article.
- Name identifiers only if they literally appear in the code, written bare: no backticks, no quotes.
- No prefix like "Summary:", no bullets or numbering, no code, no markdown, no ellipsis, no question.

Example function (python, app/model.py):
def user_count(self):
    return len(self._users)

Example reply:
Returns the number of entries in _users.

Example function (rust, src/pool.rs):
pub fn acquire(&self) -> Result<Conn, PoolError> {
    let mut inner = self.inner.lock();
    if let Some(c) = inner.idle.pop() { return Ok(c); }
    if inner.count < self.max { inner.count += 1; return Conn::connect(&self.url); }
    Err(PoolError::Exhausted)
}

Example reply:
Returns an idle pooled connection or opens a new one, failing with PoolError::Exhausted at the max limit.
Increments the pool connection count under the lock when it opens a new connection.

Example function (java, JsonScanner.java) - long, several distinct behaviors:
private int peekNumber() throws IOException {
    // 60 lines: scans buffered chars, handles sign, mantissa, exponent,
    // detects overflow into BigDecimal, stores parsed value in fields,
    // returns a PEEKED_* constant...
}

Example reply:
Scans the buffered input for a numeric literal and returns a PEEKED_* token type.
Stores the parsed long or decimal length in fields as a side effect.
Falls back from long to decimal handling on overflow.

Before answering, verify in your private reasoning: (1) 1 to 3 lines, one sentence each, (2) every line starts with a third-person verb, (3) each line at most 18 words and 120 characters, (4) extra lines only for genuinely distinct facts. If any check fails, rewrite and re-verify. Only then reply.

Function ({lang}, {path}):
{source}"""

PROMPTS["J"] = """Write a brief for the function below, for a code-search index.
A coding agent reads ONLY this brief to decide whether the function is worth opening. State what the function does, on what input/state, and the key side effect or return value. Never describe implementation steps. Judge ONLY what is visible inside this function: if its true purpose depends on code you cannot see, plainly state the visible behavior and stay vague about intent - never guess or invent what it is "really" for.

Keep it deliberately SIMPLE. These briefs train a very small model: plain everyday words, simple subject-verb-object sentences, exactly one fact per sentence. When the function does many things, DROP detail rather than packing it into a clever dense sentence. Simple and correct beats complete.

Output format - 1 to 3 lines, nothing else:
- Each line is ONE simple English sentence of at most 14 words and 100 characters, stating one fact.
- Line 1 states the primary purpose. Add lines 2 and 3 ONLY for a clearly distinct side effect, edge case, or error path worth knowing before opening. A simple function gets exactly one line. When unsure, use fewer lines.
- Every line's FIRST word is a present-tense third-person verb (e.g. "Parses", "Returns", "Registers") - never an adverb, noun, or article.
- Mention at most two identifiers per line, only the most prominent ones, only if they literally appear in the code, written bare: no backticks, no quotes.
- No prefix like "Summary:", no bullets or numbering, no code, no markdown, no ellipsis, no question.

Example function (python, app/model.py):
def user_count(self):
    return len(self._users)

Example reply:
Returns the number of entries in _users.

Example function (rust, src/pool.rs):
pub fn acquire(&self) -> Result<Conn, PoolError> {
    let mut inner = self.inner.lock();
    if let Some(c) = inner.idle.pop() { return Ok(c); }
    if inner.count < self.max { inner.count += 1; return Conn::connect(&self.url); }
    Err(PoolError::Exhausted)
}

Example reply:
Returns an idle connection from the pool or opens a new one up to the max limit.
Fails with PoolError::Exhausted when the pool is full.

Example function (java, JsonScanner.java) - long, several distinct behaviors; the reply stays simple and drops the rest:
private int peekNumber() throws IOException {
    // 60 lines: scans buffered chars, handles sign, mantissa, exponent,
    // detects overflow into BigDecimal, stores parsed value in fields,
    // returns a PEEKED_* constant...
}

Example reply:
Scans the buffered input for a number and returns a PEEKED_* token type.
Stores the parsed value in fields as a side effect.

Before answering, verify in your private reasoning: (1) 1 to 3 lines, one simple sentence each, (2) every line starts with a third-person verb, (3) each line at most 14 words and 100 characters, (4) plain words, one fact per line, no dense packing. If any check fails, rewrite and re-verify. Only then reply.

Function ({lang}, {path}):
{source}"""

PROMPTS["K"] = """Write a brief for the function below, for a code-search index.
A coding agent reads ONLY this brief to decide whether the function is worth opening. The brief must NOT replace reading the code: it gives orientation, never facts precise enough to act on. Never enumerate parameters, options, or all cases; name what the function is for, not its exact contract. Never describe implementation steps. Judge ONLY what is visible inside this function: if its true purpose depends on code you cannot see, plainly state the visible behavior and stay vague about intent - never guess or invent what it is "really" for.

Keep it deliberately SIMPLE. These briefs train a very small model: plain everyday words, simple subject-verb-object sentences, exactly one fact per sentence. When the function does many things, DROP detail rather than packing it into a clever dense sentence. Simple and correct beats complete.

Output format - 1 or 2 lines, nothing else:
- Each line is ONE simple English sentence of at most 14 words and 100 characters.
- Line 1 states the primary purpose. Add a second line ONLY when the function has a second, clearly separate responsibility. When unsure, use one line.
- Every line's FIRST word is a present-tense third-person verb (e.g. "Parses", "Returns", "Registers") - never an adverb, noun, or article.
- Mention at most two identifiers per line, only the most prominent ones, only if they literally appear in the code, written bare: no backticks, no quotes.
- No prefix like "Summary:", no bullets or numbering, no code, no markdown, no ellipsis, no question.

Example function (python, app/model.py):
def user_count(self):
    return len(self._users)

Example reply:
Returns the number of entries in _users.

Example function (rust, src/pool.rs):
pub fn acquire(&self) -> Result<Conn, PoolError> {
    let mut inner = self.inner.lock();
    if let Some(c) = inner.idle.pop() { return Ok(c); }
    if inner.count < self.max { inner.count += 1; return Conn::connect(&self.url); }
    Err(PoolError::Exhausted)
}

Example reply:
Hands out connections from a bounded pool, reusing idle ones.

Example function (java, JsonScanner.java) - long, several distinct behaviors; the reply stays simple and drops the rest:
private int peekNumber() throws IOException {
    // 60 lines: scans buffered chars, handles sign, mantissa, exponent,
    // detects overflow into BigDecimal, stores parsed value in fields,
    // returns a PEEKED_* constant...
}

Example reply:
Scans the buffered input for a number and classifies what it found.
Stores the parsed value in fields as a side effect.

Before answering, verify in your private reasoning: (1) 1 or 2 lines, one simple sentence each, (2) every line starts with a third-person verb, (3) each line at most 14 words and 100 characters, (4) orientation only - nothing an agent could act on without reading the code. If any check fails, rewrite and re-verify. Only then reply.

Function ({lang}, {path}):
{source}"""

VERBISH = re.compile(r"^(?:[A-Z][a-z]+ly\s+)?[A-Za-z][\w-]*s\b")
IDENT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+|[A-Z][a-z0-9]+)\w*\b|\b\w+\(\)")

def validate(reply, source, max_lines=1, max_chars=140):
    lines = [l for l in reply.splitlines() if l.strip()]
    errs = []
    if not lines:
        return ["empty"]
    if len(lines) > max_lines:
        errs.append(f"lines={len(lines)}")
    for l in lines[:max_lines]:
        l = l.strip()
        if len(l) > max_chars:
            errs.append(f"len={len(l)}")
        if l[:1] in "-*•" or l.lower().startswith(("summary:", "purpose:", "answer:", "the function", "this function")):
            errs.append("prefix/bullet")
        if "```" in l or "`" in l:
            errs.append("backtick")
        if "..." in l or "…" in l:
            errs.append("ellipsis")
        if l.rstrip().endswith("?"):
            errs.append("question")
        if not VERBISH.match(l):
            errs.append(f"verb-first? '{l.split()[0] if l.split() else ''}'")
        for m in IDENT.finditer(l):
            tok = m.group(0).rstrip("()")
            if ("_" in tok or (tok[:1].islower() and any(c.isupper() for c in tok))) and tok.lower() not in source.lower():
                errs.append(f"ungrounded:{tok}")
    return errs

def call_m3(prompt):
    body = json.dumps({
        "model": "MiniMax-M3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 16000,
    }).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(API, data=body, headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            base = d.get("base_resp") or {}
            if base.get("status_code"):
                code, msg = base.get("status_code"), base.get("status_msg", "")
                if code in (1002, 1004, 1008) or "rate" in str(msg).lower():
                    return f"<<RATELIMIT: {code} {msg}>>", {}
                raise RuntimeError(f"minimax status {code}: {msg}")
            ch = d["choices"][0]
            return ch["message"]["content"], {**d.get("usage", {}), "finish": ch.get("finish_reason")}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return f"<<RATELIMIT: http429>>", {}
            if attempt == 2:
                return f"<<ERROR: {e}>>", {}
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                return f"<<ERROR: {e}>>", {}
            time.sleep(5 * (attempt + 1))

def main():
    fns = [json.loads(l) for l in open(sys.argv[1])]
    ids = sys.argv[2:] or list(PROMPTS)
    jobs = [(pid, fn) for pid in ids for fn in fns]

    def run(job):
        pid, fn = job
        prompt = PROMPTS[pid]
        for k in ("lang", "path", "source"):
            prompt = prompt.replace("{" + k + "}", fn[k])
        reply, usage = call_m3(prompt)
        max_lines = {"I": 3, "J": 3, "K": 2}.get(pid, 1)
        errs = validate(reply, fn["source"], max_lines=max_lines)
        first_errors = list(errs)
        repaired = False
        if errs:
            fix = (prompt + "\n\nYour previous reply was:\n" + reply
                   + "\n\nIt violated: " + ", ".join(errs)
                   + ". Reply again with ONLY the corrected reply, obeying every rule.")
            reply2, _ = call_m3(fix)
            errs2 = validate(reply2, fn["source"], max_lines=max_lines)
            if not errs2:
                reply, errs, repaired = reply2, errs2, True
        return {"prompt": pid, "lang": fn["lang"], "path": fn["path"], "name": fn["name"], "repaired": repaired,
                "first_errors": first_errors, "reply": reply, "errors": errs,
                "tokens": usage.get("total_tokens"), "finish": usage.get("finish")}

    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(run, jobs))

    with open("pilot_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    for pid in ids:
        rs = [r for r in results if r["prompt"] == pid]
        ok = [r for r in rs if not r["errors"]]
        n_rep = sum(1 for r in ok if r.get("repaired"))
        print(f"\n=== prompt {pid}: {len(ok)}/{len(rs)} clean ({len(ok) - n_rep} first-try, {n_rep} via repair) ===")
        by_lang = {}
        for r in rs:
            by_lang.setdefault(r["lang"], [0, 0])
            by_lang[r["lang"]][1] += 1
            if not r["errors"]:
                by_lang[r["lang"]][0] += 1
        print("  " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(by_lang.items())))
        for r in rs:
            if r["errors"]:
                print(f"  FAIL [{r['lang']} {r['name']}] {r['errors']} finish={r['finish']}\n    -> {r['reply'][:200]!r}")

if __name__ == "__main__":
    main()
