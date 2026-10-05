//! Bounded local C macro expansion for edit validation, never extraction.
//!
//! This does not implement an include search path or conditional evaluation.
//! Supported definitions are unconditional local object/function macros without
//! stringification, token pasting, or variadics. Unsupported used definitions
//! fail closed with an invocation coordinate rather than disabling syntax guards.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
struct Token {
    bytes: Vec<u8>,
    origin: usize,
    kind: Kind,
}
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ident,
    Trivia,
    Literal,
    Punct,
}
#[derive(Clone)]
struct Macro {
    parameters: Option<Vec<Vec<u8>>>,
    body: Vec<Token>,
    unsupported: bool,
}

/// A validation-only byte view. All original bytes remain untouched; expansion
/// coordinates are mapped to the start of their source macro invocation.
pub struct CPreprocessorView {
    pub bytes: Vec<u8>,
    origins: Vec<usize>,
    source_len: usize,
}
impl CPreprocessorView {
    pub fn source_offset(&self, expanded: usize) -> usize {
        self.origins
            .get(expanded)
            .copied()
            .unwrap_or(self.source_len)
    }
    pub fn expanded_offset(&self, source: usize) -> usize {
        self.origins.partition_point(|origin| *origin < source)
    }
}
#[derive(Debug)]
pub struct CPreprocessorError {
    pub offset: usize,
    pub reason: &'static str,
}

fn ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}
fn ident_cont(byte: u8) -> bool {
    ident_start(byte) || byte.is_ascii_digit()
}
fn tokenize(source: &[u8]) -> Vec<Token> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < source.len() {
        let start = at;
        let kind;
        if source[at].is_ascii_whitespace() {
            kind = Kind::Trivia;
            at += 1; // Keep each newline separate for logical directive lines.
        } else if source[at..].starts_with(b"/*") {
            kind = Kind::Trivia;
            at += 2;
            while at < source.len() && !source[at..].starts_with(b"*/") {
                at += 1;
            }
            at = (at + 2).min(source.len());
        } else if source[at..].starts_with(b"//") {
            kind = Kind::Trivia;
            while at < source.len() && source[at] != b'\n' {
                at += 1;
            }
        } else if matches!(source[at], b'\'' | b'"')
            || ([b"L".as_slice(), b"u", b"U", b"u8"].iter().any(|prefix| {
                source[at..].starts_with(prefix)
                    && source
                        .get(at + prefix.len())
                        .is_some_and(|b| matches!(*b, b'\'' | b'"'))
            }))
        {
            kind = Kind::Literal;
            while !matches!(source[at], b'\'' | b'"') {
                at += 1;
            }
            let quote = source[at];
            at += 1;
            while at < source.len() {
                let byte = source[at];
                at += 1;
                if byte == quote {
                    break;
                }
                if byte == b'\\' {
                    at = (at + 1).min(source.len());
                }
            }
        } else if source[at].is_ascii_digit()
            || (source[at] == b'.' && source.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            // A preprocessing number is one token, even when it contains a
            // spelling that also names a macro or is not a valid C number.
            kind = Kind::Literal;
            at += 1;
            while at < source.len() {
                if ident_cont(source[at]) || matches!(source[at], b'.' | b'\'') {
                    at += 1;
                } else if matches!(source[at], b'+' | b'-')
                    && matches!(source[at - 1], b'e' | b'E' | b'p' | b'P')
                {
                    at += 1;
                } else {
                    break;
                }
            }
        } else if ident_start(source[at]) {
            kind = Kind::Ident;
            at += 1;
            while at < source.len() && ident_cont(source[at]) {
                at += 1;
            }
        } else {
            kind = Kind::Punct;
            let operator = [
                b"%:%:".as_slice(),
                b">>=",
                b"<<=",
                b"...",
                b"->",
                b"++",
                b"--",
                b"<<",
                b">>",
                b"<=",
                b">=",
                b"==",
                b"!=",
                b"&&",
                b"||",
                b"*=",
                b"/=",
                b"%=",
                b"+=",
                b"-=",
                b"&=",
                b"^=",
                b"|=",
                b"##",
                b"<:",
                b":>",
                b"<%",
                b"%>",
                b"%:",
            ]
            .into_iter()
            .find(|operator| source[at..].starts_with(operator));
            at += operator.map_or(1, |operator| operator.len());
        }
        out.push(Token {
            bytes: source[start..at].to_vec(),
            origin: start,
            kind,
        });
    }
    out
}
fn significant(tokens: &[Token], mut at: usize) -> usize {
    while at < tokens.len() && tokens[at].kind == Kind::Trivia {
        at += 1;
    }
    at
}
fn define(tokens: &[Token]) -> Option<(Vec<u8>, Macro)> {
    let name_at = significant(tokens, 0);
    let name = tokens.get(name_at)?;
    if name.kind != Kind::Ident {
        return None;
    }
    let mut body_at = name_at + 1;
    let mut parameters = None;
    let mut unsupported = false;
    if tokens
        .get(body_at)
        .is_some_and(|t| t.bytes == b"(" && t.origin == name.origin + name.bytes.len())
    {
        body_at += 1;
        let mut params = Vec::new();
        loop {
            body_at = significant(tokens, body_at);
            let token = tokens.get(body_at)?;
            if token.bytes == b")" {
                body_at += 1;
                break;
            }
            if token.kind != Kind::Ident || params.contains(&token.bytes) {
                unsupported = true;
            }
            params.push(token.bytes.clone());
            body_at = significant(tokens, body_at + 1);
            if tokens.get(body_at)?.bytes == b"," {
                body_at += 1;
                if tokens
                    .get(significant(tokens, body_at))
                    .is_some_and(|t| t.bytes == b")")
                {
                    unsupported = true;
                }
            } else if tokens[body_at].bytes != b")" {
                unsupported = true;
                body_at += 1;
            }
        }
        parameters = Some(params);
    }
    let body = tokens[body_at..].to_vec();
    unsupported |= body
        .iter()
        .any(|t| matches!(t.bytes.as_slice(), b"#" | b"##" | b"%:" | b"%:%:"));
    Some((
        name.bytes.clone(),
        Macro {
            parameters,
            body,
            unsupported,
        },
    ))
}

struct Expander {
    macros: BTreeMap<Vec<u8>, Macro>,
    steps: usize,
    limit: usize,
}
impl Expander {
    fn expand(
        &mut self,
        input: &[Token],
        disabled: &BTreeSet<Vec<u8>>,
        depth: usize,
    ) -> Result<Vec<Token>, CPreprocessorError> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < input.len() {
            self.steps += 1;
            if self.steps > self.limit || depth > 32 {
                return Err(CPreprocessorError {
                    offset: input[at].origin,
                    reason: "local macro expansion exceeds its bounded validation budget",
                });
            }
            let token = &input[at];
            let definition = (token.kind == Kind::Ident && !disabled.contains(&token.bytes))
                .then(|| self.macros.get(&token.bytes).cloned())
                .flatten();
            let Some(definition) = definition else {
                out.push(token.clone());
                at += 1;
                continue;
            };
            let mut after = at + 1;
            let mut arguments = Vec::new();
            if let Some(parameters) = &definition.parameters {
                let open = significant(input, after);
                if !input.get(open).is_some_and(|t| t.bytes == b"(") {
                    out.push(token.clone());
                    at += 1;
                    continue;
                }
                let mut nesting = 1;
                let mut start = open + 1;
                after = start;
                loop {
                    let Some(next) = input.get(after) else {
                        return Err(CPreprocessorError {
                            offset: token.origin,
                            reason: "unterminated local function macro invocation",
                        });
                    };
                    if next.kind == Kind::Punct {
                        if next.bytes == b"(" {
                            nesting += 1;
                        }
                        if next.bytes == b")" {
                            nesting -= 1;
                        }
                        if (next.bytes == b"," && nesting == 1) || nesting == 0 {
                            arguments.push(input[start..after].to_vec());
                            start = after + 1;
                        }
                    }
                    after += 1;
                    if nesting == 0 {
                        break;
                    }
                }
                if parameters.is_empty()
                    && arguments.len() == 1
                    && significant(&arguments[0], 0) == arguments[0].len()
                {
                    arguments.clear();
                }
                if arguments.len() != parameters.len() {
                    return Err(CPreprocessorError {
                        offset: token.origin,
                        reason: "local function macro argument count does not match its definition",
                    });
                }
            }
            if definition.unsupported {
                return Err(CPreprocessorError { offset: token.origin, reason: "local macro requires compiler preprocessing (conditional binding, variadics, stringification or token pasting)" });
            }
            // Ordinary macro arguments are prescanned before substitution;
            // this also handles nested invocations of the same macro.
            for argument in &mut arguments {
                *argument = self.expand(argument, disabled, depth + 1)?;
            }
            let mut replacement = Vec::new();
            for body_token in &definition.body {
                // Preserve preprocessing-token boundaries after substitution.
                // `+x` with argument `+value` must never become `++value`.
                replacement.push(Token {
                    bytes: vec![b' '],
                    origin: token.origin,
                    kind: Kind::Trivia,
                });
                let parameter = definition.parameters.as_ref().and_then(|params| {
                    params
                        .iter()
                        .position(|p| *p == body_token.bytes && body_token.kind == Kind::Ident)
                });
                let added = parameter.map_or(1, |index| arguments[index].len());
                if replacement.len().saturating_add(added) > self.limit {
                    return Err(CPreprocessorError {
                        offset: token.origin,
                        reason: "local macro replacement exceeds its bounded validation budget",
                    });
                }
                if let Some(index) = parameter {
                    replacement.extend(arguments[index].clone());
                } else {
                    replacement.push(body_token.clone());
                }
            }
            for expanded in &mut replacement {
                expanded.origin = token.origin;
            }
            let mut disabled = disabled.clone();
            disabled.insert(token.bytes.clone());
            // Macro expansion cannot concatenate separate preprocessing tokens.
            out.push(Token {
                bytes: vec![b' '],
                origin: token.origin,
                kind: Kind::Trivia,
            });
            out.extend(self.expand(&replacement, &disabled, depth + 1)?);
            out.push(Token {
                bytes: vec![b' '],
                origin: token.origin,
                kind: Kind::Trivia,
            });
            at = after;
        }
        Ok(out)
    }
}

/// Expand supported local definitions in source order. Preprocessor directives
/// are kept opaque and unchanged for tree-sitter. Conditional evaluation is not
/// attempted; definitions/undefs under conditional groups are marked unsupported.
pub fn c_preprocessor_validation_view(
    source: &[u8],
) -> Result<CPreprocessorView, CPreprocessorError> {
    let tokens = tokenize(source);
    let mut expander = Expander {
        macros: BTreeMap::new(),
        steps: 0,
        limit: source
            .len()
            .saturating_mul(16)
            .saturating_add(65536)
            .min(16 * 1024 * 1024),
    };
    let mut out = Vec::new();
    let mut at = 0;
    let mut code_start = 0;
    let mut line_start = true;
    let mut conditional = 0usize;
    while at < tokens.len() {
        let token = &tokens[at];
        if line_start && token.bytes == b"#" {
            out.extend(expander.expand(&tokens[code_start..at], &BTreeSet::new(), 0)?);
            let mut end = at + 1;
            while end < tokens.len() {
                if tokens[end].bytes == b"\n" {
                    if end > at && tokens[end - 1].bytes == b"\\" {
                        end += 1;
                        continue;
                    }
                    break;
                }
                end += 1;
            }
            let directive_at = significant(&tokens, at + 1);
            let directive = tokens
                .get(directive_at)
                .map(|t| t.bytes.as_slice())
                .unwrap_or_default();
            let mut definition_tokens = Vec::new();
            let mut i = (directive_at + 1).min(end);
            while i < end {
                if tokens[i].bytes == b"\\" && tokens.get(i + 1).is_some_and(|t| t.bytes == b"\n") {
                    i += 2;
                } else {
                    definition_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            match directive {
                b"define" => {
                    if let Some((name, mut definition)) = define(&definition_tokens) {
                        definition.unsupported |= conditional > 0;
                        expander.macros.insert(name, definition);
                    }
                }
                b"undef" => {
                    let name = definition_tokens.get(significant(&definition_tokens, 0));
                    if let Some(name) = name {
                        if conditional == 0 {
                            expander.macros.remove(&name.bytes);
                        } else if let Some(definition) = expander.macros.get_mut(&name.bytes) {
                            definition.unsupported = true;
                        }
                    }
                }
                b"if" | b"ifdef" | b"ifndef" => conditional += 1,
                b"endif" => conditional = conditional.saturating_sub(1),
                _ => {}
            }
            let after = (end + 1).min(tokens.len());
            out.extend(tokens[at..after].iter().cloned());
            at = after;
            code_start = at;
            line_start = true;
            continue;
        }
        if token.kind == Kind::Trivia {
            if token.bytes.contains(&b'\n') {
                line_start = true;
            }
        } else {
            line_start = false;
        }
        at += 1;
    }
    out.extend(expander.expand(&tokens[code_start..], &BTreeSet::new(), 0)?);
    let mut bytes = Vec::new();
    let mut origins = Vec::new();
    for token in out {
        if bytes.len().saturating_add(token.bytes.len()) > expander.limit {
            return Err(CPreprocessorError {
                offset: token.origin,
                reason: "local macro expansion exceeds its bounded validation byte budget",
            });
        }
        for (index, byte) in token.bytes.iter().enumerate() {
            bytes.push(*byte);
            // Copied source tokens have exact coordinates; macro-generated
            // tokens all point at their invocation, including parameter text.
            let original = source.get(token.origin..token.origin + token.bytes.len());
            origins.push(if original == Some(token.bytes.as_slice()) {
                token.origin + index
            } else {
                token.origin
            });
        }
    }
    // Expansion tokens may coincidentally equal source text; force monotone
    // mapping so source-range lookup remains conservative and well-defined.
    for i in 1..origins.len() {
        origins[i] = origins[i].max(origins[i - 1]);
    }
    Ok(CPreprocessorView {
        bytes,
        origins,
        source_len: source.len(),
    })
}
