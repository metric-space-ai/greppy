//! Bounded local C macro expansion for edit validation, never extraction.
//!
//! This does not implement an include search path or conditional evaluation.
//! Supported definitions are local object/function macros outside uncertain
//! conditionals, including a conventional whole-file include guard, without
//! stringification, token pasting, or variadics. Unsupported used definitions
//! fail closed with an invocation coordinate rather than disabling syntax guards.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const ALLOCATION_BUDGET: usize = 64 * 1024 * 1024;

struct AllocationBudget {
    remaining: usize,
}
impl AllocationBudget {
    fn charge(&mut self, amount: usize, offset: usize) -> Result<(), CPreprocessorError> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(CPreprocessorError {
                offset,
                reason: "local C macro validation exceeds its bounded allocation budget",
            })?;
        Ok(())
    }
    fn charge_tokens(&mut self, tokens: &[Token], offset: usize) -> Result<(), CPreprocessorError> {
        let amount = tokens.iter().fold(0usize, |sum, t| {
            sum.saturating_add(2 * std::mem::size_of::<Token>() + t.bytes.len() + 32)
        });
        self.charge(amount, offset)
    }
    fn clone_tokens(
        &mut self,
        tokens: &[Token],
        offset: usize,
    ) -> Result<Vec<Token>, CPreprocessorError> {
        self.charge_tokens(tokens, offset)?;
        Ok(tokens.to_vec())
    }
}

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
fn tokenize(
    source: &[u8],
    budget: &mut AllocationBudget,
) -> Result<Vec<Token>, CPreprocessorError> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < source.len() {
        let start = at;
        let kind;
        if source[at].is_ascii_whitespace() {
            kind = Kind::Trivia;
            at += 1; // Keep CR/LF separate, aggregate horizontal trivia.
            if !matches!(source[start], b'\n' | b'\r') {
                while source
                    .get(at)
                    .is_some_and(|b| b.is_ascii_whitespace() && !matches!(*b, b'\n' | b'\r'))
                {
                    at += 1;
                }
            }
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
        budget.charge(2 * std::mem::size_of::<Token>() + at - start + 32, start)?;
        out.push(Token {
            bytes: source[start..at].to_vec(),
            origin: start,
            kind,
        });
    }
    Ok(out)
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
    macros: BTreeMap<Vec<u8>, Arc<Macro>>,
    steps: usize,
    limit: usize,
    budget: AllocationBudget,
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
                out.extend(
                    self.budget
                        .clone_tokens(std::slice::from_ref(token), token.origin)?,
                );
                at += 1;
                continue;
            };
            let mut after = at + 1;
            let mut arguments = Vec::new();
            if let Some(parameters) = &definition.parameters {
                let open = significant(input, after);
                if !input.get(open).is_some_and(|t| t.bytes == b"(") {
                    out.extend(
                        self.budget
                            .clone_tokens(std::slice::from_ref(token), token.origin)?,
                    );
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
                            self.budget
                                .charge(2 * std::mem::size_of::<Vec<Token>>(), token.origin)?;
                            arguments.push(
                                self.budget
                                    .clone_tokens(&input[start..after], token.origin)?,
                            );
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
                self.budget
                    .charge(2 * std::mem::size_of::<Token>() + 33, token.origin)?;
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
                    replacement.extend(self.budget.clone_tokens(&arguments[index], token.origin)?);
                } else {
                    replacement.extend(
                        self.budget
                            .clone_tokens(std::slice::from_ref(body_token), token.origin)?,
                    );
                }
            }
            for expanded in &mut replacement {
                expanded.origin = token.origin;
            }
            self.budget.charge(
                disabled.iter().fold(token.bytes.len() + 64, |sum, name| {
                    sum.saturating_add(name.len() + 64)
                }),
                token.origin,
            )?;
            let mut disabled = disabled.clone();
            disabled.insert(token.bytes.clone());
            // Macro expansion cannot concatenate separate preprocessing tokens.
            self.budget
                .charge(4 * std::mem::size_of::<Token>() + 66, token.origin)?;
            out.push(Token {
                bytes: vec![b' '],
                origin: token.origin,
                kind: Kind::Trivia,
            });
            let expanded = self.expand(&replacement, &disabled, depth + 1)?;
            // The bounded recursive scanner does not rescan across the macro /
            // remaining-input boundary. Never leave a callable local alias as
            // an ordinary function call: its real expansion might be invalid.
            if input
                .get(significant(input, after))
                .is_some_and(|t| t.bytes == b"(")
                && expanded
                    .iter()
                    .rev()
                    .find(|t| t.kind != Kind::Trivia)
                    .is_some_and(|last| {
                        last.kind == Kind::Ident
                            && !disabled.contains(&last.bytes)
                            && self
                                .macros
                                .get(&last.bytes)
                                .is_some_and(|m| m.parameters.is_some())
                    })
            {
                return Err(CPreprocessorError { offset: token.origin, reason: "local macro function alias across an expansion boundary requires compiler preprocessing" });
            }
            out.extend(expanded);
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

// Recognize only a fresh-include envelope: leading #ifndef NAME, immediate
// empty #define NAME, one matching final #endif and no outer alternative.
// Its body must still parse; nested build conditionals remain unsupported.
fn ordinary_header_guard(tokens: &[Token]) -> bool {
    let first = significant(tokens, 0);
    if !tokens.get(first).is_some_and(|t| t.bytes == b"#") {
        return false;
    }
    let command = significant(tokens, first + 1);
    if !tokens.get(command).is_some_and(|t| t.bytes == b"ifndef") {
        return false;
    }
    let name = significant(tokens, command + 1);
    if !tokens.get(name).is_some_and(|t| t.kind == Kind::Ident) {
        return false;
    }
    let first_end = tokens[name..]
        .iter()
        .position(|t| t.bytes == b"\n")
        .map_or(tokens.len(), |n| name + n);
    if tokens[name + 1..first_end]
        .iter()
        .any(|t| t.kind != Kind::Trivia)
    {
        return false;
    }
    let define_at = significant(tokens, first_end + 1);
    if !tokens.get(define_at).is_some_and(|t| t.bytes == b"#") {
        return false;
    }
    let define_command = significant(tokens, define_at + 1);
    let define_name = significant(tokens, define_command + 1);
    if !tokens
        .get(define_command)
        .is_some_and(|t| t.bytes == b"define")
        || !tokens
            .get(define_name)
            .is_some_and(|t| t.bytes == tokens[name].bytes)
    {
        return false;
    }
    let define_end = tokens[define_name..]
        .iter()
        .position(|t| t.bytes == b"\n")
        .map_or(tokens.len(), |n| define_name + n);
    if tokens[define_name + 1..define_end]
        .iter()
        .any(|t| t.kind != Kind::Trivia)
    {
        return false;
    }
    let mut depth = 0usize;
    for at in first..tokens.len() {
        if tokens[at].bytes != b"#" {
            continue;
        }
        let command = significant(tokens, at + 1);
        let Some(command) = tokens.get(command) else {
            return false;
        };
        match command.bytes.as_slice() {
            b"if" | b"ifdef" | b"ifndef" => depth += 1,
            b"else" | b"elif" if depth == 1 => return false,
            b"endif" => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return significant(tokens, significant(tokens, at + 1) + 1) == tokens.len();
                }
            }
            _ => {}
        }
    }
    false
}

fn splice_length(tokens: &[Token], at: usize) -> usize {
    if !tokens.get(at).is_some_and(|t| t.bytes == b"\\") {
        return 0;
    }
    if tokens.get(at + 1).is_some_and(|t| t.bytes == b"\n") {
        return 2;
    }
    if tokens.get(at + 1).is_some_and(|t| t.bytes == b"\r")
        && tokens.get(at + 2).is_some_and(|t| t.bytes == b"\n")
    {
        return 3;
    }
    0
}

/// Expand supported local definitions in source order, admitting input before
/// tokenization and charging cumulative token/copy/map allocation before copies.
/// Allocation accounting includes token/vector allowance and the usize origins;
/// it bounds this helper's work, not the caller's input or whole-process memory.
pub fn c_preprocessor_validation_view(
    source: &[u8],
) -> Result<CPreprocessorView, CPreprocessorError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(CPreprocessorError {
            offset: 0,
            reason: "C macro validation source exceeds its bounded input budget",
        });
    }
    let mut budget = AllocationBudget {
        remaining: ALLOCATION_BUDGET,
    };
    let tokens = tokenize(source, &mut budget)?;
    let header_guard_depth = if ordinary_header_guard(&tokens) { 1 } else { 0 };
    let mut expander = Expander {
        macros: BTreeMap::new(),
        steps: 0,
        budget,
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
                    if (end > at && splice_length(&tokens, end - 1) == 2)
                        || (end > at + 1 && splice_length(&tokens, end - 2) == 3)
                    {
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
            // Charge all definition/parameter/key copies before allocating.
            // The immutable stored body is then shared via Arc during uses.
            for _ in 0..3 {
                expander
                    .budget
                    .charge_tokens(&tokens[i..end], token.origin)?;
            }
            while i < end {
                let splice = splice_length(&tokens, i);
                if splice != 0 {
                    i += splice;
                } else {
                    definition_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            match directive {
                b"define" => {
                    if let Some((name, mut definition)) = define(&definition_tokens) {
                        definition.unsupported |= conditional > header_guard_depth;
                        expander
                            .budget
                            .charge(std::mem::size_of::<Macro>() + 128, token.origin)?;
                        expander.macros.insert(name, Arc::new(definition));
                    }
                }
                b"undef" => {
                    let name = definition_tokens.get(significant(&definition_tokens, 0));
                    if let Some(name) = name {
                        if conditional <= header_guard_depth {
                            expander.macros.remove(&name.bytes);
                        } else if let Some(definition) = expander.macros.get_mut(&name.bytes) {
                            Arc::make_mut(definition).unsupported = true;
                        }
                    }
                }
                b"if" | b"ifdef" | b"ifndef" => conditional += 1,
                b"endif" => conditional = conditional.saturating_sub(1),
                _ => {}
            }
            let after = (end + 1).min(tokens.len());
            out.extend(
                expander
                    .budget
                    .clone_tokens(&tokens[at..after], token.origin)?,
            );
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
    let output_len = out
        .iter()
        .fold(0usize, |sum, token| sum.saturating_add(token.bytes.len()));
    if output_len > expander.limit {
        return Err(CPreprocessorError {
            offset: 0,
            reason: "local macro expansion exceeds its bounded validation byte budget",
        });
    }
    expander.budget.charge(
        output_len.saturating_mul(1 + std::mem::size_of::<usize>()),
        0,
    )?;
    let mut bytes = Vec::with_capacity(output_len);
    let mut origins = Vec::with_capacity(output_len);
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

#[cfg(test)]
mod allocation_tests {
    use super::*;
    #[test]
    fn trivia_materialization_and_copies_are_charged_before_allocation() {
        let mut budget = AllocationBudget { remaining: 128 };
        assert!(tokenize(&vec![b' '; 4096], &mut budget).is_err());
        let token = Token {
            bytes: vec![b'x'; 4096],
            origin: 0,
            kind: Kind::Ident,
        };
        let mut budget = AllocationBudget { remaining: 128 };
        assert!(budget.clone_tokens(&[token], 0).is_err());
        let mut budget = AllocationBudget { remaining: 8192 };
        let tokens = tokenize(&vec![b' '; 4096], &mut budget).unwrap();
        assert_eq!(
            tokens.len(),
            1,
            "horizontal trivia must not allocate one token per byte"
        );
    }
}
