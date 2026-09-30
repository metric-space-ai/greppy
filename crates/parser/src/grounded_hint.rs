//! Exact source facts for small conditional register/flag writers.
//! Never expands acronyms or assigns architectural meanings to field names.
use tree_sitter::Node;

#[derive(Default)]
struct Facts {
    targets: Vec<String>,
    assignments: usize,
    conditions: usize,
    bitwise: bool,
}

/// Unknown syntax, calls, macros, locals and incomplete spans retain the model
/// path. This projects source facts without claiming verified runtime effects.
pub fn rust_conditional_bit_writes(source: &str) -> Option<String> {
    if source.len() > 2048 { return None; }
    let bytes = source.as_bytes();
    let tree = crate::parse(crate::Language::Rust, bytes).ok()?;
    let root = tree.root_node();
    if root.has_error() { return None; }
    let mut cursor = root.walk();
    let items: Vec<_> = root.named_children(&mut cursor)
        .filter(|node| !matches!(node.kind(), "line_comment" | "block_comment"))
        .collect();
    if items.len() != 1 || items[0].kind() != "function_item" { return None; }
    let function = items[0];
    let parameters = function.child_by_field_name("parameters")?;
    let mut cursor = parameters.walk();
    let names: Vec<_> = parameters.named_children(&mut cursor).map(|parameter| {
        let pattern = parameter.child_by_field_name("pattern")?;
        (pattern.kind() == "identifier").then(|| pattern.utf8_text(bytes).ok()).flatten()
    }).collect::<Option<Vec<_>>>()?;
    let mut facts = Facts::default();
    visit(function.child_by_field_name("body")?, bytes, &names, &mut facts)?;
    if facts.assignments < 2 || facts.assignments > 8 || facts.conditions == 0
        || !facts.bitwise || facts.targets.is_empty() || facts.targets.len() > 3 {
        return None;
    }
    let targets = match facts.targets.as_slice() {
        [one] => one.clone(),
        [one, two] => format!("{one} and {two}"),
        [one, two, three] => format!("{one}, {two} and {three}"),
        _ => return None,
    };
    let description = format!("Updates {targets} with bitwise operations and conditional writes");
    (description.chars().count() <= 140).then_some(description)
}

fn visit(node: Node<'_>, source: &[u8], parameters: &[&str], facts: &mut Facts) -> Option<()> {
    match node.kind() {
        "line_comment" | "block_comment" => return Some(()),
        "block" | "expression_statement" | "binary_expression" | "field_expression"
        | "field_identifier" | "integer_literal" | "boolean_literal"
        | "parenthesized_expression" => {}
        "identifier" => {
            if !parameters.contains(&node.utf8_text(source).ok()?) { return None; }
        }
        "if_expression" => facts.conditions += 1,
        "assignment_expression" | "compound_assignment_expr" => {
            let left = node.child_by_field_name("left")?;
            if left.kind() != "field_expression" { return None; }
            let base = left.child_by_field_name("value")?;
            if base.kind() != "identifier" || !parameters.contains(&base.utf8_text(source).ok()?) {
                return None;
            }
            let target = left.utf8_text(source).ok()?.to_string();
            if !facts.targets.contains(&target) { facts.targets.push(target); }
            facts.assignments += 1;
        }
        _ => return None,
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.is_named() { visit(child, source, parameters, facts)?; }
        else if matches!(child.kind(), "&" | "|" | "^" | "&=" | "|=" | "^=" | "<<" | ">>") {
            facts.bitwise = true;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const WORD_FLAGS: &str = r#"pub fn and_word(s: &mut Core, value: u16) {
    s.m_isr = s.m_sr & 0x10;
    if value == 0 { s.m_isr |= 4; }
    if value & 0x8000 != 0 { s.m_isr |= 8; }
    s.m_aluo = value;
}"#;

    #[test]
    fn reports_written_fields_without_inventing_interrupt_semantics() {
        assert_eq!(rust_conditional_bit_writes(WORD_FLAGS).as_deref(), Some(
            "Updates s.m_isr and s.m_aluo with bitwise operations and conditional writes"));
    }

    #[test]
    fn architectural_names_do_not_change_source_fact_contract() {
        let source = WORD_FLAGS.replace("m_isr", "irq_mask").replace("and_word", "set_irq_mask");
        assert_eq!(rust_conditional_bit_writes(&source).as_deref(), Some(
            "Updates s.irq_mask and s.m_aluo with bitwise operations and conditional writes"));
    }

    #[test]
    fn complex_or_incomplete_code_retains_normal_model_path() {
        for source in [
            WORD_FLAGS.replace("s.m_aluo = value;", "s.set_irq_line(value);"),
            WORD_FLAGS.replace("s.m_aluo = value;", "notify!(); s.m_aluo = value;"),
            WORD_FLAGS.replace("s.m_aluo = value;", "let x = value; s.m_aluo = x;"),
            WORD_FLAGS.replace("s.m_aluo = value;", "s.m_aluo = TABLE[value];"),
            WORD_FLAGS.replace("s.m_aluo = value;", "s.m_aluo = make_value();"),
            WORD_FLAGS[..WORD_FLAGS.len() - 1].to_owned(),
            format!("{WORD_FLAGS}\nfn unrelated() {{}}"),
        ] {
            assert_eq!(rust_conditional_bit_writes(&source), None, "{source}");
        }
    }
}
