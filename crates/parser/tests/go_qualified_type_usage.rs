//! Regression from Anko navigation: package-qualified types are references.
use greppy_parser::{extract, parse, Language};

#[test]
fn qualified_go_types_are_usages_not_definition_names() {
    let source = br#"package vm
import "example/ast"
type Local struct{}
func runSingleStmt(stmt interface{}) {
 switch stmt.(type) {
 case *ast.VarStmt:
  _ = ast.VarStmt{}
 }
}
"#;
    let tree = parse(Language::Go, source).unwrap();
    assert!(!tree.root_node().has_error());
    let mut pending = vec![tree.root_node()];
    let mut qualified_names = 0;
    while let Some(node) = pending.pop() {
        if node.kind() == "qualified_type" {
            let name = node
                .child_by_field_name("name")
                .expect("Go qualified_type has name field");
            assert_eq!(&source[name.byte_range()], b"VarStmt");
            qualified_names += 1;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    assert_eq!(qualified_names, 2, "prove type assertion and literal ASTs");
    let result = extract(Language::Go, source, "vm/vmStmt.go").unwrap();
    let usages: Vec<_> = result
        .edges
        .iter()
        .filter(|edge| edge.edge_type == "USAGE" && edge.properties["ref_name"] == "VarStmt")
        .collect();
    assert_eq!(
        usages.len(),
        2,
        "qualified types must survive the definition-name filter: {usages:?}"
    );
    for usage in usages {
        assert_eq!(
            usage.source_qualified_name,
            "vm/vmStmt.go::Function::runSingleStmt"
        );
    }
    assert!(
        !result
            .edges
            .iter()
            .any(|edge| edge.edge_type == "USAGE" && edge.properties["ref_name"] == "Local"),
        "a real type declaration must remain excluded"
    );
}
