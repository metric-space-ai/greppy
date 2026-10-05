use greppy_parser::{extract, Language};

#[test]
fn js_ts_non_identifier_methods_have_persistable_reference_owners() {
    let source = br#"
function helper() {}
class Stream {
    async *[Symbol.asyncIterator]() { await new Promise(resolve => resolve()); helper(); }
    #private() { helper(); }
    "quoted"() { helper(); }
    42() { helper(); }
    ordinary() { helper(); }
}
class Other { [Symbol.asyncIterator]() { helper(); } }
const object = { [Symbol.iterator]() { helper(); } };
"#;
    for language in [Language::JavaScript, Language::TypeScript { tsx: false }, Language::TypeScript { tsx: true }] {
        let result = extract(language, source, "stream.ts").unwrap();
        for name in ["[Symbol.asyncIterator]", "#private", "\"quoted\"", "42", "ordinary"] {
            let qname = format!("stream.ts::Stream::{name}");
            assert!(result.nodes.iter().any(|node| node.qualified_name == qname && node.label == "Method"), "{language:?}: missing {qname}");
            assert!(result.edges.iter().any(|edge| edge.edge_type == "CALLS" && edge.source_qualified_name == qname && edge.properties["callee_name"] == "helper"), "{language:?}: missing helper call from {qname}");
        }
        for edge in result.edges.iter().filter(|edge| matches!(edge.edge_type.as_str(), "CALLS" | "USAGE")) {
            assert!(edge.source_qualified_name == "stream.ts::__file__" || result.nodes.iter().any(|node| node.qualified_name == edge.source_qualified_name), "{language:?}: orphan reference {edge:?}");
        }
        assert!(result.nodes.iter().any(|node| node.qualified_name == "stream.ts::Other::[Symbol.asyncIterator]"));
        assert!(result.nodes.iter().any(|node| node.qualified_name == "stream.ts::Function::[Symbol.iterator]"));
    }
}

#[test]
fn js_ts_computed_method_keys_execute_in_the_enclosing_scope() {
    let source = br#"
function makeKey() { return "key"; }
function body() {}
class Top { [makeKey()]() { body(); } }
function factory() { return { [makeKey()]() { body(); } }; }
"#;
    for language in [Language::JavaScript, Language::TypeScript { tsx: false }, Language::TypeScript { tsx: true }] {
        let result = extract(language, source, "keys.ts").unwrap();
        let keys: Vec<_> = result.edges.iter().filter(|edge| edge.edge_type == "CALLS" && edge.properties["callee_name"] == "makeKey").map(|edge| edge.source_qualified_name.as_str()).collect();
        assert_eq!(keys, ["keys.ts::__file__", "keys.ts::Function::factory"], "{language:?}");
        let bodies: Vec<_> = result.edges.iter().filter(|edge| edge.edge_type == "CALLS" && edge.properties["callee_name"] == "body").map(|edge| edge.source_qualified_name.as_str()).collect();
        assert_eq!(bodies, ["keys.ts::Top::[makeKey()]", "keys.ts::Function::[makeKey()]"], "{language:?}");
    }
}
