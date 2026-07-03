// Throwaway: parse a fixture and print the s-expression to learn node names.
fn main() {
    let src = br#"
defmodule Calc do
  import Math
  def add(a, b) do
    total(a, b)
  end
  defp total(a, b), do: a + b
end
"#;
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_elixir::LANGUAGE.into())
        .unwrap();
    let tree = p.parse(&src[..], None).unwrap();
    print_node(tree.root_node(), src, 0);
}
fn print_node(n: tree_sitter::Node, src: &[u8], depth: usize) {
    let txt = if n.child_count() == 0 {
        format!(
            " '{}'",
            std::str::from_utf8(&src[n.byte_range()])
                .unwrap_or("")
                .replace('\n', " ")
        )
    } else {
        String::new()
    };
    println!("{}{}{}", "  ".repeat(depth), n.kind(), txt);
    for i in 0..n.child_count() {
        if depth < 6 {
            print_node(n.child(i).unwrap(), src, depth + 1);
        }
    }
}
