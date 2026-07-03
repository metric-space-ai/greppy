// Secondary module for the bench fixture so semantic search has
// multiple files to choose from.

use crate::Greeter;

pub fn make_greeting(name: &str) -> String {
    Greeter::new(name).greet()
}

pub fn polite_form(name: &str) -> String {
    format!("hello, {name}; nice to meet you")
}
// comment
// comment
// comment
// comment
// comment
// comment
// comment
// comment
