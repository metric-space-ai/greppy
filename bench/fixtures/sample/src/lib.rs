// Realistic-ish Rust fixture for the Phase 7 benchmarks.
// Hand-crafted so the bench has predictable symbols to search.

use std::collections::HashMap;

/// Greeter holds a name and produces a greeting.
pub struct Greeter {
    pub name: String,
}

impl Greeter {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }

    pub fn greet(&self) -> String {
        format!("hello, {}", self.name)
    }
}

/// ProcessOrder models a simple order-handling flow.
pub struct ProcessOrder {
    pub id: u64,
    pub items: Vec<String>,
}

impl ProcessOrder {
    pub fn new(id: u64) -> Self {
        Self { id, items: vec![] }
    }

    pub fn add_item(&mut self, item: &str) {
        self.items.push(item.to_string());
    }

    pub fn total(&self) -> usize {
        self.items.len()
    }
}

/// UserService is the canonical name an agent might search for.
pub trait UserService {
    fn find_user(&self, id: u64) -> Option<String>;
    fn all_users(&self) -> Vec<String>;
}

/// InMemoryUserService is a simple implementation.
pub struct InMemoryUserService {
    users: HashMap<u64, String>,
}

impl InMemoryUserService {
    pub fn new() -> Self {
        Self { users: HashMap::new() }
    }
}

impl UserService for InMemoryUserService {
    fn find_user(&self, id: u64) -> Option<String> {
        self.users.get(&id).cloned()
    }

    fn all_users(&self) -> Vec<String> {
        self.users.values().cloned().collect()
    }
}

/// hello is a trivial free function the bench uses for visibility checks.
pub fn hello() -> String {
    "hi".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeter_greets() {
        let g = Greeter::new("world");
        assert_eq!(g.greet(), "hello, world");
    }
}
