//! Retained records share the existing context memory ceiling, never a second
//! per-buffer allowance. This accounts encoded records, not total process RSS.
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::{cell::Cell, io, ops::Deref, rc::Rc};

#[derive(Debug)]
pub struct RecordBudget {
    limit: Cell<usize>,
    used: Cell<usize>,
}
impl RecordBudget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit: Cell::new(limit),
            used: Cell::new(0),
        }
    }
    pub fn set_limit(&self, limit: usize) {
        self.limit.set(limit);
    }
    fn reserve(&self, bytes: usize) -> bool {
        let Some(next) = self.used.get().checked_add(bytes) else {
            return false;
        };
        if next > self.limit.get() {
            return false;
        }
        self.used.set(next);
        true
    }
    fn release(&self, bytes: usize) {
        self.used.set(self.used.get().saturating_sub(bytes));
    }
    fn exceeded(&self) -> bool {
        self.used.get() > self.limit.get()
    }
}

#[derive(Debug)]
pub struct RetainedRecords {
    rows: Vec<Value>,
    sizes: Vec<usize>,
    bytes: usize,
    budget: Rc<RecordBudget>,
    max_rows: usize,
    dropped: u64,
    overflowed: bool,
}
struct ByteCount(usize);
impl io::Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn row_bytes(row: &Value) -> usize {
    let mut count = ByteCount(0);
    serde_json::to_writer(&mut count, row).expect("counting JSON cannot fail");
    count.0
}
impl RetainedRecords {
    pub fn new(max_rows: usize, max_bytes: usize) -> Self {
        Self::with_budget(max_rows, Rc::new(RecordBudget::new(max_bytes)))
    }
    pub fn with_budget(max_rows: usize, budget: Rc<RecordBudget>) -> Self {
        Self {
            rows: Vec::new(),
            sizes: Vec::new(),
            bytes: 0,
            budget,
            max_rows,
            dropped: 0,
            overflowed: false,
        }
    }
    fn admit(&mut self, size: usize) -> bool {
        if self.budget.reserve(size) {
            true
        } else {
            self.overflowed = true;
            self.dropped = self.dropped.saturating_add(1);
            false
        }
    }
    pub fn push(&mut self, row: Value) {
        if self.max_rows == 0 {
            self.overflowed = true;
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        // Only network histories rotate by count, preserving their existing
        // retention contract. Console/download histories have no new row cap.
        if self.rows.len() == self.max_rows {
            let old_size = self.sizes.remove(0);
            self.rows.remove(0);
            self.bytes = self.bytes.saturating_sub(old_size);
            self.budget.release(old_size);
            self.dropped = self.dropped.saturating_add(1);
        }
        let size = row_bytes(&row);
        if self.admit(size) {
            self.bytes += size;
            self.sizes.push(size);
            self.rows.push(row);
        }
    }
    pub fn update_last(&mut self, request_id: &str, update: impl FnOnce(&mut Value)) -> bool {
        let Some(index) = self
            .rows
            .iter()
            .rposition(|row| row.get("requestId").and_then(Value::as_str) == Some(request_id))
        else {
            return false;
        };
        let mut row = self.rows.remove(index);
        let previous = self.sizes.remove(index);
        self.bytes = self.bytes.saturating_sub(previous);
        self.budget.release(previous);
        update(&mut row);
        let size = row_bytes(&row);
        if self.admit(size) {
            self.bytes += size;
            self.sizes.insert(index, size);
            self.rows.insert(index, row);
        }
        true
    }
    pub fn clear(&mut self) {
        self.budget.release(self.bytes);
        self.rows.clear();
        self.sizes.clear();
        self.bytes = 0;
        self.dropped = 0;
        self.overflowed = false;
    }
    pub fn retain(&mut self, mut predicate: impl FnMut(&Value) -> bool) {
        let mut sizes = self.sizes.iter().copied();
        let mut kept_sizes = Vec::new();
        let mut released = 0;
        self.rows.retain(|row| {
            let size = sizes.next().expect("one size per retained row");
            if predicate(row) {
                kept_sizes.push(size);
                true
            } else {
                released += size;
                false
            }
        });
        self.sizes = kept_sizes;
        self.bytes = self.bytes.saturating_sub(released);
        self.budget.release(released);
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn overflowed(&self) -> bool {
        self.overflowed || self.budget.exceeded()
    }
    pub fn snapshot(&self) -> io::Result<&[Value]> {
        if self.overflowed() {
            Err(io::Error::other("resource_limit: retained page records exceeded the context memory budget; history is incomplete"))
        } else {
            Ok(&self.rows)
        }
    }
}
impl Drop for RetainedRecords {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}
impl Deref for RetainedRecords {
    type Target = Vec<Value>;
    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}
impl Serialize for RetainedRecords {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.rows.serialize(serializer)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn retained_records_bound_bytes_before_retaining_and_account_updates() {
        let budget = Rc::new(RecordBudget::new(160));
        let mut rows = RetainedRecords::with_budget(4, Rc::clone(&budget));
        let mut other = RetainedRecords::with_budget(4, Rc::clone(&budget));
        rows.push(json!({"requestId": "first", "text": "é".repeat(20)}));
        other.push(json!({"requestId": "second", "text": "é".repeat(20)}));
        assert!(budget.used.get() <= 160);
        assert_eq!(budget.used.get(), rows.bytes() + other.bytes());
        for i in 0..200 {
            rows.push(json!({"requestId": i.to_string(), "text": "é".repeat(20)}));
            assert!(rows.bytes() + other.bytes() <= 160);
        }
        assert!(rows.overflowed());
        assert!(rows
            .snapshot()
            .unwrap_err()
            .to_string()
            .contains("resource_limit"));
        assert!(other.update_last("second", |row| row["text"] = json!("x".repeat(300))));
        assert!(other.is_empty());
        rows.clear();
        other.clear();
        assert_eq!(budget.used.get(), 0);
        assert!(!rows.overflowed());
        assert!(rows.snapshot().unwrap().is_empty());
        rows.push(json!({"text": "retained until drop"}));
        assert!(budget.used.get() > 0);
        drop(rows);
        assert_eq!(budget.used.get(), 0);
    }
    #[test]
    fn retained_records_count_rotation_remains_explicit_without_false_byte_failure() {
        let mut rows = RetainedRecords::new(2, 1024);
        for i in 0..20 {
            rows.push(json!({"requestId": i.to_string()}));
        }
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.dropped(), 18);
        assert!(!rows.overflowed());
        assert_eq!(serde_json::to_value(&rows).unwrap()[0]["requestId"], "18");
        let mut console = RetainedRecords::new(usize::MAX, 2 * 1024 * 1024);
        for _ in 0..10_005 {
            console.push(json!({"text": ""}));
        }
        assert_eq!(console.len(), 10_005);
        assert_eq!(console.dropped(), 0);
    }
}
