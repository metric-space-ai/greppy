//! Bounded retained browser records. Limits apply before retaining a new row.
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::{io, ops::Deref};

#[derive(Clone, Debug)]
pub struct RetainedRecords {
    rows: Vec<Value>,
    bytes: usize,
    max_bytes: usize,
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
        Self {
            rows: Vec::new(),
            bytes: 0,
            max_rows,
            max_bytes,
            dropped: 0,
            overflowed: false,
        }
    }
    pub fn push(&mut self, row: Value) {
        let size = row_bytes(&row);
        if size > self.max_bytes || self.max_rows == 0 {
            self.overflowed = true;
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        self.overflowed |= self.bytes.saturating_add(size) > self.max_bytes;
        let mut evicted = 0;
        while self.rows.len().saturating_sub(evicted) >= self.max_rows
            || self.bytes.saturating_add(size) > self.max_bytes
        {
            let old_size = row_bytes(&self.rows[evicted]);
            self.bytes = self.bytes.saturating_sub(old_size);
            evicted += 1;
        }
        if evicted != 0 {
            // Count eviction preserves the existing rolling network-history
            // contract; byte eviction must also fail resource accounting.
            self.rows.drain(..evicted);
            self.dropped = self.dropped.saturating_add(evicted as u64);
        }
        self.bytes = self.bytes.saturating_add(size);
        self.rows.push(row);
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
        self.bytes = self.bytes.saturating_sub(row_bytes(&row));
        update(&mut row);
        // Replace without moving a completed request to the end of its history.
        let size = row_bytes(&row);
        if self.bytes.saturating_add(size) > self.max_bytes {
            self.overflowed = true;
            self.dropped = self.dropped.saturating_add(1);
        } else {
            self.bytes = self.bytes.saturating_add(size);
            self.rows.insert(index, row);
        }
        true
    }
    pub fn clear(&mut self) {
        self.rows.clear();
        self.bytes = 0;
        self.dropped = 0;
        self.overflowed = false;
    }
    pub fn retain(&mut self, mut predicate: impl FnMut(&Value) -> bool) {
        self.rows.retain(|row| predicate(row));
        self.bytes = self.rows.iter().map(row_bytes).sum();
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }
    pub fn snapshot(&self) -> io::Result<&[Value]> {
        if self.overflowed {
            Err(io::Error::other("resource_limit: retained page records exceeded their memory budget; history is incomplete"))
        } else {
            Ok(&self.rows)
        }
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
        let mut rows = RetainedRecords::new(4, 160);
        for i in 0..200 {
            rows.push(json!({"requestId": i.to_string(), "text": "é".repeat(20)}));
            assert!(rows.bytes() <= 160);
            assert_eq!(rows.bytes(), rows.iter().map(row_bytes).sum::<usize>());
        }
        assert!(rows.overflowed());
        assert!(rows
            .snapshot()
            .unwrap_err()
            .to_string()
            .contains("resource_limit"));
        assert!(rows.dropped() > 0);
        assert!(rows.update_last("199", |row| row["text"] = json!("x".repeat(300))));
        assert!(rows.iter().all(|row| row["requestId"] != "199"));
        assert!(rows.bytes() <= 160);
        rows.push(json!({"text": "x".repeat(1000)}));
        assert!(rows.bytes() <= 160);
        rows.clear();
        assert_eq!(rows.bytes(), 0);
        assert_eq!(rows.dropped(), 0);
        assert!(!rows.overflowed());
        assert!(rows.is_empty());
        assert!(rows.snapshot().unwrap().is_empty());
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
    }
}
