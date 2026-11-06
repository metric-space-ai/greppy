// One synchronous traversal per page; late receipts cannot complete a new request.
pub(crate) struct HistoryReceipt<T> {
    pending: Option<(T, bool)>,
}

impl<T: PartialEq> HistoryReceipt<T> {
    pub(crate) fn new() -> Self {
        Self { pending: None }
    }

    pub(crate) fn begin(&mut self, token: T) {
        self.pending = Some((token, false));
    }

    pub(crate) fn complete(&mut self, token: &T) -> bool {
        if let Some((expected, complete)) = &mut self.pending {
            if expected == token {
                *complete = true;
                return true;
            }
        }
        false
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.pending.as_ref().is_some_and(|(_, complete)| *complete)
    }

    pub(crate) fn clear(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::HistoryReceipt;
    use std::rc::Rc;

    #[test]
    fn history_receipt_requires_current_native_token_and_releases_it() {
        let first = Rc::new("first".to_owned());
        let second = Rc::new("second".to_owned());
        let mut receipt = HistoryReceipt::new();
        assert!(!receipt.is_complete());
        receipt.begin(Rc::clone(&first));
        assert_eq!(Rc::strong_count(&first), 2);
        assert!(!receipt.is_complete(), "pending history was treated as complete");
        assert!(!receipt.complete(&second), "foreign history receipt was accepted");
        assert!(!receipt.is_complete(), "foreign history receipt was accepted");
        assert!(receipt.complete(&first));
        assert!(receipt.is_complete());
        receipt.begin(Rc::clone(&second));
        assert_eq!(Rc::strong_count(&first), 1);
        assert!(!receipt.is_complete(), "new history reused old completion");
        assert!(!receipt.complete(&first));
        assert!(!receipt.is_complete(), "late history receipt was accepted");
        assert!(receipt.complete(&second));
        assert!(receipt.is_complete());
        receipt.clear();
        assert_eq!(Rc::strong_count(&second), 1);
        assert!(!receipt.is_complete());
        assert!(!receipt.complete(&second), "foreign history receipt was accepted");
    }
}
