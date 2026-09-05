//! Non-recycled reference ranges owned by the supervisor, not a content worker.
//!
//! Observations assign fresh IDs only to previously unseen DOM nodes. Reserving
//! a new range for each observation prevents a replacement document or restarted
//! content worker from reusing an ID. Actual node identity is checked separately
//! by the content worker; a number or copied DOM attribute is not a capability.

/// Keep observations bounded independently of the number of nodes on a page.
pub const OBSERVED_REF_LIMIT: u64 = 200;
/// IDs cross JSON into JavaScript and must remain exact integers there.
const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefRange {
    pub first: u64,
    pub last: u64,
}

impl RefRange {
    pub fn contains(self, reference: u64) -> bool {
        (self.first..=self.last).contains(&reference)
    }
}

#[derive(Debug)]
pub struct RefAllocator {
    next: u64,
}

impl Default for RefAllocator {
    fn default() -> Self {
        Self { next: 1 }
    }
}

impl RefAllocator {
    /// Reserve even when observation later fails: an uncertain delivery must
    /// never make an already exposed identifier available for reuse.
    pub fn reserve(&mut self) -> Result<RefRange, &'static str> {
        let last = self
            .next
            .checked_add(OBSERVED_REF_LIMIT - 1)
            .filter(|last| *last <= MAX_SAFE_INTEGER)
            .ok_or("observed reference identifier space exhausted")?;
        let range = RefRange {
            first: self.next,
            last,
        };
        self.next = last + 1;
        Ok(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_positive_disjoint_ranges_across_observations() {
        let mut allocator = RefAllocator::default();
        let first = allocator.reserve().unwrap();
        let second = allocator.reserve().unwrap();
        assert_eq!(
            first,
            RefRange {
                first: 1,
                last: 200
            }
        );
        assert_eq!(
            second,
            RefRange {
                first: 201,
                last: 400
            }
        );
        assert!(!first.contains(0));
        assert!(first.contains(1));
        assert!(first.contains(200));
        assert!(!first.contains(second.first));
    }

    #[test]
    fn a_failed_observation_does_not_recycle_its_reserved_range() {
        let mut allocator = RefAllocator::default();
        let abandoned = allocator.reserve().unwrap();
        // The caller can fail or lose its worker after delivering this range.
        // There is deliberately no release/recycle operation on the allocator.
        let retry = allocator.reserve().unwrap();
        assert_eq!(retry.first, abandoned.last + 1);
    }

    #[test]
    fn allows_the_last_exact_javascript_integer_then_fails_closed() {
        let mut allocator = RefAllocator {
            next: MAX_SAFE_INTEGER - OBSERVED_REF_LIMIT + 1,
        };
        let last = allocator.reserve().unwrap();
        assert_eq!(last.last, MAX_SAFE_INTEGER);
        let next = allocator.next;
        assert!(allocator.reserve().is_err());
        assert_eq!(allocator.next, next);
        assert!(allocator.reserve().is_err());
    }

    #[test]
    fn rejects_partial_ranges_and_integer_overflow_without_mutation() {
        for next in [MAX_SAFE_INTEGER, u64::MAX] {
            let mut allocator = RefAllocator { next };
            assert!(allocator.reserve().is_err());
            assert_eq!(allocator.next, next);
        }
    }
}
