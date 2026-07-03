//! Std-only MinHash signatures for token sets.
//!
//! Ported (in spirit) from the upstream `src/simhash/minhash.c`, which
//! builds a `K`-permutation MinHash signature for near-clone detection.
//! The upstream computes its signature from normalised *AST* node-type
//! trigrams; the search crate does not have the AST at query time (it
//! works against stored graph nodes), so this port operates over the
//! caller-supplied **token set** instead — the node's name and
//! qualified-name tokens. The MinHash machinery itself (K independent
//! hash permutations, slot = min hash, Jaccard = fraction of matching
//! slots) is identical, and is what the semantic ranker uses as a
//! structural-overlap signal.
//!
//! No external dependencies: the per-permutation hash is a std-only
//! FNV-1a variant mixed with a per-slot seed, fully deterministic.

/// Number of hash permutations. Matches the upstream `CBM_MINHASH_K`
/// (64): larger `K` gives a tighter Jaccard estimate at the cost of a
/// larger signature.
pub const MINHASH_K: usize = 64;

/// A MinHash signature: `MINHASH_K` minimum hash values, one per
/// permutation. Two signatures built from overlapping token sets agree
/// on a fraction of slots proportional to the Jaccard similarity of the
/// underlying sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHash {
    values: [u32; MINHASH_K],
}

/// FNV-1a 64-bit constants.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Deterministic std-only hash of `bytes` under permutation `seed`.
/// FNV-1a seeded with a per-permutation salt, then avalanche-mixed so
/// adjacent seeds produce well-separated permutations.
fn seeded_hash(bytes: &[u8], seed: u64) -> u32 {
    let mut h = FNV_OFFSET ^ seed.wrapping_mul(FNV_PRIME);
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    // splitmix64-style finalizer for good bit dispersion.
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h & 0xFFFF_FFFF) as u32
}

impl MinHash {
    /// An empty signature (all slots saturated). `jaccard` against any
    /// real signature is `0.0`; `hamming` is the full width.
    pub fn empty() -> Self {
        Self {
            values: [u32::MAX; MINHASH_K],
        }
    }

    /// Build a signature from an iterator of feature tokens. Each token
    /// is hashed under all `K` permutations; the per-slot minimum is
    /// retained. Order-independent and deterministic.
    pub fn from_tokens<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut values = [u32::MAX; MINHASH_K];
        let mut any = false;
        for tok in tokens {
            let t = tok.as_ref();
            if t.is_empty() {
                continue;
            }
            any = true;
            let bytes = t.as_bytes();
            for (k, slot) in values.iter_mut().enumerate() {
                let h = seeded_hash(bytes, k as u64);
                if h < *slot {
                    *slot = h;
                }
            }
        }
        if !any {
            return Self::empty();
        }
        Self { values }
    }

    /// Estimated Jaccard similarity in `[0.0, 1.0]`: the fraction of
    /// permutation slots on which the two signatures agree. Mirrors
    /// upstream `cbm_minhash_jaccard`.
    pub fn jaccard(&self, other: &MinHash) -> f64 {
        let matching = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / MINHASH_K as f64
    }

    /// Hamming distance: the number of permutation slots on which the
    /// two signatures **differ** (`0` = identical, `MINHASH_K` = no
    /// agreement). This is `MINHASH_K - matching_slots`, the inverse of
    /// the agreement count behind `jaccard`.
    pub fn hamming(&self, other: &MinHash) -> usize {
        self.values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// `true` if this signature carries no features (all slots
    /// saturated). Used to skip the simhash signal for empty inputs.
    pub fn is_empty(&self) -> bool {
        self.values.iter().all(|&v| v == u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_token_sets_have_jaccard_one_and_hamming_zero() {
        let a = MinHash::from_tokens(["process", "order", "payment"]);
        let b = MinHash::from_tokens(["process", "order", "payment"]);
        assert_eq!(a.jaccard(&b), 1.0);
        assert_eq!(a.hamming(&b), 0);
        assert_eq!(a, b);
    }

    #[test]
    fn order_independent() {
        let a = MinHash::from_tokens(["alpha", "beta", "gamma"]);
        let b = MinHash::from_tokens(["gamma", "alpha", "beta"]);
        assert_eq!(a, b, "MinHash must be a set signature (order-independent)");
    }

    #[test]
    fn disjoint_sets_have_low_jaccard_and_high_hamming() {
        let a = MinHash::from_tokens(["alpha", "beta", "gamma", "delta"]);
        let b = MinHash::from_tokens(["one", "two", "three", "four"]);
        // Disjoint sets: expect very low agreement. (Not necessarily
        // exactly 0 due to hash collisions, but well under half.)
        assert!(a.jaccard(&b) < 0.25, "jaccard was {}", a.jaccard(&b));
        assert!(a.hamming(&b) > MINHASH_K / 2);
    }

    #[test]
    fn partial_overlap_is_between_disjoint_and_identical() {
        let base = MinHash::from_tokens(["a", "b", "c", "d", "e", "f", "g", "h"]);
        let half = MinHash::from_tokens(["a", "b", "c", "d", "x", "y", "z", "w"]);
        let disjoint = MinHash::from_tokens(["p", "q", "r", "s", "t", "u", "v", "ww"]);
        let j_half = base.jaccard(&half);
        let j_disjoint = base.jaccard(&disjoint);
        assert!(
            j_half > j_disjoint,
            "half-overlap ({j_half}) should rank above disjoint ({j_disjoint})"
        );
        assert!(j_half < 1.0);
    }

    #[test]
    fn jaccard_and_hamming_are_complementary() {
        let a = MinHash::from_tokens(["a", "b", "c", "d", "e"]);
        let b = MinHash::from_tokens(["a", "b", "c", "x", "y"]);
        let agree = (a.jaccard(&b) * MINHASH_K as f64).round() as usize;
        assert_eq!(agree + a.hamming(&b), MINHASH_K);
    }

    #[test]
    fn empty_signature_handling() {
        let e = MinHash::from_tokens(Vec::<String>::new());
        assert!(e.is_empty());
        assert_eq!(e, MinHash::empty());
        let real = MinHash::from_tokens(["alpha"]);
        assert!(!real.is_empty());
        // Empty vs real: no agreement on real (saturated slots only
        // match if real also saturates that slot, vanishingly likely).
        assert!(e.jaccard(&real) < 0.25);
    }

    #[test]
    fn deterministic_across_calls() {
        let a = MinHash::from_tokens(["foo", "bar", "baz"]);
        let b = MinHash::from_tokens(["foo", "bar", "baz"]);
        assert_eq!(a, b);
    }
}
