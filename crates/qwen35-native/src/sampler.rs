#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
}

/// Qwen model-card recommendation for non-thinking text tasks, with the
/// requested 64-token hard cap for `brief`.
pub const BRIEF_GENERATION_PARAMS: GenerationParams = GenerationParams {
    temperature: 1.0,
    top_p: 1.0,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 2.0,
    repetition_penalty: 1.0,
    max_tokens: 64,
};

pub const BRIEF_FALLBACK_GENERATION_PARAMS: GenerationParams = GenerationParams {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 1,
    min_p: 0.0,
    presence_penalty: 0.0,
    repetition_penalty: 1.0,
    max_tokens: 64,
};

pub const TRIAGE_GENERATION_PARAMS: GenerationParams = GenerationParams {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 1,
    min_p: 0.0,
    presence_penalty: 0.0,
    repetition_penalty: 1.0,
    max_tokens: 24,
};

#[derive(Debug, Clone)]
pub struct SamplerRng {
    state: u64,
}

impl SamplerRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x.max(1);
        x
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        (value as f32) / ((1u64 << 24) as f32)
    }
}

/// Apply the deterministic parts of the sampling contract in-place.
///
/// The final random draw is intentionally left to the backend implementation:
/// CPU, Metal, and CUDA must share this filtering logic and only differ in how
/// logits are produced.
pub fn apply_sampling_filters(logits: &mut [f32], generated: &[u32], params: GenerationParams) {
    if logits.is_empty() {
        return;
    }
    for &tok in generated {
        let Some(logit) = logits.get_mut(tok as usize) else {
            continue;
        };
        if params.presence_penalty != 0.0 {
            *logit -= params.presence_penalty;
        }
        if params.repetition_penalty != 1.0 {
            if *logit >= 0.0 {
                *logit /= params.repetition_penalty;
            } else {
                *logit *= params.repetition_penalty;
            }
        }
    }

    if params.top_k > 0 && params.top_k < logits.len() {
        let mut sorted = logits.to_vec();
        let cutoff_index = params.top_k - 1;
        sorted.select_nth_unstable_by(cutoff_index, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let cutoff = sorted[cutoff_index];
        for logit in logits.iter_mut() {
            if *logit < cutoff {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    if params.min_p > 0.0 {
        let max = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let min_allowed = max + params.min_p.ln();
        for logit in logits.iter_mut() {
            if *logit < min_allowed {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    if params.top_p < 1.0 {
        apply_top_p(logits, params.top_p.max(0.0));
    }
}

pub fn sample_token(
    logits: &mut [f32],
    generated: &[u32],
    params: GenerationParams,
    rng: &mut SamplerRng,
) -> Option<u32> {
    apply_sampling_filters(logits, generated, params);
    if params.temperature <= 0.0 {
        return logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);
    }
    let temperature = params.temperature.max(1.0e-6);
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    if !max.is_finite() {
        return None;
    }
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        if logit.is_finite() {
            *logit = ((*logit - max) / temperature).exp();
            sum += *logit;
        } else {
            *logit = 0.0;
        }
    }
    if sum <= 0.0 || !sum.is_finite() {
        return logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);
    }
    let mut pick = rng.next_f32() * sum;
    for (idx, prob) in logits.iter().copied().enumerate() {
        pick -= prob;
        if pick <= 0.0 {
            return Some(idx as u32);
        }
    }
    logits.iter().rposition(|p| *p > 0.0).map(|idx| idx as u32)
}

fn apply_top_p(logits: &mut [f32], top_p: f32) {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut probs = logits
        .iter()
        .enumerate()
        .filter_map(|(idx, &logit)| logit.is_finite().then_some((idx, (logit - max).exp())))
        .collect::<Vec<_>>();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return;
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0_f32;
    let mut keep = std::collections::BTreeSet::new();
    for (idx, prob) in probs {
        keep.insert(idx);
        cumulative += prob;
        if cumulative >= top_p {
            break;
        }
    }
    for (idx, logit) in logits.iter_mut().enumerate() {
        if !keep.contains(&idx) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_params_match_contract() {
        assert_eq!(BRIEF_GENERATION_PARAMS.temperature, 1.0);
        assert_eq!(BRIEF_GENERATION_PARAMS.top_p, 1.0);
        assert_eq!(BRIEF_GENERATION_PARAMS.top_k, 20);
        assert_eq!(BRIEF_GENERATION_PARAMS.min_p, 0.0);
        assert_eq!(BRIEF_GENERATION_PARAMS.presence_penalty, 2.0);
        assert_eq!(BRIEF_GENERATION_PARAMS.repetition_penalty, 1.0);
        assert_eq!(BRIEF_GENERATION_PARAMS.max_tokens, 64);
    }

    #[test]
    fn brief_fallback_params_are_greedy() {
        assert_eq!(BRIEF_FALLBACK_GENERATION_PARAMS.temperature, 0.0);
        assert_eq!(BRIEF_FALLBACK_GENERATION_PARAMS.top_k, 1);
        assert_eq!(BRIEF_FALLBACK_GENERATION_PARAMS.presence_penalty, 0.0);
        assert_eq!(BRIEF_FALLBACK_GENERATION_PARAMS.max_tokens, 64);
    }

    #[test]
    fn triage_params_are_greedy_and_tiny() {
        assert_eq!(TRIAGE_GENERATION_PARAMS.temperature, 0.0);
        assert_eq!(TRIAGE_GENERATION_PARAMS.top_k, 1);
        assert!(TRIAGE_GENERATION_PARAMS.max_tokens <= 24);
    }

    #[test]
    fn presence_penalty_is_applied() {
        let mut logits = vec![0.0, 10.0, 5.0];
        apply_sampling_filters(&mut logits, &[1], BRIEF_GENERATION_PARAMS);
        assert_eq!(logits[1], 8.0);
    }

    #[test]
    fn sample_token_obeys_top_k() {
        let mut logits = vec![10.0, 9.0, 8.0, 7.0];
        let params = GenerationParams {
            top_k: 1,
            ..BRIEF_GENERATION_PARAMS
        };
        let mut rng = SamplerRng::new(7);
        assert_eq!(sample_token(&mut logits, &[], params, &mut rng), Some(0));
    }

    #[test]
    fn sample_token_is_greedy_at_temperature_zero() {
        let mut logits = vec![1.0, 4.0, 3.0];
        let mut rng = SamplerRng::new(7);
        assert_eq!(
            sample_token(&mut logits, &[], TRIAGE_GENERATION_PARAMS, &mut rng),
            Some(1)
        );
    }
}
