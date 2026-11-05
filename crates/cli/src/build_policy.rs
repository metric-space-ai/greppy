pub(crate) fn cpu_feature_allowed(profile: &str) -> bool {
    matches!(profile, "debug" | "release" | "release-cpu")
}

#[cfg(test)]
mod tests {
    use super::cpu_feature_allowed;

    #[test]
    fn release_profile_allows_the_cpu_feature() {
        assert!(cpu_feature_allowed("release"));
        assert!(cpu_feature_allowed("release-cpu"));
    }

    #[test]
    fn debug_profile_allows_the_cpu_feature() {
        assert!(cpu_feature_allowed("debug"));
    }
}
