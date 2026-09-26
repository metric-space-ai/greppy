pub(crate) fn cpu_only_allowed(profile: &str, _debug_info: bool) -> bool {
    profile == "debug"
}

#[cfg(test)]
mod tests {
    use super::cpu_only_allowed;

    #[test]
    fn release_profile_refuses_cpu_only_even_with_debug_info() {
        assert!(!cpu_only_allowed("release", true));
    }

    #[test]
    fn debug_profile_allows_cpu_only_even_without_debug_info() {
        assert!(cpu_only_allowed("debug", false));
    }
}
