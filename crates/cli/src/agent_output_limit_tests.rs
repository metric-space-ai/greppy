use super::*;

#[test]
fn caller_output_limit_requires_finite_positive_metadata_and_explicit_model() {
    for value in ["0", "18446744073709551615", "-1", "unknown"] {
        assert!(
            AgentArgs::try_parse_from([
                "greppy -p",
                "task",
                "--model",
                "route-model",
                "--max-output-tokens",
                value
            ])
            .is_err(),
            "invalid caller metadata: {value}"
        );
        assert!(ServeArgs::try_parse_from([
            "greppy agent serve",
            "--model",
            "route-model",
            "--max-output-tokens",
            value
        ])
        .is_err());
    }
    let args = AgentArgs::try_parse_from([
        "greppy -p",
        "task",
        "--model",
        "route-model",
        "--max-output-tokens",
        "131072",
    ])
    .unwrap();
    assert_eq!(args.max_output_tokens.unwrap().get(), 131072);
    // An interactive model chosen later cannot inherit unbound metadata.
    let previous_model = std::env::var_os("GREPPY_MODEL");
    std::env::remove_var("GREPPY_MODEL");
    let unbound = AgentArgs::try_parse_from(["greppy agent", "--max-output-tokens", "131072"]);
    if let Some(previous_model) = previous_model {
        std::env::set_var("GREPPY_MODEL", previous_model);
    }
    assert!(unbound.is_err());
}

#[test]
fn caller_output_limit_is_discarded_by_a_setup_endpoint_override() {
    let previous_endpoint = std::env::var_os("GREPPY_ENDPOINT");
    std::env::remove_var("GREPPY_ENDPOINT");
    let mut args = AgentArgs::try_parse_from([
        "greppy agent",
        "--model",
        "glm-5.3",
        "--max-output-tokens",
        "65536",
    ])
    .unwrap();
    let settings = crate::agent_tui::AgentSettings {
        endpoint: Some("http://127.0.0.1:18318".into()),
        ..crate::agent_tui::AgentSettings::default()
    };
    apply_interactive_settings(&mut args, &[], &settings);
    if let Some(previous_endpoint) = previous_endpoint {
        std::env::set_var("GREPPY_ENDPOINT", previous_endpoint);
    }
    assert_eq!(args.endpoint, "http://127.0.0.1:18318");
    assert_eq!(args.max_output_tokens, None);
}

#[test]
fn caller_output_limit_is_discarded_when_the_session_model_changes() {
    let mut config = AgentConfig::default()
        .with_model("glm-5.3")
        .with_max_tokens(65536);
    select_session_model(&mut config, "glm-5.3".into());
    assert_eq!(config.max_tokens, 65536);
    select_session_model(&mut config, "MiniMax-M3".into());
    assert_eq!(config.max_tokens, u64::MAX);
    select_session_model(&mut config, "glm-5.3".into());
    assert_eq!(
        config.max_tokens,
        u64::MAX,
        "switching back needs fresh caller metadata"
    );
}
