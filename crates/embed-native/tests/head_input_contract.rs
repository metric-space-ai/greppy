use greppy_embed_native::head_input::*;

fn source(text: &str) -> Source {
    Source {
        id: "source-a".into(),
        sha256: sha256(text.as_bytes()),
        text: text.into(),
    }
}
fn candidate(target: Span) -> Candidate {
    Candidate {
        id: "candidate-a".into(),
        head: Head::LogClassifier,
        target,
        context: vec![],
        task: None,
        observation_id: None,
        goal_version: None,
        last_action: None,
    }
}
fn count(s: &str) -> Result<usize, String> {
    Ok(s.chars().count())
}

#[test]
fn entire_long_output_and_utf8_crlf_tail_are_addressable() {
    let text = "neutral\r\n".repeat(100_000) + "error: spätes Ende 🦀";
    let spans = log_spans(&text).collect::<Vec<_>>();
    assert_eq!(spans.len(), 100_001);
    let rebuilt = spans
        .iter()
        .map(|s| &text[s.start..s.end])
        .collect::<String>();
    assert_eq!(rebuilt, text);
    assert!(text[spans.last().unwrap().start..].starts_with("error:"));
    let src = source(&text);
    let verified = VerifiedSource::new(&src).unwrap();
    let prepared = verified
        .prepare(&candidate(*spans.last().unwrap()), Limits::default(), count)
        .unwrap();
    assert!(prepared[0].prompt.contains("spätes Ende"));
}

#[test]
fn oversized_target_splits_without_losing_any_original_byte() {
    let src = source(&"ä🦀\t\"x\r\n".repeat(500));
    let c = candidate(Span {
        start: 0,
        end: src.text.len(),
    });
    let limits = Limits {
        max_tokens: 150,
        max_target_bytes: 512,
        ..Limits::default()
    };
    let rows = VerifiedSource::new(&src)
        .unwrap()
        .prepare(&c, limits, count)
        .unwrap();
    assert!(rows.len() > 1);
    assert_eq!(
        rows.iter()
            .map(|r| &src.text[r.target.start..r.target.end])
            .collect::<String>(),
        src.text
    );
    let mut end = 0;
    for row in rows {
        assert_eq!(row.target.start, end);
        end = row.target.end;
        assert!(row.token_count <= 150);
        assert_eq!(row.input_sha256, sha256(row.prompt.as_bytes()));
    }
}

#[test]
fn target_and_context_are_separate_and_omissions_explicit() {
    let src = source("quoted error\nwarning: retry\ncontext too large\n");
    let spans = log_spans(&src.text).collect::<Vec<_>>();
    let mut c = candidate(spans[1]);
    c.context = vec![spans[0], spans[2]];
    let rows = VerifiedSource::new(&src)
        .unwrap()
        .prepare(
            &c,
            Limits {
                max_context_bytes: 13,
                ..Limits::default()
            },
            count,
        )
        .unwrap();
    assert_eq!(rows[0].context_used, vec![spans[0]]);
    assert_eq!(rows[0].context_omitted, vec![spans[2]]);
    assert!(rows[0].prompt.contains("\"target\":\"warning: retry\\n\""));
}

#[test]
fn relevance_changes_with_task_and_observation_version_invalidates_identity() {
    let src = source("{\"disabled\":true}");
    let mut c = candidate(Span {
        start: 0,
        end: src.text.len(),
    });
    c.head = Head::WebRanker;
    c.task = Some("Edit quantity".into());
    c.observation_id = Some("obs-1".into());
    c.goal_version = Some(1);
    let v = VerifiedSource::new(&src).unwrap();
    let a = v.prepare(&c, Limits::default(), count).unwrap();
    assert!(a[0].prompt.contains("\"last_action\":null"));
    assert!(!a[0].prompt.contains("checked"));
    c.task = Some("Locate the menu".into());
    let b = v.prepare(&c, Limits::default(), count).unwrap();
    assert_ne!(a[0].input_sha256, b[0].input_sha256);
    c.goal_version = Some(2);
    let d = v.prepare(&c, Limits::default(), count).unwrap();
    assert_eq!(b[0].input_sha256, d[0].input_sha256);
    assert_ne!(b[0].conditioning_sha256, d[0].conditioning_sha256);
    assert_ne!(b[0].id, d[0].id);
}

#[test]
fn invalid_sources_overlap_and_utf8_boundaries_rejected() {
    let mut src = source("ä target context");
    src.sha256 = "bad".into();
    assert!(VerifiedSource::new(&src).is_err());
    src.sha256 = sha256(src.text.as_bytes());
    let v = VerifiedSource::new(&src).unwrap();
    assert!(v
        .prepare(
            &candidate(Span { start: 1, end: 4 }),
            Limits::default(),
            count
        )
        .is_err());
    let mut c = candidate(Span { start: 0, end: 3 });
    c.context = vec![Span { start: 2, end: 4 }];
    assert!(v.prepare(&c, Limits::default(), count).is_err());
}

#[test]
fn no_silent_task_or_target_truncation_and_no_partial_budget_result() {
    let src = source(&"x".repeat(100));
    let v = VerifiedSource::new(&src).unwrap();
    let mut c = candidate(Span { start: 0, end: 100 });
    assert!(v
        .prepare(
            &c,
            Limits {
                max_target_bytes: 4,
                max_parts: 1,
                ..Limits::default()
            },
            count
        )
        .is_err());
    c.head = Head::LogRanker;
    c.task = Some("long task ".repeat(100));
    assert!(v
        .prepare(
            &c,
            Limits {
                max_tokens: 100,
                ..Limits::default()
            },
            count
        )
        .is_err());
    c.head = Head::LogClassifier;
    assert!(v.prepare(&c, Limits::default(), count).is_err());
}

#[test]
fn fixed_contract_is_reproducible_and_limits_are_hash_bound() {
    let src = source("error: failed\n");
    let v = VerifiedSource::new(&src).unwrap();
    let c = candidate(Span {
        start: 0,
        end: src.text.len(),
    });
    let a = v.prepare(&c, Limits::default(), count).unwrap();
    let b = v.prepare(&c, Limits::default(), count).unwrap();
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
    assert_ne!(
        contract_hash(Limits::default()).unwrap(),
        contract_hash(Limits {
            max_tokens: 256,
            ..Limits::default()
        })
        .unwrap()
    );
}

#[test]
fn only_lf_splits_physical_lines_and_other_separators_remain_source_bytes() {
    let text =
        "progress\rupdate\u{000b}field\u{000c}form\u{0085}next\u{2028}line\u{2029}para\r\n\nlast\r";
    let parts = log_spans(text)
        .map(|s| &text[s.start..s.end])
        .collect::<Vec<_>>();
    assert_eq!(
        parts,
        vec![
            "progress\rupdate\u{000b}field\u{000c}form\u{0085}next\u{2028}line\u{2029}para\r\n",
            "\n",
            "last\r"
        ]
    );
    assert_eq!(parts.concat(), text);
    assert_eq!(log_spans("").count(), 0);
}
