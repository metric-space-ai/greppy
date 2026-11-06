//! Identity-based live object accounting. Reservations are separate from
//! acknowledged objects; an unknown/idempotent close cannot manufacture credit.
use crate::session::Session;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub struct LiveObjects {
    pages: HashMap<String, Option<String>>,
    contexts: HashMap<String, (Option<String>, bool)>,
    browsers: HashSet<String>,
}
fn id(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}
impl LiveObjects {
    pub fn pages_to_close(&self) -> Vec<String> {
        self.pages.keys().cloned().collect()
    }
    pub fn forget_page(&mut self, page: &str) {
        self.pages.remove(page);
    }
    pub fn contexts_to_close(&self) -> Vec<String> {
        self.contexts.keys().cloned().collect()
    }
    pub fn browsers_to_close(&self) -> Vec<String> {
        self.browsers.iter().cloned().collect()
    }
    fn close_context(&mut self, context: &str) {
        self.contexts.remove(context);
        self.pages
            .retain(|_, owner| owner.as_deref() != Some(context));
    }
}

pub fn validate_creation(session: &Session, method: &str, params: &Value) -> Result<(), String> {
    if method == "context.newPage" {
        if let Some(context) = id(params, "context") {
            if !session.objects.contexts.contains_key(&context)
                && session.engine_context_id.as_deref() != Some(context.as_str())
            {
                return Err("object_disposed: context is not owned by this session".into());
            }
        }
    }
    if method == "browser.newContext" {
        if let Some(browser) = id(params, "browser") {
            if !session.objects.browsers.contains(&browser) {
                return Err("object_disposed: browser is not owned by this session".into());
            }
        }
    }
    if method == "session.ensurePage"
        && id(params, "session").as_deref() != Some(session.id.as_str())
    {
        return Err("session page owner does not match the active session".into());
    }
    Ok(())
}
pub fn validate_close(
    sessions: &HashMap<String, Session>,
    owner: &str,
    method: &str,
    params: &Value,
) -> Result<(), String> {
    let key = match method {
        "page.close" | "session.closePage" => "page",
        "context.close" => "context",
        "browser.close" => "browser",
        _ => return Ok(()),
    };
    let Some(target) = id(params, key) else {
        return Ok(());
    };
    for (session_id, session) in sessions {
        if session_id == owner {
            continue;
        }
        let foreign = match key {
            "page" => {
                session.objects.pages.contains_key(&target)
                    || session.tabs.contains(&target)
                    || session.page_id.as_deref() == Some(target.as_str())
            }
            "context" => {
                session.objects.contexts.contains_key(&target)
                    || session.engine_context_id.as_deref() == Some(target.as_str())
            }
            _ => session.objects.browsers.contains(&target),
        };
        if foreign {
            return Err("object_disposed: close target belongs to another session".into());
        }
    }
    Ok(())
}

pub fn reconcile(session: &mut Session) {
    let untracked_tabs = session
        .tabs
        .iter()
        .filter(|page| !session.objects.pages.contains_key(*page))
        .count();
    session.pages = u32::try_from(session.objects.pages.len() + untracked_tabs)
        .unwrap_or(u32::MAX)
        .saturating_add(session.pending_pages);
    session.contexts = u32::try_from(
        session
            .objects
            .contexts
            .values()
            .filter(|(_, counted)| *counted)
            .count(),
    )
    .unwrap_or(u32::MAX)
    .saturating_add(session.pending_contexts);
}
pub fn note_default_page(session: &mut Session) {
    if let Some(page) = &session.page_id {
        session
            .objects
            .pages
            .insert(page.clone(), session.engine_context_id.clone());
    }
    if let Some(context) = &session.engine_context_id {
        session
            .objects
            .contexts
            .entry(context.clone())
            .or_insert((None, false));
    }
    reconcile(session);
}
pub fn reset(session: &mut Session) {
    session.objects = LiveObjects::default();
    session.pending_pages = 0;
    session.pending_contexts = 0;
    reconcile(session);
}
pub fn after_call(
    session: &mut Session,
    method: &str,
    params: &Value,
    ok: bool,
    result: &Value,
) -> Result<(), String> {
    let page_creation = matches!(method, "context.newPage" | "session.ensurePage");
    let context_creation = method == "browser.newContext";
    if ok && page_creation && id(result, "page").is_none() {
        return Err("successful page creation omitted its object identity".into());
    }
    if ok && context_creation && id(result, "context").is_none() {
        return Err("successful context creation omitted its object identity".into());
    }
    if page_creation {
        session.pending_pages = session.pending_pages.saturating_sub(1);
    }
    if context_creation {
        session.pending_contexts = session.pending_contexts.saturating_sub(1);
    }
    if !ok {
        if page_creation || context_creation {
            reconcile(session);
        }
        return Ok(());
    }
    match method {
        "chromium.launch" => {
            if let Some(browser) = id(result, "browser") {
                session.objects.browsers.insert(browser);
            }
        }
        "browser.newContext" => {
            session.objects.contexts.insert(
                id(result, "context").unwrap(),
                (id(params, "browser"), true),
            );
        }
        "context.newPage" | "session.ensurePage" => {
            let context = id(result, "context").or_else(|| id(params, "context"));
            session
                .objects
                .pages
                .insert(id(result, "page").unwrap(), context.clone());
            if let Some(context) = context {
                session
                    .objects
                    .contexts
                    .entry(context)
                    .or_insert((None, false));
            }
        }
        "session.attachPage" => {
            if let (Some(page), Some(context)) = (id(result, "page"), id(result, "context")) {
                session.objects.pages.insert(page, Some(context.clone()));
                let browser = id(result, "browser");
                session
                    .objects
                    .contexts
                    .entry(context)
                    .and_modify(|entry| entry.0 = browser.clone())
                    .or_insert((browser.clone(), false));
                if let Some(browser) = browser {
                    session.objects.browsers.insert(browser);
                }
            }
        }
        "page.popups" => {
            let owner = id(params, "page")
                .and_then(|page| session.objects.pages.get(&page).cloned())
                .flatten();
            for row in result
                .get("pages")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(page) = id(row, "page") {
                    session.objects.pages.insert(page, owner.clone());
                }
            }
        }
        "page.close" | "session.closePage" => {
            if let Some(page) = id(params, "page") {
                session.objects.forget_page(&page);
                session.tabs.retain(|tab| tab != &page);
                session.locator_snapshots.remove(&page);
                if session.page_id.as_deref() == Some(page.as_str()) {
                    session.page_id = session.tabs.last().cloned();
                }
            }
        }
        "context.close" => {
            if let Some(context) = id(params, "context") {
                session.objects.close_context(&context);
                if session.engine_context_id.as_deref() == Some(context.as_str()) {
                    session.engine_context_id = None;
                    session.page_id = None;
                    session.tabs.clear();
                    session.locator_snapshots.clear();
                }
            }
        }
        "browser.close" => {
            if let Some(browser) = id(params, "browser") {
                session.objects.browsers.remove(&browser);
                let contexts: Vec<_> = session
                    .objects
                    .contexts
                    .iter()
                    .filter(|(_, (owner, _))| owner.as_deref() == Some(browser.as_str()))
                    .map(|(context, _)| context.clone())
                    .collect();
                for context in contexts {
                    session.objects.close_context(&context);
                    if session.engine_context_id.as_deref() == Some(context.as_str()) {
                        session.engine_context_id = None;
                        session.page_id = None;
                        session.tabs.clear();
                        session.locator_snapshots.clear();
                    }
                }
            }
        }
        _ => return Ok(()),
    }
    reconcile(session);
    session.limits.check_pages(session.pages)?;
    session.limits.check_contexts(session.contexts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::NetworkProfile;
    use serde_json::json;
    #[test]
    fn live_object_slots_follow_acknowledged_identity_not_historical_creations() {
        let mut session = Session::new("session", "run", NetworkProfile::Research);
        session.pending_contexts = 1;
        after_call(
            &mut session,
            "browser.newContext",
            &json!({"browser":"b"}),
            true,
            &json!({"context":"c"}),
        )
        .unwrap();
        for i in 0..40 {
            session.pending_pages += 1;
            reconcile(&mut session);
            let page = format!("p{i}");
            after_call(
                &mut session,
                "context.newPage",
                &json!({"context":"c"}),
                true,
                &json!({"page":page}),
            )
            .unwrap();
            assert_eq!(session.pages, 1);
            after_call(
                &mut session,
                "page.close",
                &json!({"page":page}),
                true,
                &json!({}),
            )
            .unwrap();
            assert_eq!(session.pages, 0);
        }
        session.pending_pages = 1;
        after_call(
            &mut session,
            "context.newPage",
            &json!({"context":"c"}),
            false,
            &Value::Null,
        )
        .unwrap();
        assert_eq!(session.pages, 0);
        session.pending_pages = 2;
        after_call(
            &mut session,
            "context.newPage",
            &json!({"context":"c"}),
            true,
            &json!({"page":"live"}),
        )
        .unwrap();
        assert_eq!(session.pages, 2, "other inflight reservation disappeared");
        after_call(
            &mut session,
            "page.close",
            &json!({"page":"unknown"}),
            true,
            &json!({}),
        )
        .unwrap();
        assert_eq!(session.pages, 2, "unknown close created quota credit");
        after_call(
            &mut session,
            "context.newPage",
            &json!({"context":"c"}),
            false,
            &Value::Null,
        )
        .unwrap();
        after_call(
            &mut session,
            "context.close",
            &json!({"context":"c"}),
            true,
            &json!({}),
        )
        .unwrap();
        assert_eq!((session.pages, session.contexts), (0, 0));
        after_call(
            &mut session,
            "context.close",
            &json!({"context":"c"}),
            true,
            &json!({}),
        )
        .unwrap();
        assert_eq!((session.pages, session.contexts), (0, 0));
    }
}
