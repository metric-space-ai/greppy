//! Overlay/picker state used by help, model, session, tools, and confirms.

use super::commands::{CommandSpec, COMMANDS, KEYBINDINGS};
use super::session::SessionRecord;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Overlay {
    None,
    Help,
    ConfirmClear,
    Usage,
    TooSmall { cols: u16, rows: u16 },
    Model(Picker),
    Sessions(Picker),
    Tools(ToolOverlay),
    Setup(SetupMenu),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picker {
    pub title: String,
    pub filter: String,
    pub items: Vec<PickerItem>,
    pub selected: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerItem {
    pub id: String,
    pub label: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOverlay {
    pub selected: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupMenu {
    pub selected: usize,
}

impl SetupMenu {
    pub const ITEM_COUNT: usize = 9;

    pub fn move_next(&mut self) {
        self.selected = (self.selected + 1) % Self::ITEM_COUNT;
    }

    pub fn move_prev(&mut self) {
        self.selected = if self.selected == 0 {
            Self::ITEM_COUNT - 1
        } else {
            self.selected - 1
        };
    }
}

impl Overlay {
    pub fn is_open(&self) -> bool {
        !matches!(self, Overlay::None)
    }

    pub fn help() -> Self {
        Overlay::Help
    }

    pub fn setup() -> Self {
        Overlay::Setup(SetupMenu { selected: 0 })
    }

    pub fn models(models: &[String], current: &str, query: &str) -> Self {
        let mut items: Vec<PickerItem> = models
            .iter()
            .map(|id| PickerItem {
                id: id.clone(),
                label: id.clone(),
                detail: if id == current {
                    "current".into()
                } else {
                    String::new()
                },
            })
            .collect();
        if items.is_empty() {
            items.push(PickerItem {
                id: current.to_string(),
                label: current.to_string(),
                detail: "current".into(),
            });
        }
        let mut picker = Picker {
            title: "models".into(),
            filter: query.to_string(),
            items,
            selected: 0,
        };
        picker.apply_filter();
        Overlay::Model(picker)
    }

    pub fn sessions(records: &[SessionRecord], query: &str) -> Self {
        let items = records
            .iter()
            .map(|record| PickerItem {
                id: record.id.clone(),
                label: format!("{}  {}", record.id, record.title),
                detail: record.model.clone(),
            })
            .collect();
        let mut picker = Picker {
            title: "sessions".into(),
            filter: query.to_string(),
            items,
            selected: 0,
        };
        picker.apply_filter();
        Overlay::Sessions(picker)
    }

    pub fn help_lines() -> Vec<String> {
        let mut lines = vec!["Commands".to_string(), String::new()];
        for CommandSpec { name, summary } in COMMANDS {
            lines.push(format!("{name:<12} {summary}"));
        }
        lines.push(String::new());
        lines.push("Keys".to_string());
        lines.push(String::new());
        for binding in KEYBINDINGS {
            lines.push((*binding).to_string());
        }
        lines
    }
}

impl Picker {
    pub fn visible(&self) -> Vec<&PickerItem> {
        let needle = self.filter.to_ascii_lowercase();
        self.items
            .iter()
            .filter(|item| {
                needle.is_empty()
                    || item.label.to_ascii_lowercase().contains(&needle)
                    || item.id.to_ascii_lowercase().contains(&needle)
            })
            .collect()
    }

    pub fn apply_filter(&mut self) {
        let visible = self.visible().len();
        if visible == 0 {
            self.selected = 0;
        } else {
            self.selected = self.selected.min(visible - 1);
        }
    }

    pub fn move_next(&mut self) {
        let visible = self.visible().len();
        if visible == 0 {
            return;
        }
        self.selected = (self.selected + 1) % visible;
    }

    pub fn move_prev(&mut self) {
        let visible = self.visible().len();
        if visible == 0 {
            return;
        }
        self.selected = if self.selected == 0 {
            visible - 1
        } else {
            self.selected - 1
        };
    }

    pub fn selected_item(&self) -> Option<&PickerItem> {
        self.visible().get(self.selected).copied()
    }

    pub fn push_filter(&mut self, ch: char) {
        if !ch.is_control() {
            self.filter.push(ch);
            self.apply_filter();
        }
    }

    pub fn pop_filter(&mut self) {
        self.filter.pop();
        self.apply_filter();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_filters_and_wraps() {
        let mut picker = Picker {
            title: "models".into(),
            filter: String::new(),
            items: vec![
                PickerItem {
                    id: "alpha".into(),
                    label: "alpha".into(),
                    detail: String::new(),
                },
                PickerItem {
                    id: "beta".into(),
                    label: "beta".into(),
                    detail: String::new(),
                },
            ],
            selected: 0,
        };
        picker.move_next();
        assert_eq!(picker.selected_item().unwrap().id, "beta");
        picker.push_filter('l');
        assert_eq!(picker.visible().len(), 1);
        assert_eq!(picker.selected_item().unwrap().id, "alpha");
    }
}
