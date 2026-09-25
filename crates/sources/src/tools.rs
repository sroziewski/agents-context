//! One-line summaries of tool calls, shared by both sources. Claude uses `PascalCase` tool names
//! and `snake_case` inputs, OpenCode `lowercase` names and `camelCase` inputs.

use serde_json::Value;

const MAX_DETAIL: usize = 120;

pub struct ToolCall {
    pub summary: String,
    /// Set when the call edited or wrote a file.
    pub edited_file: Option<String>,
}

pub fn summarise(name: &str, input: &Value) -> ToolCall {
    let field = |keys: &[&str]| keys.iter().find_map(|k| input.get(*k).and_then(Value::as_str));
    let path = field(&["file_path", "filePath", "notebook_path", "path"]);
    let lower = name.to_ascii_lowercase();
    let detail = match lower.as_str() {
        "bash" | "powershell" => field(&["description"]).or(field(&["command"])),
        "read" | "edit" | "write" | "multiedit" | "notebookedit" | "patch" => path,
        "grep" | "glob" => field(&["pattern"]),
        "agent" | "task" => field(&["description", "subagent_type"]),
        "webfetch" => field(&["url"]),
        "websearch" => field(&["query"]),
        _ => None,
    };
    let summary = match detail.map(first_line) {
        Some(d) if !d.is_empty() => format!("{name} {}", truncate(d, MAX_DETAIL)),
        _ => name.to_owned(),
    };
    let edits = matches!(lower.as_str(), "edit" | "write" | "multiedit" | "notebookedit" | "patch");
    ToolCall { summary, edited_file: if edits { path.map(str::to_owned) } else { None } }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("").trim()
}

pub fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_owned()
    } else {
        let mut t: String = s.chars().take(max_chars).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn claude_edit_reports_path_and_edited_file() {
        let t = summarise("Edit", &json!({"file_path": "src/db.ts", "old_string": "a"}));
        assert_eq!(t.summary, "Edit src/db.ts");
        assert_eq!(t.edited_file.as_deref(), Some("src/db.ts"));
    }

    #[test]
    fn opencode_camel_case_inputs_are_understood() {
        let t = summarise("write", &json!({"filePath": "D:\\x\\a.rs"}));
        assert_eq!(t.summary, "write D:\\x\\a.rs");
        assert_eq!(t.edited_file.as_deref(), Some("D:\\x\\a.rs"));
    }

    #[test]
    fn read_is_summarised_but_not_an_edit() {
        let t = summarise("Read", &json!({"file_path": "a.txt"}));
        assert_eq!(t.summary, "Read a.txt");
        assert!(t.edited_file.is_none());
    }

    #[test]
    fn bash_prefers_description_over_command() {
        let with = summarise("Bash", &json!({"command": "npm test", "description": "Run tests"}));
        assert_eq!(with.summary, "Bash Run tests");
        let without = summarise("bash", &json!({"command": "npm test\nsecond line"}));
        assert_eq!(without.summary, "bash npm test");
    }

    #[test]
    fn long_details_are_truncated() {
        let cmd = "x".repeat(500);
        let t = summarise("Bash", &json!({"command": cmd}));
        assert_eq!(t.summary.chars().count(), "Bash ".len() + MAX_DETAIL + 1);
        assert!(t.summary.ends_with('…'));
    }

    #[test]
    fn unknown_tool_or_missing_input_is_just_the_name() {
        assert_eq!(summarise("TodoWrite", &json!({"todos": []})).summary, "TodoWrite");
        assert_eq!(summarise("Edit", &json!({})).summary, "Edit");
        assert_eq!(summarise("Grep", &Value::Null).summary, "Grep");
    }

    #[test]
    fn truncate_counts_characters_not_bytes() {
        assert_eq!(truncate("zażółć", 3), "zaż…");
        assert_eq!(truncate("abc", 3), "abc");
    }
}
