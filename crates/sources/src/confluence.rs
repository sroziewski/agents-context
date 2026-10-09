//! Read-only Confluence Data Center client (REST `/rest/api`, Bearer PAT), plus the conversion of
//! a page's storage format (XHTML with `ac:` macros) to plain text for indexing.

use serde_json::Value;

#[derive(Debug)]
pub enum ConfluenceError {
    /// 401/403 — token missing, wrong, or lacking permission.
    Auth,
    /// 404 — no such space/page.
    NotFound(String),
    Http(String),
    Parse(String),
}

impl std::fmt::Display for ConfluenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth => write!(f, "Confluence authentication failed (401/403); check CONFLUENCE_TOKEN in the config .env"),
            Self::NotFound(what) => write!(f, "Confluence: {what} not found (404)"),
            Self::Http(e) => write!(f, "Confluence request failed: {e}"),
            Self::Parse(e) => write!(f, "Confluence response could not be parsed: {e}"),
        }
    }
}

impl std::error::Error for ConfluenceError {}

pub type Result<T> = std::result::Result<T, ConfluenceError>;

#[derive(Debug, Clone, PartialEq)]
pub struct Space {
    pub key: String,
    pub name: String,
}

/// A page as listed in its space: just enough to tell whether it changed since the last index.
#[derive(Debug, Clone, PartialEq)]
pub struct PageRef {
    pub id: String,
    pub title: String,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    pub id: String,
    pub space: String,
    pub title: String,
    pub version: u64,
    /// Last edit, ISO 8601 as Confluence returns it.
    pub updated: String,
    pub author: String,
    /// Titles of the parent pages, root first.
    pub ancestors: Vec<String>,
    /// The body converted to plain text.
    pub text: String,
    pub url: String,
}

/// Page size for the paginated listings; Confluence caps it server-side anyway.
const PAGE: u32 = 100;
/// Far beyond any real space; reached only by a server that keeps sending `next`.
const MAX_LISTED: u32 = 200_000;

#[derive(Clone)]
pub struct Confluence {
    agent: ureq::Agent,
    base: String,
    token: String,
}

impl Confluence {
    pub fn new(base: &str, token: &str) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(60))).build().into();
        Self { agent, base: base.trim_end_matches('/').to_owned(), token: token.to_owned() }
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let url = format!("{}{path}", self.base);
        match self.agent.get(&url).header("Authorization", format!("Bearer {}", self.token)).header("Accept", "application/json").call() {
            Ok(mut resp) => resp.body_mut().with_config().limit(64 * 1024 * 1024).read_json().map_err(|e| ConfluenceError::Parse(e.to_string())),
            Err(ureq::Error::StatusCode(401 | 403)) => Err(ConfluenceError::Auth),
            Err(ureq::Error::StatusCode(404)) => Err(ConfluenceError::NotFound(path.to_owned())),
            Err(e) => Err(ConfluenceError::Http(e.to_string())),
        }
    }

    /// Every result of a paginated `/rest/api` listing. Follows `_links.next` rather than stopping
    /// on a short page, so a server that caps `limit` below what was asked is still walked to the
    /// end. A listing that never ends is an error, not a truncated result: the sync deletes pages
    /// missing from a listing, so a cut-off one would drop everything past the cut.
    fn all_results(&self, path: &str) -> Result<Vec<Value>> {
        let sep = if path.contains('?') { '&' } else { '?' };
        let mut out = Vec::new();
        let mut start = 0u32;
        loop {
            let v = self.get_json(&format!("{path}{sep}start={start}&limit={PAGE}"))?;
            let results = v.get("results").and_then(Value::as_array).cloned().unwrap_or_default();
            let n = results.len() as u32;
            out.extend(results);
            if n == 0 || v.pointer("/_links/next").is_none() {
                return Ok(out);
            }
            start += n;
            if start >= MAX_LISTED {
                return Err(ConfluenceError::Http(format!("{path}: listing did not end after {MAX_LISTED} results")));
            }
        }
    }

    /// Global spaces visible to the token (personal spaces are left out).
    pub fn spaces(&self) -> Result<Vec<Space>> {
        let all = self.all_results("/rest/api/space?type=global")?;
        Ok(all.iter().map(|s| Space { key: string(s, "key"), name: string(s, "name") }).filter(|s| !s.key.is_empty()).collect())
    }

    /// Current pages of a space with their version numbers, without bodies.
    pub fn page_refs(&self, space: &str) -> Result<Vec<PageRef>> {
        let path = format!("/rest/api/content?spaceKey={}&type=page&status=current&expand=version", url_encode(space));
        let all = self.all_results(&path)?;
        Ok(all.iter().map(page_ref).filter(|p| !p.id.is_empty()).collect())
    }

    /// A page and every page below it, with version numbers, without bodies.
    pub fn tree_refs(&self, root: &str) -> Result<Vec<PageRef>> {
        let v = self.get_json(&format!("/rest/api/content/{}?expand=version", url_encode(root)))?;
        let mut out = vec![page_ref(&v)];
        // CQL, not `/content/{id}/descendant/page`: that endpoint answers 500 on OPI's Data Center.
        let cql = format!("ancestor={root} and type=page");
        let below = self.all_results(&format!("/rest/api/content/search?cql={}&expand=version", url_encode(&cql)))?;
        out.extend(below.iter().map(page_ref));
        out.retain(|p| !p.id.is_empty());
        Ok(out)
    }

    /// One page with its body as text, its breadcrumb and its last editor.
    pub fn page(&self, id: &str) -> Result<Page> {
        let v = self.get_json(&format!("/rest/api/content/{}?expand=body.storage,version,space,ancestors", url_encode(id)))?;
        Ok(self.parse_page(&v))
    }

    fn parse_page(&self, v: &Value) -> Page {
        let id = string(v, "id");
        let webui = v.pointer("/_links/webui").and_then(Value::as_str).unwrap_or_default();
        Page {
            space: v.pointer("/space/key").and_then(Value::as_str).unwrap_or_default().to_owned(),
            title: string(v, "title"),
            version: v.pointer("/version/number").and_then(Value::as_u64).unwrap_or(0),
            updated: v.pointer("/version/when").and_then(Value::as_str).unwrap_or_default().to_owned(),
            author: v.pointer("/version/by/displayName").and_then(Value::as_str).unwrap_or_default().to_owned(),
            ancestors: v.get("ancestors").and_then(Value::as_array).map(|a| a.iter().map(|x| string(x, "title")).filter(|t| !t.is_empty()).collect()).unwrap_or_default(),
            text: storage_to_text(v.pointer("/body/storage/value").and_then(Value::as_str).unwrap_or_default()),
            url: if webui.is_empty() { format!("{}/pages/viewpage.action?pageId={id}", self.base) } else { format!("{}{webui}", self.base) },
            id,
        }
    }
}

fn page_ref(p: &Value) -> PageRef {
    PageRef { id: string(p, "id"), title: string(p, "title"), version: p.pointer("/version/number").and_then(Value::as_u64).unwrap_or(0) }
}

fn string(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or_default().to_owned()
}

fn url_encode(s: &str) -> String {
    crate::jira::url_encode(s)
}

/// Tags after which the text breaks a line, so paragraphs, list items and table rows stay apart.
const BLOCK_TAGS: &[&str] = &["p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "table", "ul", "ol", "hr", "ac:task"];
/// Tags whose content is not text a reader sees (macro parameters, link targets, styles).
const SKIPPED_TAGS: &[&str] = &["ac:parameter", "style", "script", "ri:attachment", "ri:page", "ri:url", "ri:user", "ac:task-id", "ac:task-status"];

/// Plain text of a Confluence storage-format body: tags dropped, block tags turned into line
/// breaks, CDATA (code macros) kept, entities decoded, blank runs folded. A table row becomes one
/// line (see [`Table`]). A forgiving scan rather than an XML parser: storage bodies are not
/// always well-formed.
pub fn storage_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut tables: Vec<Table> = Vec::new();
    let mut skip_depth = 0usize;
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        if skip_depth == 0 {
            emit(&mut tables, &mut out, &decode_entities(&rest[..lt]));
        }
        rest = &rest[lt..];
        if let Some(body) = rest.strip_prefix("<![CDATA[") {
            let end = body.find("]]>").unwrap_or(body.len());
            if skip_depth == 0 {
                emit(&mut tables, &mut out, &body[..end]);
            }
            rest = body.get(end + 3..).unwrap_or("");
            continue;
        }
        if let Some(body) = rest.strip_prefix("<!--") {
            rest = body.find("-->").map_or("", |e| &body[e + 3..]);
            continue;
        }
        let Some(gt) = rest.find('>') else {
            rest = "";
            break;
        };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let self_closing = tag.ends_with('/');
        let name = tag.trim_start_matches('/').split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("").to_ascii_lowercase();
        if SKIPPED_TAGS.contains(&name.as_str()) {
            if self_closing {
                continue;
            }
            if closing {
                skip_depth = skip_depth.saturating_sub(1);
            } else {
                skip_depth += 1;
            }
            continue;
        }
        if skip_depth > 0 {
            continue;
        }
        match (name.as_str(), closing) {
            ("table", false) => tables.push(Table::default()),
            ("table", true) => {
                if let Some(mut t) = tables.pop() {
                    let row = t.finish();
                    emit_row(&mut tables, &mut out, row);
                }
            }
            ("tr", _) if !tables.is_empty() => {
                let (table, outer) = tables.split_last_mut().expect("checked");
                emit_row(outer, &mut out, table.end_row());
            }
            ("td" | "th", false) if !tables.is_empty() => tables.last_mut().expect("checked").start_cell(name == "th"),
            ("td" | "th", true) if !tables.is_empty() => tables.last_mut().expect("checked").end_cell(),
            ("time", false) => {
                // A date macro keeps its value in an attribute; without it "Release: <date>" reads "Release:".
                emit(&mut tables, &mut out, &attribute(tag, "datetime").unwrap_or_default());
            }
            _ if BLOCK_TAGS.contains(&name.as_str()) => emit(&mut tables, &mut out, "\n"),
            _ => {}
        }
    }
    if skip_depth == 0 {
        emit(&mut tables, &mut out, &decode_entities(rest));
    }
    while let Some(mut t) = tables.pop() {
        let row = t.finish();
        emit_row(&mut tables, &mut out, row);
    }
    tidy(&out)
}

/// A table being read. Each row becomes one line, so a row is never split between chunks and
/// its cells stay together; paragraphs and lists inside a cell become spaces. When the first row
/// is a header (all `th`), the rows below are labelled with it (`ID: K002 | Treść: …`): a chunk
/// from the middle of a long table still says which column a value is in, and empty columns
/// vanish instead of leaving runs of `|`.
#[derive(Default)]
struct Table {
    header: Option<Vec<String>>,
    rows: usize,
    row: Vec<(String, bool)>,
    cell: Option<(String, bool)>,
}

impl Table {
    fn start_cell(&mut self, th: bool) {
        self.end_cell();
        self.cell = Some((String::new(), th));
    }

    fn end_cell(&mut self) {
        if let Some((text, th)) = self.cell.take() {
            self.row.push((text.split_whitespace().collect::<Vec<_>>().join(" "), th));
        }
    }

    /// The last row; a table that is only a header row keeps it as its text.
    fn finish(&mut self) -> Option<String> {
        let last = self.end_row();
        match self.header.take() {
            Some(h) if self.rows == 1 => Some(h.into_iter().filter(|c| !c.is_empty()).collect::<Vec<_>>().join(" | ")),
            _ => last,
        }
    }

    /// The finished row as one line; `None` for an empty row and for a header row once it has
    /// been taken as the labels.
    fn end_row(&mut self) -> Option<String> {
        self.end_cell();
        let row = std::mem::take(&mut self.row);
        if row.iter().all(|(c, _)| c.is_empty()) {
            return None;
        }
        self.rows += 1;
        let cells: Vec<String> = row.iter().map(|(c, _)| c.clone()).collect();
        if self.rows == 1 && row.iter().all(|(c, th)| *th || c.is_empty()) {
            self.header = Some(cells);
            return None;
        }
        match &self.header {
            Some(h) if h.len() == cells.len() => Some(
                h.iter()
                    .zip(&cells)
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(k, v)| if k.is_empty() { v.clone() } else { format!("{k}: {v}") })
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
            // No header, or a row that spans columns: positional, trailing blanks dropped.
            _ => {
                let used = cells.iter().rposition(|c| !c.is_empty()).map_or(0, |i| i + 1);
                Some(cells[..used].join(" | "))
            }
        }
    }
}

/// Text goes into the open cell of the innermost table, or to the output outside any cell. A
/// line break inside a cell is a space: the row stays one line.
fn emit(tables: &mut [Table], out: &mut String, s: &str) {
    match tables.last_mut() {
        Some(Table { cell: Some((cell, _)), .. }) => {
            cell.push_str(if s == "\n" { " " } else { s });
        }
        Some(_) if s.trim().is_empty() => {}
        _ => out.push_str(s),
    }
}

/// A finished row: its own line, or, for a table nested in a cell, part of that cell.
fn emit_row(tables: &mut [Table], out: &mut String, row: Option<String>) {
    let Some(row) = row else { return };
    match tables.last_mut() {
        Some(Table { cell: Some((cell, _)), .. }) => {
            if !cell.trim().is_empty() {
                cell.push_str("; ");
            }
            cell.push_str(&row);
        }
        _ => {
            out.push('\n');
            out.push_str(&row);
            out.push('\n');
        }
    }
}

/// The value of `name="…"` in a tag's inside, e.g. `datetime` of `time datetime="2026-10-01" /`.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let start = tag.find(&format!("{name}=\""))? + name.len() + 2;
    let len = tag[start..].find('"')?;
    Some(decode_entities(&tag[start..start + len]))
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        // Entities are short; look for the `;` within a few characters (not bytes: a byte cut
        // could land inside a multibyte letter).
        let Some(semi) = rest.char_indices().take(12).find(|&(_, c)| c == ';').map(|(i, _)| i) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..semi];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "ndash" => Some('–'),
            "mdash" => Some('—'),
            "hellip" => Some('…'),
            "bdquo" => Some('„'),
            "rdquo" => Some('”'),
            "ldquo" => Some('“'),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Trims each line, collapses runs of spaces and drops blank lines: every block tag opens and
/// closes with a break, so blank lines carry no structure.
fn tidy(s: &str) -> String {
    let lines: Vec<String> = s
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ").trim_start_matches("| ").trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn storage_format_becomes_readable_text() {
        let html = r#"<h1>Wdrożenie</h1><p>Krok&nbsp;1: zaloguj się &amp; uruchom <strong>deploy</strong>.</p>
<ul><li>prod</li><li>test</li></ul>
<table><tbody><tr><th>Env</th><th>URL</th></tr><tr><td>prod</td><td>https://x</td></tr></tbody></table>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[kubectl apply -f a.yaml]]></ac:plain-text-body></ac:structured-macro>
<p>Link: <ac:link><ri:page ri:content-title="Inna strona" /></ac:link> koniec</p><!-- comment -->"#;
        let text = storage_to_text(html);
        assert!(text.starts_with("Wdrożenie\n"), "{text}");
        assert!(text.contains("Krok 1: zaloguj się & uruchom deploy."), "{text}");
        assert!(text.contains("prod\ntest"), "{text}");
        assert!(text.contains("Env: prod | URL: https://x"), "{text}");
        assert!(text.contains("kubectl apply -f a.yaml"), "code macros keep their CDATA: {text}");
        assert!(!text.contains("bash"), "macro parameters are not text: {text}");
        assert!(!text.contains("comment") && !text.contains('<'), "{text}");
    }

    #[test]
    fn a_table_row_is_one_labelled_line_without_empty_columns() {
        // The shape of SEDN's "Alerty, komunikaty błędów": paragraphs inside cells, mostly empty
        // placeholder columns. Read cell by cell it was K001's text on K002's line.
        let html = "<table><tbody><tr><th><p>ID</p></th><th><p>Opis</p></th><th><p>Treść</p></th><th><p>x1</p></th><th><p>Wyjątek</p></th></tr>\
<tr><td><p>K001</p></td><td><p>Brak roli.</p><p>Albo brak połączenia kont.</p></td><td><p>Brak uprawnień.</p></td><td><p /></td><td><br /></td></tr>\
<tr><td>K002</td><td>Błędne dane logowania.</td><td>Niepoprawne dane logowania.</td><td></td><td>AuthException</td></tr>\
<tr><td colspan=\"5\"><strong>Sekcja: hasła</strong></td></tr></tbody></table><p>Po tabeli</p>";
        assert_eq!(
            storage_to_text(html),
            "ID: K001 | Opis: Brak roli. Albo brak połączenia kont. | Treść: Brak uprawnień.\n\
ID: K002 | Opis: Błędne dane logowania. | Treść: Niepoprawne dane logowania. | Wyjątek: AuthException\n\
Sekcja: hasła\n\
Po tabeli"
        );
    }

    #[test]
    fn tables_without_a_header_stay_positional_and_nested_tables_stay_in_their_cell() {
        let html = "<table><tr><th>Env</th><td>prod</td><td></td></tr><tr><th>URL</th><td>https://x</td></tr></table>";
        assert_eq!(storage_to_text(html), "Env | prod\nURL | https://x");
        let nested = "<table><tr><td>A</td><td><table><tr><td>a1</td><td>a2</td></tr><tr><td>b1</td></tr></table></td></tr></table>";
        assert_eq!(storage_to_text(nested), "A | a1 | a2; b1");
        assert_eq!(storage_to_text("<table><tr><th>Tylko</th><th>nagłówek</th></tr></table>"), "Tylko | nagłówek");
        assert_eq!(storage_to_text("<table><tr><td>niezamknięta"), "niezamknięta");
    }

    #[test]
    fn numeric_entities_and_stray_ampersands_survive() {
        assert_eq!(storage_to_text("<p>&#261;&#x15B; R&D &unknown; x</p>"), "ąś R&D &unknown; x");
    }

    #[test]
    fn a_date_macro_keeps_its_date() {
        assert_eq!(storage_to_text(r#"<p>Release: <time datetime="2026-11-03" /> po testach</p>"#), "Release: 2026-11-03 po testach");
        assert_eq!(storage_to_text("<p>bez daty: <time /></p>"), "bez daty:");
    }

    #[test]
    fn an_ampersand_before_multibyte_letters_does_not_panic() {
        assert_eq!(storage_to_text("<p>R&ąęśćżźółń dalej</p>"), "R&ąęśćżźółń dalej");
    }

    #[test]
    fn a_truncated_tag_does_not_panic() {
        assert_eq!(storage_to_text("<p>tekst</p><b"), "tekst");
        assert_eq!(storage_to_text("<![CDATA[open"), "open");
    }

    #[test]
    fn nested_macros_keep_their_bodies_and_drop_attachments_and_parameters() {
        let html = r#"<ac:structured-macro ac:name="info"><ac:parameter ac:name="title">Uwaga</ac:parameter><ac:rich-text-body><p>Przed wdrożeniem:</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">sql</ac:parameter><ac:plain-text-body><![CDATA[SELECT * FROM t WHERE a < 3 && b > 1;]]></ac:plain-text-body></ac:structured-macro>
<ac:structured-macro ac:name="expand"><ac:rich-text-body><ul><li>zrób kopię</li><li>sprawdź logi</li></ul></ac:rich-text-body></ac:structured-macro>
</ac:rich-text-body></ac:structured-macro>
<p><ac:image ac:height="250"><ri:attachment ri:filename="schemat-bazy.png" /></ac:image>Schemat</p>
<p><ac:link><ri:attachment ri:filename="raport.pdf"><ri:page ri:content-title="Archiwum" /></ri:attachment><ac:plain-text-link-body><![CDATA[raport]]></ac:plain-text-link-body></ac:link> po linku</p>
<p><ac:link><ri:user ri:userkey="8a7f" /></ac:link> zatwierdza</p>"#;
        let text = storage_to_text(html);
        assert!(text.contains("Przed wdrożeniem:"), "{text}");
        assert!(text.contains("SELECT * FROM t WHERE a < 3 && b > 1;"), "CDATA is raw, not entity-decoded: {text}");
        assert!(text.contains("zrób kopię\nsprawdź logi"), "{text}");
        assert!(text.contains("Schemat") && text.contains("raport po linku") && text.contains("zatwierdza"), "{text}");
        for hidden in ["Uwaga", "sql", "schemat-bazy.png", "raport.pdf", "Archiwum", "8a7f"] {
            assert!(!text.contains(hidden), "{hidden} leaked: {text}");
        }
    }

    #[test]
    fn multibyte_text_right_at_tags_and_entities_is_kept_intact() {
        assert_eq!(storage_to_text("<p>żółć</p><p>Ąę<br/>ŚĆ</p><table><tr><td>ł</td><td>ń</td></tr></table>"), "żółć\nĄę\nŚĆ\nł | ń");
        assert_eq!(storage_to_text("<p>&bdquo;Zażółć&rdquo;&nbsp;gęślą&mdash;jaźń</p>"), "„Zażółć” gęślą—jaźń");
        assert_eq!(storage_to_text("<p>zażółć"), "zażółć");
        assert_eq!(storage_to_text("ąę<ż"), "ąę");
    }

    #[test]
    fn task_list_items_stay_apart_without_their_ids_and_statuses() {
        let html = "<ac:task-list><ac:task><ac:task-id>11</ac:task-id><ac:task-status>incomplete</ac:task-status><ac:task-body>Zrobić backup</ac:task-body></ac:task>\
<ac:task><ac:task-id>12</ac:task-id><ac:task-status>complete</ac:task-status><ac:task-body>Wgrać migrację</ac:task-body></ac:task></ac:task-list>";
        assert_eq!(storage_to_text(html), "Zrobić backup\nWgrać migrację");
    }

    #[test]
    fn malformed_bodies_degrade_without_panicking() {
        for html in ["<", "<>", "</>", "<!--", "<![CDATA[", "]]>", "&", "&;", "&#;", "&#xZZ;", "&#99999999;", "<ac:parameter>never closed", "</ac:parameter>after", "<p<p>x</p>"] {
            let _ = storage_to_text(html);
        }
        assert_eq!(storage_to_text("<ac:parameter>never closed"), "");
        assert_eq!(storage_to_text("</ac:parameter>after"), "after", "an unmatched close does not start skipping");
    }

    #[test]
    fn parses_a_page_with_breadcrumb_and_editor() {
        let c = Confluence::new("https://confluence.example.org/", "tok");
        let p = c.parse_page(&json!({
            "id": "123", "title": "Runbook",
            "space": {"key": "SEDN"},
            "version": {"number": 7, "when": "2026-10-01T10:00:00.000+02:00", "by": {"displayName": "Anna W"}},
            "ancestors": [{"title": "Home"}, {"title": "Ops"}],
            "body": {"storage": {"value": "<p>restart the pod</p>"}},
            "_links": {"webui": "/display/SEDN/Runbook"}
        }));
        assert_eq!((p.id.as_str(), p.space.as_str(), p.version, p.author.as_str()), ("123", "SEDN", 7, "Anna W"));
        assert_eq!(p.ancestors, ["Home", "Ops"]);
        assert_eq!(p.text, "restart the pod");
        assert_eq!(p.url, "https://confluence.example.org/display/SEDN/Runbook");
    }

    #[test]
    fn a_page_without_a_webui_link_gets_the_viewpage_url() {
        let p = Confluence::new("https://c.example.org", "t").parse_page(&json!({"id": "9", "title": "x"}));
        assert_eq!(p.url, "https://c.example.org/pages/viewpage.action?pageId=9");
    }
}
