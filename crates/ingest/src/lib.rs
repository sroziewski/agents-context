//! Session → chunks → vectors → Qdrant, incrementally.

pub mod batch;
pub mod chunk;
pub mod indexer;
pub mod scrub;
pub mod state;
pub mod tokens;

pub use chunk::{CHUNKER_VERSION, ChunkConfig};
pub use indexer::{IndexConfig, IndexError, Indexer, Progress, Report};
pub use state::State;
pub use tokens::{HfTokenizer, TokenCounter, WordCounter};

/// Payload fields the sessions collection indexes. `project` is the tenant key: every session
/// query is scoped to one project unless the caller explicitly asks for all of them.
pub const SESSION_FIELDS: &[store::Field] = &[
    store::Field::Tenant("project"),
    store::Field::Keyword("source"),
    store::Field::Keyword("session_id"),
    store::Field::Keyword("kind"),
    store::Field::Bool("subagent"),
];
