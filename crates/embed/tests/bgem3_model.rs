//! Tests against the real bge-m3 model. Ignored by default: the first run downloads ~570 MB.
//! Run with `cargo test -p embed --test bgem3_model -- --ignored --test-threads=1`.

use embed::{Bgem3Embedder, Bgem3Options, Embedder, ModelSource, Provider, cosine};

/// `BGE_M3_DIR` selects a local ONNX export instead of the int8 hub model, and
/// `BGE_M3_PROVIDER` (cpu | directml) the execution provider, so one suite covers every combination.
fn model() -> Bgem3Embedder {
    // fastembed's default cache is relative to the cwd, and cargo runs tests from the crate dir;
    // pin the workspace-level cache so tests reuse the model `bench` already downloaded.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = std::env::var_os("BGE_M3_DIR")
        .map_or(ModelSource::Int8Hub, |d| ModelSource::OnnxDir(root.join(d)));
    let provider = std::env::var("BGE_M3_PROVIDER").map_or(Provider::Cpu, |p| p.parse().unwrap());
    Bgem3Embedder::new(Bgem3Options {
        model,
        provider,
        // Small batches: DirectML on a 4 GB GPU hangs (TDR) on large ones.
        batch_size: 2,
        cache_dir: Some(root.join(".fastembed_cache")),
        show_download_progress: false,
        ..Default::default()
    })
    .expect("load bge-m3")
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn dense_is_1024_dim_unit_vector_and_sparse_is_non_empty() {
    let mut m = model();
    let e = m.embed_one("The database migration failed with a timeout.").unwrap();
    assert_eq!(e.dense.len(), 1024);
    let norm = e.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    assert!(!e.sparse.is_empty());
    assert!(e.sparse.values().iter().all(|&w| w > 0.0));
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn batch_matches_single_embedding() {
    let mut m = model();
    let texts = ["pierwszy tekst", "second text, somewhat longer than the first one", "3"];
    let batch = m.embed(&texts).unwrap();
    assert_eq!(batch.len(), 3);
    for (text, from_batch) in texts.iter().zip(&batch) {
        let single = m.embed_one(text).unwrap();
        // A query is embedded alone and documents in batches; if batch neighbours shift a
        // vector, the two sides drift apart. The int8 export fails this (~0.977).
        let sim = cosine(&single.dense, &from_batch.dense);
        assert!(sim > 0.999, "{text}: cosine {sim}");
    }
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn polish_query_finds_english_document_by_meaning() {
    let mut m = model();
    let query = m.embed_one("Dlaczego migracja bazy danych się nie udała?").unwrap();
    let docs = [
        "The database migration failed due to a timeout on port 5439.",
        "Updated the CSS for the dashboard buttons to use the new palette.",
        "Lunch is at noon in the kitchen on the third floor.",
    ];
    let scores: Vec<f32> =
        m.embed(&docs).unwrap().iter().map(|d| cosine(&query.dense, &d.dense)).collect();
    assert!(scores[0] > scores[1] && scores[0] > scores[2], "scores {scores:?}");
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn english_query_finds_polish_document_by_meaning() {
    let mut m = model();
    let query = m.embed_one("login problems with single sign-on").unwrap();
    let docs = [
        "Użytkownicy nie mogą się zalogować przez SSO od rana.",
        "Faktura za serwery została opłacona w piątek.",
    ];
    let d = m.embed(&docs).unwrap();
    assert!(cosine(&query.dense, &d[0].dense) > cosine(&query.dense, &d[1].dense));
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn sparse_rewards_exact_identifier_overlap() {
    let mut m = model();
    let query = m.embed_one("HD-609").unwrap();
    let with_id = m.embed_one("Fixed the Selenium form bug from ticket HD-609.").unwrap();
    let without = m.embed_one("Fixed the Selenium form bug from ticket HD-712.").unwrap();
    assert!(query.sparse.dot(&with_id.sparse) > query.sparse.dot(&without.sparse));
}

#[test]
#[ignore = "downloads the bge-m3 model"]
fn empty_batch_is_a_no_op() {
    assert!(model().embed(&[]).unwrap().is_empty());
}
