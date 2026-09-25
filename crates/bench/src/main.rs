//! Spike benchmark: how fast and how heavy is bge-m3 on this machine, CPU vs DirectML?
//!
//! `cargo run -p bench --release -- --provider cpu`
//! `cargo run -p bench --release --features directml -- --provider directml`

mod corpus;
mod stats;

use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use embed::{Bgem3Embedder, Bgem3Options, Embedder, ModelSource, Provider, cosine};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "cpu")]
    provider: Provider,
    /// Texts per length class (short, medium, long).
    #[arg(long, default_value_t = 64)]
    per_class: usize,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Local ONNX export directory instead of the int8 model from Hugging Face.
    #[arg(long)]
    model_dir: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut mem = stats::Memory::new();
    println!(
        "provider={:?} batch={} max_length={} model={}",
        args.provider,
        args.batch,
        args.max_length,
        args.model_dir.as_deref().map_or("int8 (hub)".into(), |p| p.display().to_string())
    );
    println!("rss before load: {}", stats::mib(mem.rss()));

    let t = Instant::now();
    let mut model = Bgem3Embedder::new(Bgem3Options {
        model: args.model_dir.map_or(ModelSource::Int8Hub, ModelSource::OnnxDir),
        provider: args.provider,
        batch_size: args.batch,
        max_length: args.max_length,
        ..Default::default()
    })
    .context("load bge-m3")?;
    println!("load: {:.2?}, rss after load: {}", t.elapsed(), stats::mib(mem.rss()));

    // The first run pays for graph optimisation and (on DirectML) shader compilation.
    let t = Instant::now();
    model.embed(&["warm-up"])?;
    println!("warm-up: {:.2?}", t.elapsed());

    for class in corpus::LengthClass::ALL {
        let texts = corpus::texts(class, args.per_class);
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let t = Instant::now();
        let out = model.embed(&refs)?;
        let r = stats::Throughput::new(out.len(), t.elapsed());
        println!(
            "{:<6} ~{:>3} words: {:>7.1} texts/s  ({:.2?} for {})",
            class.name(),
            class.words(),
            r.per_second(),
            r.elapsed,
            r.count
        );
    }
    println!("peak rss: {}", stats::mib(mem.peak()));

    println!("\ncross-lingual sanity (higher = closer):");
    for (a, b) in corpus::PAIRS {
        let e = model.embed(&[a, b])?;
        println!("  {:.3}  {a}  <->  {b}", cosine(&e[0].dense, &e[1].dense));
    }
    Ok(())
}
