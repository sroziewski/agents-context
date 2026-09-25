use std::path::{Path, PathBuf};
use std::str::FromStr;

use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model, InitOptionsUserDefined, TokenizerFiles};

use crate::{EmbedError, Embedder, Embedding, Result, SparseVector};

pub const BGEM3_DIM: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Cpu,
    /// Any DirectX 12 GPU on Windows (the GTX 1650 here). Requires the `directml` feature.
    DirectMl,
}

impl FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "directml" | "dml" | "gpu" => Ok(Self::DirectMl),
            other => Err(format!("unknown provider {other:?}; expected cpu or directml")),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ModelSource {
    /// `gpahal/bge-m3-onnx-int8` (570 MB) fetched from Hugging Face. Fast on CPU; DirectML handles
    /// its int8 kernels poorly.
    #[default]
    Int8Hub,
    /// A local directory with a bge-m3 export emitting `[dense, sparse, colbert]` in that order
    /// (e.g. `aapot/bge-m3-onnx`, fp32): `model.onnx` plus its external data and the tokenizer files.
    OnnxDir(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Bgem3Options {
    pub model: ModelSource,
    pub provider: Provider,
    /// Longer inputs are truncated silently by the tokenizer, so chunkers must stay below this.
    pub max_length: usize,
    pub batch_size: usize,
    pub cache_dir: Option<PathBuf>,
    pub intra_threads: Option<usize>,
    pub show_download_progress: bool,
}

impl Default for Bgem3Options {
    fn default() -> Self {
        Self {
            model: ModelSource::Int8Hub,
            provider: Provider::Cpu,
            max_length: 512,
            batch_size: 16,
            cache_dir: None,
            intra_threads: None,
            show_download_progress: true,
        }
    }
}

/// bge-m3 via fastembed's joint model: dense and sparse come out of one forward pass
/// (`gpahal/bge-m3-onnx-int8`), so the model is loaded once. ColBERT output is discarded.
pub struct Bgem3Embedder {
    model: Bgem3Embedding,
    batch_size: usize,
}

impl Bgem3Embedder {
    pub fn new(opts: Bgem3Options) -> Result<Self> {
        let model = match &opts.model {
            ModelSource::Int8Hub => Self::load_hub(&opts)?,
            ModelSource::OnnxDir(dir) => Self::load_dir(dir, &opts)?,
        };
        Ok(Self { model, batch_size: opts.batch_size.max(1) })
    }

    fn load_dir(dir: &Path, opts: &Bgem3Options) -> Result<Bgem3Embedding> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| EmbedError::Model(format!("{}: {e}", dir.join(name).display())))
        };
        let tokenizer = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let mut init = InitOptionsUserDefined::new()
            .with_max_length(opts.max_length)
            .with_execution_providers(vec![execution_provider(opts.provider)?]);
        if let Some(n) = opts.intra_threads {
            init = init.with_intra_threads(n);
        }
        // From a path, not bytes, so ONNX Runtime resolves model.onnx.data next to it.
        Bgem3Embedding::try_new_from_path(dir.join("model.onnx"), tokenizer, init)
            .map_err(|e| EmbedError::Model(e.to_string()))
    }

    fn load_hub(opts: &Bgem3Options) -> Result<Bgem3Embedding> {
        let mut init = Bgem3InitOptions::new(Bgem3Model::BGEM3Q)
            .with_max_length(opts.max_length)
            .with_show_download_progress(opts.show_download_progress)
            .with_execution_providers(vec![execution_provider(opts.provider)?]);
        if let Some(dir) = &opts.cache_dir {
            init = init.with_cache_dir(dir.clone());
        }
        if let Some(n) = opts.intra_threads {
            init = init.with_intra_threads(n);
        }
        Bgem3Embedding::try_new(init).map_err(|e| EmbedError::Model(e.to_string()))
    }
}

fn execution_provider(p: Provider) -> Result<fastembed::ExecutionProviderDispatch> {
    // error_on_failure: ort's default is to fall back to CPU silently, which would make a
    // "GPU" benchmark quietly measure the CPU.
    match p {
        Provider::Cpu => Ok(ort::ep::CPU::default().build().error_on_failure()),
        #[cfg(feature = "directml")]
        Provider::DirectMl => Ok(ort::ep::DirectML::default().build().error_on_failure()),
        #[cfg(not(feature = "directml"))]
        Provider::DirectMl => Err(EmbedError::ProviderUnavailable("DirectML", "directml")),
    }
}

impl Embedder for Bgem3Embedder {
    fn dim(&self) -> usize {
        BGEM3_DIM
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let out = self
            .model
            .embed(texts, Some(self.batch_size))
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        if out.dense.len() != texts.len() || out.sparse.len() != texts.len() {
            return Err(EmbedError::CountMismatch {
                expected: texts.len(),
                got: out.dense.len().min(out.sparse.len()),
            });
        }
        out.dense
            .into_iter()
            .zip(out.sparse)
            .map(|(dense, sparse)| {
                Ok(Embedding { dense: l2_normalize(dense), sparse: SparseVector::from_fastembed(sparse)? })
            })
            .collect()
    }
}

// Normalised here rather than trusting the ONNX export, so dot-product scores equal cosine
// regardless of which export or distance metric is used downstream.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_parses_aliases_case_insensitively() {
        assert_eq!("CPU".parse::<Provider>().unwrap(), Provider::Cpu);
        for alias in ["directml", "DML", "gpu"] {
            assert_eq!(alias.parse::<Provider>().unwrap(), Provider::DirectMl);
        }
        assert!("cuda".parse::<Provider>().is_err());
    }

    #[test]
    fn default_options_match_chunker_budget() {
        let o = Bgem3Options::default();
        assert_eq!(o.provider, Provider::Cpu);
        assert_eq!(o.max_length, 512);
        assert!(o.batch_size > 0);
    }

    #[test]
    fn l2_normalize_produces_unit_vectors_and_keeps_zero() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(l2_normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[cfg(not(feature = "directml"))]
    #[test]
    fn directml_without_feature_is_a_clear_error() {
        let err = execution_provider(Provider::DirectMl).unwrap_err();
        assert!(err.to_string().contains("--features directml"), "{err}");
    }

    #[test]
    fn missing_model_dir_names_the_missing_file() {
        let opts = Bgem3Options {
            model: ModelSource::OnnxDir(PathBuf::from("does/not/exist")),
            ..Default::default()
        };
        let err = Bgem3Embedder::new(opts).err().expect("must fail");
        assert!(err.to_string().contains("tokenizer.json"), "{err}");
    }

    #[test]
    fn cpu_provider_is_always_available() {
        assert!(execution_provider(Provider::Cpu).is_ok());
    }
}
