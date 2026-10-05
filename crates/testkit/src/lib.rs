//! Shared test fixtures: a throwaway Qdrant per test via testcontainers.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, TestcontainersError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Same version as docker-compose.yml, so tests exercise the server that runs for real.
pub const QDRANT_TAG: &str = "v1.19.1";
const GRPC: u16 = 6334;
const CONTAINER_ATTEMPTS: usize = 3;
/// Qdrant containers alive at once within one test binary. Too many (a binary's tests all run in
/// parallel) overwhelm Docker's gRPC and indexing calls time out, so the permit is held for the
/// container's whole lifetime, not just its startup — the rest of the tests queue behind it.
const MAX_LIVE_CONTAINERS: usize = 4;

pub struct Qdrant {
    pub url: String,
    _container: ContainerAsync<GenericImage>,
    _permit: OwnedSemaphorePermit,
}

static LIVE: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(MAX_LIVE_CONTAINERS)));

/// Starts Qdrant and returns its gRPC URL. The container is removed, and its concurrency permit
/// released, when `Qdrant` drops.
///
/// Docker Desktop occasionally reports a container as running but never publishes its port
/// (`PortNotExposed` indefinitely, even with no load). Waiting longer does not help, so such a
/// container is discarded and a fresh one started.
pub async fn qdrant() -> Qdrant {
    let permit = LIVE.clone().acquire_owned().await.expect("semaphore never closed");
    let mut last = None;
    for _ in 0..CONTAINER_ATTEMPTS {
        let container = GenericImage::new("qdrant/qdrant", QDRANT_TAG)
            .with_exposed_port(ContainerPort::Tcp(GRPC))
            .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on 6334"))
            .start()
            .await
            .expect("start qdrant container (is Docker running?)");
        match host_port(&container).await {
            Ok(port) => return Qdrant { url: format!("http://127.0.0.1:{port}"), _container: container, _permit: permit },
            Err(e) => last = Some(e), // dropping `container` removes it
        }
    }
    panic!("qdrant port never published in {CONTAINER_ATTEMPTS} containers: {last:?}");
}

/// Polls for the published port for up to ~10 s.
async fn host_port(container: &ContainerAsync<GenericImage>) -> Result<u16, TestcontainersError> {
    let mut last = None;
    for attempt in 0..20u64 {
        match container.get_host_port_ipv4(GRPC).await {
            Ok(port) => return Ok(port),
            Err(e @ TestcontainersError::PortNotExposed { .. }) => last = Some(e),
            Err(e) => panic!("qdrant port lookup failed: {e}"),
        }
        tokio::time::sleep(Duration::from_millis(100 * (attempt + 1).min(10))).await;
    }
    Err(last.expect("loop ran"))
}
