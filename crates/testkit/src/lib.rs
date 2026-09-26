//! Shared test fixtures: a throwaway Qdrant per test via testcontainers.

use std::time::Duration;

use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, TestcontainersError};

/// Same version as docker-compose.yml, so tests exercise the server that runs for real.
pub const QDRANT_TAG: &str = "v1.19.1";
const GRPC: u16 = 6334;
const CONTAINER_ATTEMPTS: usize = 3;

pub struct Qdrant {
    pub url: String,
    _container: ContainerAsync<GenericImage>,
}

/// Container startups in flight at once within one test binary. Docker Desktop publishes port
/// mappings slowly when a dozen containers start in the same second, which made port lookups
/// time out; queueing the startup phase (not the containers' lifetimes) keeps it fast.
static STARTUPS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);

/// Starts Qdrant and returns its gRPC URL. The container is removed when `Qdrant` drops.
///
/// Docker Desktop occasionally reports a container as running but never publishes its port
/// (`PortNotExposed` indefinitely, even with no load). Waiting longer does not help, so such a
/// container is discarded and a fresh one started.
pub async fn qdrant() -> Qdrant {
    let _permit = STARTUPS.acquire().await.expect("semaphore never closed");
    let mut last = None;
    for _ in 0..CONTAINER_ATTEMPTS {
        let container = GenericImage::new("qdrant/qdrant", QDRANT_TAG)
            .with_exposed_port(ContainerPort::Tcp(GRPC))
            .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on 6334"))
            .start()
            .await
            .expect("start qdrant container (is Docker running?)");
        match host_port(&container).await {
            Ok(port) => return Qdrant { url: format!("http://127.0.0.1:{port}"), _container: container },
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
