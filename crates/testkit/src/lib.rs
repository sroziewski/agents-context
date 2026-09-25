//! Shared test fixtures: a throwaway Qdrant per test via testcontainers.

use std::time::Duration;

use testcontainers::core::{ContainerPort, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, TestcontainersError};

/// Same version as docker-compose.yml, so tests exercise the server that runs for real.
pub const QDRANT_TAG: &str = "v1.19.1";
const GRPC: u16 = 6334;

pub struct Qdrant {
    pub url: String,
    _container: ContainerAsync<GenericImage>,
}

/// Starts Qdrant and returns its gRPC URL. The container is removed when `Qdrant` drops.
pub async fn qdrant() -> Qdrant {
    let container = GenericImage::new("qdrant/qdrant", QDRANT_TAG)
        .with_exposed_port(ContainerPort::Tcp(GRPC))
        .with_wait_for(WaitFor::message_on_stdout("Qdrant gRPC listening on 6334"))
        .start()
        .await
        .expect("start qdrant container (is Docker running?)");
    let port = host_port(&container).await;
    Qdrant { url: format!("http://127.0.0.1:{port}"), _container: container }
}

/// Docker Desktop can report a running container before its port mapping is published when
/// many containers start at once (seen as `PortNotExposed` in parallel test runs), so the lookup
/// is retried briefly instead of failing the test.
async fn host_port(container: &ContainerAsync<GenericImage>) -> u16 {
    let mut last = None;
    for attempt in 0..20 {
        match container.get_host_port_ipv4(GRPC).await {
            Ok(port) => return port,
            Err(e @ TestcontainersError::PortNotExposed { .. }) => last = Some(e),
            Err(e) => panic!("qdrant port lookup failed: {e}"),
        }
        tokio::time::sleep(Duration::from_millis(100 * (attempt + 1))).await;
    }
    panic!("qdrant port never published: {last:?}");
}
