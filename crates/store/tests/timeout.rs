//! The request timeout `Store::connect` sets, against a server that accepts connections and never
//! answers. Takes about two minutes by design: that wait is the behaviour under test.

use std::time::{Duration, Instant};

use store::{REQUEST_TIMEOUT, SearchFilter, Store};

#[tokio::test(flavor = "multi_thread")]
async fn a_request_to_a_silent_server_fails_after_two_60_second_timeouts() {
    assert_eq!(REQUEST_TIMEOUT, Duration::from_secs(60));
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        // Hold every connection open without reading or writing.
        let mut held = Vec::new();
        for conn in listener.incoming() {
            held.push(conn);
        }
    });
    let store = Store::connect(&url, None, "silent", 64).unwrap();

    let started = Instant::now();
    let r = tokio::time::timeout(Duration::from_secs(200), store.count(&SearchFilter::default())).await;
    let took = started.elapsed();

    let r = r.unwrap_or_else(|_| panic!("no timeout at all: still waiting after {took:?}"));
    assert!(r.is_err(), "a silent server cannot answer: {r:?}");
    // qdrant-client 1.19 retries a timed-out request once on a fresh channel
    // (channel_pool.rs `with_channel`, allow_retry), so the call takes two timeouts: ~120 s here,
    // ~10 s with the client's 5 s default.
    assert!(took >= Duration::from_secs(110) && took < Duration::from_secs(150), "failed after {took:?}");
}
