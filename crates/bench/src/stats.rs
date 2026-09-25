use std::time::Duration;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

pub struct Throughput {
    pub count: usize,
    pub elapsed: Duration,
}

impl Throughput {
    pub fn new(count: usize, elapsed: Duration) -> Self {
        Self { count, elapsed }
    }

    pub fn per_second(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 { 0.0 } else { self.count as f64 / secs }
    }
}

/// Resident memory of this process. Peak is the max of the samples taken, not an OS
/// high-water mark, so it is sampled after each phase.
pub struct Memory {
    sys: System,
    pid: Pid,
    peak: u64,
}

impl Memory {
    pub fn new() -> Self {
        Self { sys: System::new(), pid: Pid::from_u32(std::process::id()), peak: 0 }
    }

    pub fn rss(&mut self) -> u64 {
        self.sys
            .refresh_processes_specifics(ProcessesToUpdate::Some(&[self.pid]), true, ProcessRefreshKind::nothing().with_memory());
        let rss = self.sys.process(self.pid).map_or(0, |p| p.memory());
        self.peak = self.peak.max(rss);
        rss
    }

    pub fn peak(&mut self) -> u64 {
        self.rss();
        self.peak
    }
}

pub fn mib(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_divides_count_by_seconds() {
        assert_eq!(Throughput::new(50, Duration::from_millis(500)).per_second(), 100.0);
    }

    #[test]
    fn throughput_of_zero_duration_is_zero_not_infinite() {
        assert_eq!(Throughput::new(5, Duration::ZERO).per_second(), 0.0);
    }

    #[test]
    fn mib_formats_whole_mebibytes() {
        assert_eq!(mib(3 * 1024 * 1024), "3 MiB");
        assert_eq!(mib(0), "0 MiB");
    }

    #[test]
    fn memory_reports_non_zero_rss_and_monotonic_peak() {
        let mut m = Memory::new();
        let first = m.rss();
        assert!(first > 0);
        assert!(m.peak() >= first);
    }
}
