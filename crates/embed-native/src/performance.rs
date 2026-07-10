//! CPU worker pool restricted to performance cores on heterogeneous systems.

use crate::{Error, Result};

pub struct PerformanceCorePool {
    inner: rayon::ThreadPool,
}

impl PerformanceCorePool {
    pub fn new(thread_prefix: &'static str) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let cpus = linux_performance_cpus()?;
            let worker_cpus = std::sync::Arc::new(cpus);
            let startup_cpus = worker_cpus.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_cpus.len())
                .thread_name(move |idx| format!("{thread_prefix}-pcore-{idx}"))
                .start_handler(move |_| {
                    let _ = linux_set_current_affinity(&startup_cpus);
                })
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create performance-core pool: {e}")))?;
            if !pool
                .broadcast(|_| linux_set_current_affinity(&worker_cpus))
                .into_iter()
                .all(|configured| configured)
            {
                return Err(Error::Cpu(
                    "cannot restrict CPU workers to Linux performance cores".into(),
                ));
            }
            return Ok(Self { inner: pool });
        }

        #[cfg(target_os = "macos")]
        {
            let threads = macos_performance_cpu_count().ok_or_else(|| {
                Error::Cpu("cannot determine Apple Silicon performance-core count".into())
            })?;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |idx| format!("{thread_prefix}-pcore-{idx}"))
                .start_handler(|_| {
                    let _ = macos_select_performance_qos();
                })
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create performance-core pool: {e}")))?;
            if !pool
                .broadcast(|_| macos_select_performance_qos())
                .into_iter()
                .all(|configured| configured)
            {
                return Err(Error::Cpu(
                    "cannot assign performance QoS to CPU workers".into(),
                ));
            }
            return Ok(Self { inner: pool });
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |idx| format!("{thread_prefix}-cpu-{idx}"))
                .build()
                .map_err(|e| Error::Cpu(format!("cannot create CPU worker pool: {e}")))?;
            Ok(Self { inner: pool })
        }
    }

    pub fn install<OP, R>(&self, operation: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(operation)
    }

    #[cfg(test)]
    fn thread_count(&self) -> usize {
        self.inner.current_num_threads()
    }
}

#[cfg(target_os = "linux")]
fn linux_performance_cpus() -> Result<Vec<usize>> {
    let mut allowed = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) }
        != 0
    {
        return Err(Error::Cpu(
            "cannot read Linux CPU affinity for native inference".into(),
        ));
    }

    let allowed = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &allowed) })
        .collect::<Vec<_>>();
    if allowed.is_empty() {
        return Err(Error::Cpu(
            "Linux CPU affinity contains no processors".into(),
        ));
    }

    let online = parse_linux_cpu_list(
        &std::fs::read_to_string("/sys/devices/system/cpu/online")
            .map_err(|e| Error::Cpu(format!("cannot read Linux online CPU list: {e}")))?,
    )?;
    let scores = linux_performance_scores(&online)?;
    let max_score = scores
        .iter()
        .map(|(_, score)| *score)
        .max()
        .ok_or_else(|| Error::Cpu("Linux online CPU list is empty".into()))?;
    let allowed = allowed
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    let performance = scores
        .into_iter()
        .filter_map(|(cpu, score)| (allowed.contains(&cpu) && score == max_score).then_some(cpu))
        .collect::<Vec<_>>();
    if performance.is_empty() {
        return Err(Error::Cpu(
            "Linux CPU affinity excludes every detected performance core".into(),
        ));
    }
    Ok(performance)
}

#[cfg(target_os = "linux")]
fn linux_performance_scores(cpus: &[usize]) -> Result<Vec<(usize, u64)>> {
    const SCORE_PATHS: &[&str] = &[
        "topology/core_type",
        "cpu_capacity",
        "cpufreq/cpuinfo_max_freq",
    ];
    for relative_path in SCORE_PATHS {
        let scores = cpus
            .iter()
            .filter_map(|&cpu| {
                let path = format!("/sys/devices/system/cpu/cpu{cpu}/{relative_path}");
                std::fs::read_to_string(path)
                    .ok()?
                    .trim()
                    .parse::<u64>()
                    .ok()
                    .map(|score| (cpu, score))
            })
            .collect::<Vec<_>>();
        if scores.len() == cpus.len() {
            return Ok(scores);
        }
    }
    Err(Error::Cpu(
        "cannot classify Linux performance cores from core type, capacity, or maximum frequency"
            .into(),
    ))
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(value: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|_| Error::Cpu(format!("invalid Linux CPU range `{part}`")))?;
            let end = end
                .parse::<usize>()
                .map_err(|_| Error::Cpu(format!("invalid Linux CPU range `{part}`")))?;
            if end < start {
                return Err(Error::Cpu(format!("invalid Linux CPU range `{part}`")));
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(
                part.parse::<usize>()
                    .map_err(|_| Error::Cpu(format!("invalid Linux CPU id `{part}`")))?,
            );
        }
    }
    if cpus.is_empty() {
        return Err(Error::Cpu("Linux CPU list is empty".into()));
    }
    Ok(cpus)
}

#[cfg(target_os = "linux")]
fn linux_set_current_affinity(cpus: &[usize]) -> bool {
    let mut affinity = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_ZERO(&mut affinity) };
    for &cpu in cpus {
        unsafe { libc::CPU_SET(cpu, &mut affinity) };
    }
    unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &affinity) == 0 }
}

#[cfg(target_os = "macos")]
fn macos_performance_cpu_count() -> Option<usize> {
    let mut value = 0i32;
    let mut size = std::mem::size_of_val(&value);
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            (&mut value as *mut i32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value > 0).then_some(value as usize)
}

#[cfg(target_os = "macos")]
fn macos_select_performance_qos() -> bool {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_cpu_lists() {
        assert_eq!(
            parse_linux_cpu_list("0-3,8,10-11\n").expect("parse CPU list"),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
        assert!(parse_linux_cpu_list("3-1").is_err());
        assert!(parse_linux_cpu_list("").is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pool_workers_are_restricted_to_performance_cpus() {
        let expected = linux_performance_cpus().expect("detect performance CPUs");
        let pool = PerformanceCorePool::new("performance-test").expect("create performance pool");
        assert_eq!(pool.thread_count(), expected.len());

        let expected = expected
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let affinities = pool.inner.broadcast(|_| {
            let mut affinity = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
            assert_eq!(
                unsafe {
                    libc::sched_getaffinity(
                        0,
                        std::mem::size_of::<libc::cpu_set_t>(),
                        &mut affinity,
                    )
                },
                0
            );
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &affinity) })
                .collect::<std::collections::HashSet<_>>()
        });
        assert!(affinities.iter().all(|affinity| affinity == &expected));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pool_uses_only_the_performance_cpu_count() {
        let expected = macos_performance_cpu_count().expect("detect performance CPU count");
        let pool = PerformanceCorePool::new("performance-test").expect("create performance pool");
        assert_eq!(pool.thread_count(), expected);
    }
}
