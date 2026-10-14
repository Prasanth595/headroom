//! Magika must not leave an ONNX Runtime intra-op thread pool spinning (#4187).
//!
//! ORT's pool threads spin-wait between inferences, so a pool sized to the
//! core count pegs idle CPUs in a long-lived proxy. This checks that symptom
//! directly: under light steady load and the idle gap after it, no thread
//! other than the caller may burn CPU. Counting threads instead (this test's
//! first version) also flagged threads that never spin, such as the Apple
//! Neural Engine service threads ORT 1.31 brings up on macOS (#4191).
//!
//! The magika session is a process-global singleton, so this file holds
//! exactly ONE `#[test]` fn and must run in its own process.
#![cfg(all(feature = "ml", any(target_os = "linux", target_os = "macos")))]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use headroom_core::transforms::magika_detector::{magika_detect, MagikaDetectorError};

const SAMPLE: &str = "fn main() { println!(\"hi\"); }";

/// Light steady load: one detection every `LOAD_GAP` for `LOAD`, the pattern
/// that kept the pool spinning in the proxy.
const LOAD: Duration = Duration::from_millis(1000);
const LOAD_GAP: Duration = Duration::from_millis(20);
/// Idle time after the load, inside the measured window.
const IDLE: Duration = Duration::from_millis(500);
/// Most CPU any other thread may use over the whole window.
const MAX_OTHER_THREAD_CPU: Duration = Duration::from_millis(100);

/// CPU time (user + system) used so far by each live thread of this process,
/// keyed by kernel thread id.
#[cfg(target_os = "linux")]
fn thread_cpu_times() -> HashMap<u64, Duration> {
    // USER_HZ, the unit of utime/stime in /proc, is fixed at 100 for the
    // userspace ABI on every architecture Linux supports.
    const TICKS_PER_SEC: u64 = 100;
    let mut out = HashMap::new();
    for entry in std::fs::read_dir("/proc/self/task").expect("read /proc/self/task") {
        let entry = entry.expect("task entry");
        let Ok(tid) = entry.file_name().to_string_lossy().parse::<u64>() else {
            continue;
        };
        // The thread exited between read_dir and here.
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // Field 2 (comm) can contain spaces, so split after its closing
        // paren; utime and stime are fields 14 and 15.
        let rest = &stat[stat.rfind(')').expect("comm field") + 2..];
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let ticks: u64 =
            fields[11].parse::<u64>().expect("utime") + fields[12].parse::<u64>().expect("stime");
        out.insert(tid, Duration::from_millis(ticks * 1000 / TICKS_PER_SEC));
    }
    out
}

#[cfg(target_os = "linux")]
fn current_thread_id() -> u64 {
    // "/proc/thread-self" links to "<pid>/task/<tid>".
    std::fs::read_link("/proc/thread-self")
        .expect("read /proc/thread-self")
        .file_name()
        .and_then(|tid| tid.to_str()?.parse().ok())
        .expect("tid")
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc's Mach bindings point at the mach2 crate; these are stable.
fn thread_cpu_times() -> HashMap<u64, Duration> {
    extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let seconds = |t: libc::time_value_t| {
        Duration::from_secs(t.seconds as u64) + Duration::from_micros(t.microseconds as u64)
    };
    let mut out = HashMap::new();
    // SAFETY: task_threads hands back `count` thread send rights in a buffer
    // the kernel allocated. Each right is released after it is read, and the
    // buffer is returned with vm_deallocate.
    unsafe {
        let task = libc::mach_task_self();
        let mut threads: libc::thread_act_array_t = std::ptr::null_mut();
        let mut count: libc::mach_msg_type_number_t = 0;
        assert_eq!(
            libc::task_threads(task, &mut threads, &mut count),
            libc::KERN_SUCCESS,
            "task_threads"
        );
        for &thread in std::slice::from_raw_parts(threads, count as usize) {
            let mut basic: libc::thread_basic_info = std::mem::zeroed();
            let mut basic_count = libc::THREAD_BASIC_INFO_COUNT;
            let mut ident: libc::thread_identifier_info = std::mem::zeroed();
            let mut ident_count = libc::THREAD_IDENTIFIER_INFO_COUNT;
            // Either call fails only if the thread exited in between.
            let read = libc::thread_info(
                thread,
                libc::THREAD_BASIC_INFO as libc::thread_flavor_t,
                &mut basic as *mut _ as libc::thread_info_t,
                &mut basic_count,
            ) == libc::KERN_SUCCESS
                && libc::thread_info(
                    thread,
                    libc::THREAD_IDENTIFIER_INFO as libc::thread_flavor_t,
                    &mut ident as *mut _ as libc::thread_info_t,
                    &mut ident_count,
                ) == libc::KERN_SUCCESS;
            if read {
                out.insert(
                    ident.thread_id,
                    seconds(basic.user_time) + seconds(basic.system_time),
                );
            }
            mach_port_deallocate(task, thread);
        }
        libc::vm_deallocate(
            task,
            threads as libc::vm_address_t,
            count as usize * std::mem::size_of::<libc::thread_act_t>(),
        );
    }
    out
}

#[cfg(target_os = "macos")]
fn current_thread_id() -> u64 {
    let mut id = 0;
    // SAFETY: a null thread means the calling thread; `id` outlives the call.
    let rc = unsafe { libc::pthread_threadid_np(0, &mut id) };
    assert_eq!(rc, 0, "pthread_threadid_np");
    id
}

#[test]
fn magika_session_does_not_spin_idle_threads() {
    // The first call builds the session; it is not part of the window.
    match magika_detect(SAMPLE) {
        Ok(_) => {}
        // With an explicit ORT_DYLIB_PATH (as in CI) the runtime is present,
        // so an init failure is a real session-build regression, not a skip.
        Err(MagikaDetectorError::Init(reason))
            if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.trim().is_empty()) =>
        {
            eprintln!("skipping: magika runtime unavailable on this host: {reason}");
            return;
        }
        Err(e) => panic!("magika detection failed: {e}"),
    }

    let caller = current_thread_id();
    let before = thread_cpu_times();
    let start = Instant::now();
    while start.elapsed() < LOAD {
        magika_detect(SAMPLE).expect("magika detection");
        std::thread::sleep(LOAD_GAP);
    }
    std::thread::sleep(IDLE);
    let window = start.elapsed();
    let after = thread_cpu_times();

    let mut others: Vec<(u64, Duration)> = after
        .iter()
        .filter(|(tid, _)| **tid != caller)
        .map(|(tid, cpu)| {
            (
                *tid,
                cpu.saturating_sub(*before.get(tid).unwrap_or(&Duration::ZERO)),
            )
        })
        .collect();
    others.sort_by_key(|(_, cpu)| std::cmp::Reverse(*cpu));
    let caller_cpu = after[&caller].saturating_sub(before[&caller]);
    eprintln!(
        "window {window:?}: caller used {caller_cpu:?}; other threads (tid, cpu): {others:?}"
    );

    let (busiest, cpu) = others.first().copied().unwrap_or((0, Duration::ZERO));
    assert!(
        cpu <= MAX_OTHER_THREAD_CPU,
        "thread {busiest} used {cpu:?} of CPU over a {window:?} window of light load and idle, \
         over the {MAX_OTHER_THREAD_CPU:?} allowance: an ORT thread pool is spinning (#4187). \
         Other threads (tid, cpu): {others:?}"
    );
}
