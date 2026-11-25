//! The cost of the loads and stores of the parrot JIT and interpreter, with the unaligned and the
//! aligned memory mapping.
//!
//! Each benchmark is a loop of 8 accesses per iteration, timed against the same loop with a `mov`
//! in place of each access. Run with `cargo run --release --example memory_access_bench`.

use solana_sbpf::{
    assembler::assemble,
    ebpf,
    elf::Executable,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, SBPFVersion},
    test_utils::TestContextObject,
    verifier::RequisiteVerifier,
    vm::{Config, EbpfVm, ExecutionMode},
};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ACCESSES_PER_ITERATION: u64 = 8;

/// A loop of `iterations`, each doing 8 of `access` (with `{}` replaced by an offset). `access`
/// can have alternatives separated by `|`, which are taken in turns.
fn looped(access: &str, iterations: u64) -> String {
    let alternatives: Vec<&str> = access.split('|').collect();
    let mut body = String::new();
    for i in 0..ACCESSES_PER_ITERATION {
        let access = alternatives[i as usize % alternatives.len()];
        body.push_str(&access.replace("{}", &((i + 1) * 8).to_string()));
        body.push('\n');
    }
    format!(
        "
        mov64 r6, {iterations}
        loop:
        {body}
        sub64 r6, 1
        jne r6, 0, loop
        exit"
    )
}

/// SBPFv4 onwards always has the aligned mapping, so v3 is the latest to compare them in.
fn executable(source: &str, aligned_memory_mapping: bool) -> Executable<TestContextObject> {
    let config = Config {
        aligned_memory_mapping,
        enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V3,
        ..Config::default()
    };
    let executable =
        assemble::<TestContextObject>(source, Arc::new(BuiltinProgram::new_loader(config)))
            .unwrap();
    assert_eq!(executable.get_sbpf_version(), SBPFVersion::V3);
    executable.verify::<RequisiteVerifier>().unwrap();
    executable.parrot_compile().unwrap();
    executable
}

/// The median time of `runs` executions of `executable` in `mode`.
fn run(executable: &Executable<TestContextObject>, interpret: bool, runs: usize) -> Duration {
    let config = executable.get_config();
    let mut stack = vec![0u8; config.stack_size()];
    let mut heap = vec![0u8; 32 * 1024];
    let mut input = vec![0u8; 64 * 1024];
    // For chasing pointers: the first word of the input points to itself.
    input[..8].copy_from_slice(&ebpf::MM_INPUT_START.to_le_bytes());
    let regions = vec![
        executable.get_ro_region(),
        MemoryRegion::new(&raw mut stack[..], ebpf::MM_STACK_START),
        MemoryRegion::new(&raw mut heap[..], ebpf::MM_HEAP_START),
        MemoryRegion::new(&raw mut input[..], ebpf::MM_INPUT_START),
    ];
    let mut context = TestContextObject::new(0);
    // SAFETY: The memory of the regions outlives the mapping.
    context.memory_mapping =
        unsafe { MemoryMapping::new(regions, config, executable.get_sbpf_version()) }.unwrap();
    let mut vm = EbpfVm::new(
        executable.get_loader().clone(),
        executable.get_sbpf_version(),
        &mut context,
        config.stack_size(),
    );
    let registers = vm.registers;
    let mut call_frames = vec![Default::default(); config.max_call_depth];
    let mut times: Vec<Duration> = (0..runs)
        .map(|_| {
            vm.registers = registers;
            vm.registers[1] = ebpf::MM_INPUT_START;
            vm.context().remaining = 1 << 40;
            let start = Instant::now();
            let mut mode = if interpret {
                ExecutionMode::ParrotInterpreted
            } else {
                ExecutionMode::Jit
            };
            let (_, result) = vm.execute_program(executable, &mut mode, &mut call_frames);
            let elapsed = start.elapsed();
            assert!(result.is_ok(), "{:?}", result);
            elapsed
        })
        .collect();
    times.sort();
    times[times.len() / 2]
}

fn main() {
    const ITERATIONS: u64 = 1_000_000;
    const RUNS: usize = 21;
    let benchmarks = [
        ("ldxdw", "ldxdw r2, [r1+{}]", "mov64 r2, r1"),
        ("ldxb", "ldxb r2, [r1+{}]", "mov64 r2, r1"),
        ("stxdw", "stxdw [r1+{}], r6", "mov64 r2, r6"),
        ("stdw imm", "stdw [r1+{}], 1", "mov64 r2, 1"),
        ("ldxdw stack", "ldxdw r2, [r10-{}]", "mov64 r2, r10"),
        (
            "ldxdw 2 regions",
            "ldxdw r2, [r1+{}]|ldxdw r3, [r10-{}]",
            "mov64 r2, r1|mov64 r3, r10",
        ),
        // Each load depends on the previous one: the latency rather than the throughput.
        ("ldxdw chain", "ldxdw r1, [r1+0]", "add64 r1, 0"),
    ];
    let modes = [("jit", false), ("interpreter", true)];
    // Only the benchmarks whose `mode mapping name` contains the first argument, if any.
    let filter = std::env::args().nth(1).unwrap_or_default();
    for (mode_name, mode) in modes {
        for (name, access, baseline) in benchmarks {
            for (mapping, aligned) in [("unaligned", false), ("aligned", true)] {
                if !format!("{mode_name} {mapping} {name}").contains(&filter) {
                    continue;
                }
                let accessing = run(
                    &executable(&looped(access, ITERATIONS), aligned),
                    mode,
                    RUNS,
                );
                let not_accessing = run(
                    &executable(&looped(baseline, ITERATIONS), aligned),
                    mode,
                    RUNS,
                );
                let accesses = ITERATIONS * ACCESSES_PER_ITERATION;
                println!(
                    "{mode_name:<11} {mapping:<9} {name:<15} {:>7.3} ns/access ({:.2?} with the \
                     accesses, {:.2?} with a mov instead)",
                    (accessing.as_secs_f64() - not_accessing.as_secs_f64()) * 1e9 / accesses as f64,
                    accessing,
                    not_accessing,
                );
            }
        }
    }
}
