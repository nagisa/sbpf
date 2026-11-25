//! The instruction-fetch behaviour of the parrot interpreter's step table against the legacy
//! interpreter.
//!
//! Each loop executes a body of `N` distinct interpreter steps. `same-set` picks steps that the
//! current step layout puts in the same L1i set (one opcode, `N` register pairs); `spread` picks
//! steps round robin over 8 opcodes whose steps land in 8 different sets, and `spread-blocked`
//! the same steps grouped by opcode, so that consecutive ones share the set. `jcc` is like
//! `same-set` with a conditional jump to the next instruction in place of `add64`. Every configuration is
//! timed, counted with the core counters, and sampled with IBS fetch to attribute instruction
//! cache, op cache and iTLB misses to the fetches of the step table.
//!
//! Needs the `codegen-debug` feature, for the symbols of the steps. Linux on AMD Zen only. `ibs_fetch` is a dynamic PMU; pass its type in `IBS_FETCH_TYPE` (see
//! `/sys/bus/event_source/devices/ibs_fetch/type`, defaults to 10).
//!
//! Run with `cargo run --release --features codegen-debug --example l1i_probe [same-set|spread] [N...]`.

use solana_sbpf::{
    assembler::assemble,
    codegen::debug::{self, Symbol},
    elf::Executable,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, SBPFVersion},
    test_utils::TestContextObject,
    verifier::RequisiteVerifier,
    vm::{Config, EbpfVm, ExecutionMode},
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// The destination registers, all but r6, the loop counter, and r10.
const DSTS: [u8; 9] = [0, 1, 2, 3, 4, 5, 7, 8, 9];

/// The opcodes of `spread`, with distinct low 5 bits, so distinct L1i sets in the current layout.
/// `{d}` and `{s}` are replaced by the registers.
const SPREAD: [&str; 8] = [
    "add64 r{d}, r{s}", // 0x0f
    "add64 r{d}, 7",    // 0x07
    "add32 r{d}, r{s}", // 0x0c
    "add32 r{d}, 7",    // 0x04
    "sub64 r{d}, r{s}", // 0x1f
    "sub64 r{d}, 7",    // 0x17
    "sub32 r{d}, r{s}", // 0x1c
    "sub32 r{d}, 7",    // 0x14
];

/// The `n` distinct steps of the `kind` of loop body.
fn body(kind: &str, n: usize) -> Vec<String> {
    // The (dst, src) pairs, interleaved so that consecutive pairs differ in both.
    let pairs = (0..DSTS.len() * 11).map(|k| (DSTS[k % 9], ((k % 9) + k / 9) % 11));
    let mut steps: Vec<String> = Vec::new();
    match kind {
        "same-set" => steps.extend(pairs.map(|(d, s)| format!("add64 r{d}, r{s}"))),
        // Stores of immediates of every size, through 6 registers that point into the stack.
        "store-imm" => {
            let regs = [3, 4, 5, 7, 8, 9];
            let mut body: Vec<String> = regs.iter().map(|r| format!("mov64 r{r}, r10")).collect();
            for (i, r) in regs.iter().enumerate() {
                for (j, size) in ["stb", "sth", "stw", "stdw"].iter().enumerate() {
                    body.push(format!("{size} [r{r}-{}], 1", 8 * (1 + i * 4 + j)));
                }
            }
            return body;
        }
        // `n` distinct instructions picked at random from a variety of ALU instructions and
        // conditional jumps to the next instruction, with a branch over the next instruction every
        // 6 of them, on a pseudo-random bit (`mix`) or on a bit of the loop counter (`mix-pred`).
        "mix" | "mix-pred" => {
            use rand::{rngs::SmallRng, Rng, SeedableRng};
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let dsts = [0, 3, 4, 5, 7, 8, 9];
            let alu = ["add", "sub", "or", "and", "xor", "mov"];
            let shifts = ["lsh", "rsh", "arsh"];
            let jumps = [
                "jeq", "jne", "jgt", "jge", "jlt", "jle", "jsgt", "jslt", "jset",
            ];
            let mut body: Vec<String> = [
                "mov64 r2, r1",
                "lsh64 r2, 13",
                "xor64 r1, r2",
                "mov64 r2, r1",
                "rsh64 r2, 7",
                "xor64 r1, r2",
                "mov64 r2, r1",
                "lsh64 r2, 17",
                "xor64 r1, r2",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let mut seen = std::collections::HashSet::new();
            let mut picked = 0;
            while picked < n {
                let d = dsts[rng.gen_range(0..dsts.len())];
                let src = rng.gen_range(0..=10);
                let width = if rng.gen_bool(0.5) { "64" } else { "32" };
                let insn = match rng.gen_range(0..10) {
                    0..=4 => format!("{}{width} r{d}, r{src}", alu[rng.gen_range(0..alu.len())]),
                    5..=6 => format!(
                        "{}{width} r{d}, {}",
                        alu[rng.gen_range(0..alu.len())],
                        rng.gen_range(1..1000)
                    ),
                    7 => format!(
                        "{}{width} r{d}, {}",
                        shifts[rng.gen_range(0..shifts.len())],
                        rng.gen_range(1..31)
                    ),
                    _ => format!("{} r{d}, r{src}, +0", jumps[rng.gen_range(0..jumps.len())]),
                };
                // Distinct steps: the immediates do not matter.
                let step = insn.rsplit_once(", ").map_or(insn.clone(), |(head, tail)| {
                    if tail.starts_with('r') {
                        insn.clone()
                    } else {
                        head.to_string()
                    }
                });
                if !seen.insert(step) {
                    continue;
                }
                body.push(insn);
                picked += 1;
                if picked % 6 == 0 {
                    let bit = 1u32 << (picked / 6 % 24);
                    let reg = if kind == "mix" { "r1" } else { "r6" };
                    body.push(format!("jset {reg}, {bit}, +1"));
                }
            }
            return body;
        }
        // Like `rand` and `alt`, with the branch going to add64 of an immediate to r3 (opcode
        // 0x0307) or to r5 (0x0507), after 42 add64 whose dispatch table entries are in the same
        // L1D sets as those of these two (the sets are picked by bits 4 to 9 of the opcode), more
        // than the sets hold. The entries of the two are not in the L1D when the branch dispatches
        // to them.
        "rand-cold" | "alt-cold" => {
            let bit = if kind == "rand-cold" { "r1" } else { "r6" };
            let mut body = Vec::new();
            for (dst, from) in [(7, 0), (3, 1), (9, 0), (5, 1)] {
                body.extend((from..=10).map(|src| format!("add64 r{dst}, r{src}")));
            }
            body.extend(
                [
                    "mov64 r2, r1",
                    "lsh64 r2, 13",
                    "xor64 r1, r2",
                    "mov64 r2, r1",
                    "rsh64 r2, 7",
                    "xor64 r1, r2",
                    "mov64 r2, r1",
                    "lsh64 r2, 17",
                    "xor64 r1, r2",
                ]
                .iter()
                .map(|s| s.to_string()),
            );
            body.push(format!("jset {bit}, 1, +1"));
            body.push("add64 r3, 1".to_string());
            body.push("add64 r5, 1".to_string());
            return body;
        }
        // A branch on a pseudo-random bit (`rand`), or on the alternating low bit of the loop
        // counter (`alt`), followed by `n` distinct add64 that leave the state of the xorshift in
        // r1 and its temporary in r2 alone.
        "rand" | "alt" => {
            let bit = if kind == "rand" { "r1" } else { "r6" };
            let mut body: Vec<String> = [
                "mov64 r2, r1",
                "lsh64 r2, 13",
                "xor64 r1, r2",
                "mov64 r2, r1",
                "rsh64 r2, 7",
                "xor64 r1, r2",
                "mov64 r2, r1",
                "lsh64 r2, 17",
                "xor64 r1, r2",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            body.push(format!("jset {bit}, 1, +1"));
            body.push("add64 r3, 1".to_string());
            let dsts = [0, 3, 4, 5, 7, 8, 9];
            body.extend(
                (0..dsts.len() * 11)
                    .map(|k| (dsts[k % dsts.len()], (k % dsts.len() + k / dsts.len()) % 11))
                    .take(n)
                    .map(|(d, s)| format!("add64 r{d}, r{s}")),
            );
            return body;
        }
        // Conditional jumps to the next instruction, taken or not.
        "jcc" => steps.extend(pairs.map(|(d, s)| format!("jgt r{d}, r{s}, +0"))),
        "spread" | "spread-blocked" => {
            // Each opcode's own distinct steps, then taken round robin.
            let per_op: Vec<Vec<String>> = SPREAD
                .iter()
                .map(|template| {
                    let mut own: Vec<String> = pairs
                        .clone()
                        .map(|(d, s)| {
                            template
                                .replace("{d}", &d.to_string())
                                .replace("{s}", &s.to_string())
                        })
                        .collect();
                    own.dedup();
                    let mut seen = std::collections::HashSet::new();
                    own.retain(|step| seen.insert(step.clone()));
                    own
                })
                .collect();
            if kind == "spread-blocked" {
                // The same steps as `spread`, but consecutive ones in the same set.
                let round_robin = body("spread", n);
                for template in SPREAD {
                    let op = template.split(' ').next().unwrap();
                    let imm = !template.contains("{s}");
                    steps.extend(
                        round_robin
                            .iter()
                            .filter(|step| {
                                step.split(' ').next() == Some(op) && imm == !step.contains(", r")
                            })
                            .cloned(),
                    );
                }
                assert_eq!(steps.len(), n);
                return steps;
            }
            for i in 0.. {
                if per_op.iter().all(|own| i >= own.len()) {
                    break;
                }
                for own in &per_op {
                    if let Some(step) = own.get(i) {
                        steps.push(step.clone());
                    }
                }
            }
        }
        _ => panic!("unknown kind {}", kind),
    }
    assert!(n <= steps.len(), "only {} distinct steps", steps.len());
    steps.truncate(n);
    steps
}

/// A loop of `iterations` over the `steps`, repeated to at least 64 instructions per iteration.
fn looped(steps: &[String], iterations: u64) -> (String, u64) {
    let repeats = 64usize.div_ceil(steps.len());
    let mut text = format!("mov64 r1, 88172645\nmov64 r6, {iterations}\nloop:\n");
    for _ in 0..repeats {
        for step in steps {
            text.push_str(step);
            text.push('\n');
        }
    }
    text.push_str("sub64 r6, 1\njne r6, 0, loop\nexit\n");
    let per_iteration = (repeats * steps.len() + 2) as u64;
    (text, per_iteration)
}

fn executable(source: &str) -> Executable<TestContextObject> {
    let config = Config {
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

/// Runs `executable` in `mode` `runs` times, calling `around` with each execution.
fn run(
    executable: &Executable<TestContextObject>,
    parrot: bool,
    runs: usize,
    mut around: impl FnMut(&mut dyn FnMut()),
) {
    let config = executable.get_config();
    let mut stack = vec![0u8; config.stack_size()];
    let regions = vec![
        executable.get_ro_region(),
        MemoryRegion::new(&raw mut stack[..], solana_sbpf::ebpf::MM_STACK_START),
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
    for _ in 0..runs {
        vm.registers = registers;
        vm.context().remaining = 1 << 40;
        let mut mode = if parrot {
            ExecutionMode::ParrotInterpreted
        } else {
            ExecutionMode::Interpreted
        };
        let mut result = None;
        around(&mut || {
            result = Some(
                vm.execute_program(executable, &mut mode, &mut call_frames)
                    .1,
            );
        });
        let result = result.unwrap();
        assert!(result.is_ok(), "{:?}", result);
    }
}

mod perf {
    //! Just enough of `perf_event_open` for counting groups and IBS fetch sampling.

    use std::convert::TryInto;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_TYPE_RAW: u32 = 4;
    const ENABLE: libc::c_ulong = 0x2400;
    const DISABLE: libc::c_ulong = 0x2401;
    const RESET: libc::c_ulong = 0x2403;
    const FLAG_GROUP: libc::c_ulong = 1;

    fn open(
        kind: u32,
        config: u64,
        period: u64,
        sample_type: u64,
        flags: u64,
        group: i32,
    ) -> OwnedFd {
        let mut attr = [0u64; 16];
        attr[0] = kind as u64 | (128 << 32);
        attr[1] = config;
        attr[2] = period;
        attr[3] = sample_type;
        // read_format: TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING | GROUP
        attr[4] = 1 | 2 | 8;
        attr[5] = flags;
        // SAFETY: `attr` is a zeroed `perf_event_attr` of the size it claims.
        let fd = unsafe {
            libc::syscall(libc::SYS_perf_event_open, attr.as_ptr(), 0, -1, group, 0) as i32
        };
        assert!(
            fd >= 0,
            "perf_event_open({kind}, {config:#x}): {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: `fd` was just opened.
        unsafe { OwnedFd::from_raw_fd(fd) }
    }

    fn ioctl(fd: &OwnedFd, request: libc::c_ulong, arg: libc::c_ulong) {
        // SAFETY: The perf ioctls take an integer argument.
        assert_eq!(unsafe { libc::ioctl(fd.as_raw_fd(), request, arg) }, 0);
    }

    /// A raw Zen event.
    pub const fn raw(event: u64, umask: u64) -> (u32, u64) {
        (
            PERF_TYPE_RAW,
            (event & 0xff) | (umask << 8) | ((event >> 8) << 32),
        )
    }
    pub const CYCLES: (u32, u64) = (PERF_TYPE_HARDWARE, 0);
    pub const INSTRUCTIONS: (u32, u64) = (PERF_TYPE_HARDWARE, 1);

    /// Events counted together, in user mode.
    pub struct Group {
        fds: Vec<OwnedFd>,
    }

    impl Group {
        pub fn new(events: &[(u32, u64)]) -> Self {
            let mut fds: Vec<OwnedFd> = Vec::new();
            for &(kind, config) in events {
                let leader = fds.first().map_or(-1, |fd| fd.as_raw_fd());
                // disabled (leader only), exclude_kernel, exclude_hv
                let disabled = if fds.is_empty() { 1 } else { 0 };
                fds.push(open(kind, config, 0, 0, disabled | 1 << 5 | 1 << 6, leader));
            }
            Self { fds }
        }

        /// The counts during `f`, scaled for multiplexing.
        pub fn count(&self, f: &mut dyn FnMut()) -> Vec<f64> {
            let leader = &self.fds[0];
            ioctl(leader, RESET, FLAG_GROUP);
            ioctl(leader, ENABLE, FLAG_GROUP);
            f();
            ioctl(leader, DISABLE, FLAG_GROUP);
            let mut buf = vec![0u64; 3 + self.fds.len()];
            let len = buf.len() * 8;
            // SAFETY: `buf` has room for `len` bytes.
            let read = unsafe { libc::read(leader.as_raw_fd(), buf.as_mut_ptr().cast(), len) };
            assert_eq!(read as usize, len);
            assert_eq!(buf[0] as usize, self.fds.len());
            let (enabled, running) = (buf[1] as f64, buf[2] as f64);
            assert!(running > 0.0, "the group was never scheduled");
            buf[3..]
                .iter()
                .map(|&v| v as f64 * enabled / running)
                .collect()
        }
    }

    /// One IBS fetch sample.
    #[derive(Clone, Copy, Debug)]
    pub struct FetchSample {
        pub ctl: u64,
        pub linear: u64,
        pub extd: u64,
    }

    impl FetchSample {
        fn bit(&self, bit: u32) -> bool {
            self.ctl >> bit & 1 != 0
        }
        pub fn latency(&self) -> u64 {
            self.ctl >> 32 & 0xffff
        }
        pub fn complete(&self) -> bool {
            self.bit(50)
        }
        pub fn ic_miss(&self) -> bool {
            self.bit(51)
        }
        pub fn l1tlb_miss(&self) -> bool {
            self.bit(55)
        }
        pub fn l2tlb_miss(&self) -> bool {
            self.bit(56)
        }
        pub fn l2_miss(&self) -> bool {
            self.bit(58)
        }
        pub fn oc_miss(&self) -> bool {
            self.bit(60)
        }
        pub fn itlb_refill_latency(&self) -> u64 {
            self.extd & 0xffff
        }
    }

    impl FetchSample {
        /// From the raw registers: IbsFetchCtl, IbsFetchLinAd, IbsFetchPhysAd, IbsFetchCtlExtd
        /// (if the CPU has it).
        pub fn from_regs(regs: &[u64]) -> Self {
            Self {
                ctl: regs[0],
                linear: regs[1],
                extd: regs.get(3).copied().unwrap_or(0),
            }
        }
    }

    /// One IBS op sample.
    #[derive(Clone, Copy, Debug)]
    pub struct OpSample {
        pub rip: u64,
        pub data: u64,
        pub data3: u64,
    }

    impl OpSample {
        /// From the raw registers: IbsOpCtl, IbsOpRip, IbsOpData, ...
        pub fn from_regs(regs: &[u64]) -> Self {
            Self {
                rip: regs[1],
                data: regs[2],
                data3: regs[4],
            }
        }
        pub fn rip_valid(&self) -> bool {
            self.data >> 38 & 1 == 0
        }
        /// Cycles from the completion of the op to its retirement.
        pub fn completion_to_retire(&self) -> u64 {
            self.data & 0xffff
        }
        /// Cycles from the tagging (dispatch) of the op to its retirement.
        pub fn tag_to_retire(&self) -> u64 {
            self.data >> 16 & 0xffff
        }
        pub fn mispredicted(&self) -> bool {
            self.data >> 36 & 1 != 0
        }
        pub fn load(&self) -> bool {
            self.data3 & 1 != 0
        }
        pub fn dc_miss(&self) -> bool {
            self.data3 >> 7 & 1 != 0
        }
    }

    /// IBS sampling of this thread, user mode and kernel alike: fetch or op, by the PMU type.
    pub struct Ibs {
        fd: OwnedFd,
        ring: *mut u8,
        ring_len: usize,
        page: usize,
        pub lost: u64,
    }

    impl Ibs {
        const PAGES: usize = 1 << 10;

        /// IBS fetch sampling, every about `period` fetches.
        pub fn fetch(pmu_type: u32, period: u64) -> Self {
            // RandEn randomizes the low bits of the period.
            Self::new(pmu_type, 1 << 57, period)
        }

        /// IBS op sampling, every about `period` dispatched ops.
        pub fn op(pmu_type: u32, period: u64) -> Self {
            // CntCtl counts dispatched ops rather than cycles.
            Self::new(pmu_type, 1 << 19, period)
        }

        fn new(pmu_type: u32, config: u64, period: u64) -> Self {
            // sample_type: IP | RAW
            let fd = open(pmu_type, config, period, 1 | 1 << 10, 1, -1);
            // SAFETY: Plain `sysconf`.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let ring_len = (1 + Self::PAGES) * page;
            // SAFETY: Mapping the ring buffer of `fd`.
            let ring = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    ring_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(
                ring,
                libc::MAP_FAILED,
                "{}",
                std::io::Error::last_os_error()
            );
            Self {
                fd,
                ring: ring.cast(),
                ring_len,
                page,
                lost: 0,
            }
        }

        /// The raw registers of the samples during `f`.
        pub fn sample(&mut self, f: &mut dyn FnMut(), out: &mut Vec<Vec<u64>>) {
            ioctl(&self.fd, ENABLE, 0);
            f();
            ioctl(&self.fd, DISABLE, 0);
            self.drain(out);
        }

        fn drain(&mut self, out: &mut Vec<Vec<u64>>) {
            let data_len = Self::PAGES * self.page;
            // SAFETY: `data_head` and `data_tail` of `perf_event_mmap_page`; `data_head` is
            // published with release ordering by the kernel.
            let (head_ptr, tail_ptr) = unsafe {
                (
                    self.ring.add(1024) as *const u64,
                    self.ring.add(1032) as *mut u64,
                )
            };
            let head = unsafe { std::ptr::read_volatile(head_ptr) };
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let mut tail = unsafe { std::ptr::read_volatile(tail_ptr) };
            // SAFETY: The data area follows the first page.
            let data = unsafe { std::slice::from_raw_parts(self.ring.add(self.page), data_len) };
            let byte = |at: u64| data[(at as usize) % data_len];
            let read = |at: u64, len: usize| -> Vec<u8> {
                (0..len as u64).map(|i| byte(at + i)).collect()
            };
            let u64_at =
                |bytes: &[u8], at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            while tail < head {
                let header = read(tail, 8);
                let kind = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let size = u16::from_le_bytes(header[6..8].try_into().unwrap()) as usize;
                let record = read(tail, size);
                match kind {
                    // PERF_RECORD_SAMPLE: ip, raw size, raw data (u32 caps, u64 regs...)
                    9 => {
                        let raw_size =
                            u32::from_le_bytes(record[16..20].try_into().unwrap()) as usize;
                        let raw = &record[20..20 + raw_size];
                        let regs = &raw[4..];
                        out.push((0..regs.len() / 8).map(|i| u64_at(regs, i * 8)).collect());
                    }
                    // PERF_RECORD_LOST
                    2 => self.lost += u64_at(&record, 16),
                    _ => {}
                }
                tail += size as u64;
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            // SAFETY: See above.
            unsafe { std::ptr::write_volatile(tail_ptr, tail) };
        }
    }

    impl Drop for Ibs {
        fn drop(&mut self) {
            // SAFETY: Mapped in `new`.
            unsafe { libc::munmap(self.ring.cast(), self.ring_len) };
        }
    }
}

/// The symbols of the parrot interpreter of SBPFv3, its steps and its copy of the supporting
/// code, sorted by address.
fn interpreter_symbols() -> Vec<Symbol> {
    let mut symbols = debug::interpreter(SBPFVersion::V3).symbols;
    symbols.sort_by_key(|symbol| symbol.address);
    symbols
}

/// The symbol `address` is in, if any.
fn symbolize(symbols: &[Symbol], address: u64) -> Option<&Symbol> {
    let address = address as usize;
    let at = symbols.partition_point(|symbol| symbol.address <= address);
    let symbol = &symbols[at.checked_sub(1)?];
    (address < symbol.address + symbol.size.max(1)).then_some(symbol)
}

/// Groups of at most 4 events, each led by the cycles that pick the least disturbed of the runs.
const EVENT_GROUPS: [&[(&str, (u32, u64))]; 6] = [
    &[
        ("cycles", perf::CYCLES),
        ("insns", perf::INSTRUCTIONS),
        ("ic_access", perf::raw(0x18e, 0x1f)),
        ("ic_miss", perf::raw(0x18e, 0x18)),
    ],
    &[
        ("cycles", perf::CYCLES),
        ("oc_access", perf::raw(0x28f, 0x07)),
        ("oc_miss", perf::raw(0x28f, 0x04)),
        ("fe_empty_slots", perf::raw(0x1a0, 0x01)),
    ],
    &[
        ("cycles", perf::CYCLES),
        ("itlb_l1m_l2h", perf::raw(0x84, 0x00)),
        ("itlb_l2m", perf::raw(0x85, 0x0f)),
        ("ic_fill_l2", perf::raw(0x82, 0x00)),
    ],
    &[
        ("cycles", perf::CYCLES),
        ("ic_fill_sys", perf::raw(0x83, 0x00)),
        ("redirects_misp", perf::raw(0x9f, 0x02)),
        ("de_redirect", perf::raw(0x91, 0x00)),
    ],
    &[
        ("cycles", perf::CYCLES),
        ("ret_ops", perf::raw(0xc1, 0x00)),
        ("al_token_stall_cycles", perf::raw(0xaf, 0x01)),
        ("backend_stall_slots", perf::raw(0x1a0, 0x1e)),
    ],
    &[
        ("cycles", perf::CYCLES),
        ("taken_branch_buffer_stall_cycles", perf::raw(0xae, 0x10)),
        ("no_retire_not_complete_cycles", perf::raw(0xd6, 0x02)),
        ("ret_taken_branches", perf::raw(0xc4, 0x00)),
    ],
];

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = args.next().unwrap_or_else(|| "same-set".into());
    if kind == "step-sizes" {
        step_sizes();
        return;
    }
    let ns: Vec<usize> = {
        let ns: Vec<usize> = args.map(|n| n.parse().unwrap()).collect();
        if ns.is_empty() {
            vec![1, 4, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32, 48, 64]
        } else {
            ns
        }
    };
    let ibs_type: u32 = std::env::var("IBS_FETCH_TYPE").map_or(10, |t| t.parse().unwrap());
    const TARGET_INSNS: u64 = 50_000_000;
    const RUNS: usize = 7;

    // `PROBE_TIME_ONLY` leaves out IBS and the AMD specific counters, for other CPUs.
    let time_only = std::env::var_os("PROBE_TIME_ONLY").is_some();
    let event_groups: &[&[(&str, (u32, u64))]] = if time_only {
        &[&[("cycles", perf::CYCLES), ("insns", perf::INSTRUCTIONS)]]
    } else {
        &EVENT_GROUPS
    };
    let groups: Vec<(Vec<&str>, perf::Group)> = event_groups
        .iter()
        .map(|events| {
            (
                events.iter().map(|e| e.0).collect(),
                perf::Group::new(&events.iter().map(|e| e.1).collect::<Vec<_>>()),
            )
        })
        .collect();
    let mut ibs = (!time_only).then(|| perf::Ibs::fetch(ibs_type, 0x1000));
    // `ibs_op` is registered right after `ibs_fetch`.
    let ibs_op_type: u32 =
        std::env::var("IBS_OP_TYPE").map_or(ibs_type + 1, |t| t.parse().unwrap());
    let mut ibs_op = std::env::var_os("IBS_OP")
        .filter(|_| !time_only)
        .map(|_| perf::Ibs::op(ibs_op_type, 0x10000));
    let symbols = interpreter_symbols();
    print_huge_pages(symbols[0].address);

    println!("# {kind}");
    for &n in &ns {
        let steps = body(&kind, n);
        let probe_iterations = 1;
        let (_, per_iteration) = looped(&steps, probe_iterations);
        let iterations = TARGET_INSNS / per_iteration;
        let (source, _) = looped(&steps, iterations);
        let insns = iterations * per_iteration + 2;
        let executable = executable(&source);
        for (mode_name, mode) in [("legacy", false), ("parrot", true)] {
            let mut times = Vec::new();
            run(&executable, mode, RUNS, |f| {
                let start = Instant::now();
                f();
                times.push(start.elapsed().as_secs_f64());
            });
            times.sort_by(f64::total_cmp);
            let ns_per_insn = times[RUNS / 2] * 1e9 / insns as f64;
            let mut counts: BTreeMap<&str, f64> = BTreeMap::new();
            for (names, group) in &groups {
                let mut best: Option<Vec<f64>> = None;
                run(&executable, mode, 3, |f| {
                    let c = group.count(f);
                    if best.as_ref().is_none_or(|b| c[0] < b[0]) {
                        best = Some(c);
                    }
                });
                for (name, value) in names.iter().zip(best.unwrap()) {
                    counts.entry(name).or_insert(value);
                }
            }
            let per_k = |name: &str| counts[name] * 1000.0 / insns as f64;
            let mut line = format!(
                "N={n:<3} {mode_name:<6} {ns_per_insn:6.3} ns/insn  {:5.2} cyc/insn  {:5.2} hostinsn/insn",
                counts["cycles"] / insns as f64,
                counts["insns"] / insns as f64,
            );
            if time_only {
                println!("{line}");
                continue;
            }
            for name in [
                "ic_access",
                "ic_miss",
                "ic_fill_l2",
                "ic_fill_sys",
                "oc_access",
                "oc_miss",
                "itlb_l1m_l2h",
                "itlb_l2m",
                "redirects_misp",
                "de_redirect",
            ] {
                line.push_str(&format!("  {name}={:.2}", per_k(name)));
            }
            line.push_str(&format!(
                "  fe_empty_slots/cyc={:.2}",
                counts["fe_empty_slots"] / counts["cycles"]
            ));
            line.push_str(&format!(
                "  ops/insn={:.2} ops/cyc={:.2} backend_stall_slots/cyc={:.2} al_token_stall={:.0}% taken_brnch_buf_stall={:.0}% no_retire_not_complete={:.0}%",
                counts["ret_ops"] / insns as f64,
                counts["ret_ops"] / counts["cycles"],
                counts["backend_stall_slots"] / counts["cycles"],
                100.0 * counts["al_token_stall_cycles"] / counts["cycles"],
                100.0 * counts["taken_branch_buffer_stall_cycles"] / counts["cycles"],
                100.0 * counts["no_retire_not_complete_cycles"] / counts["cycles"],
            ));
            println!("{line}");

            if let Some(ibs) = ibs.as_mut().filter(|_| mode) {
                let mut samples = Vec::new();
                run(&executable, mode, 3, |f| ibs.sample(f, &mut samples));
                let samples: Vec<perf::FetchSample> = samples
                    .iter()
                    .map(|regs| perf::FetchSample::from_regs(regs))
                    .collect();
                report_ibs(&samples, &symbols);
                if let Some(ibs_op) = &mut ibs_op {
                    let mut samples = Vec::new();
                    run(&executable, mode, 3, |f| ibs_op.sample(f, &mut samples));
                    let samples: Vec<perf::OpSample> = samples
                        .iter()
                        .map(|regs| perf::OpSample::from_regs(regs))
                        .collect();
                    report_ibs_ops(&samples, &symbols);
                }
            }
        }
    }
    if let Some(lost) = ibs.as_ref().map(|ibs| ibs.lost).filter(|&lost| lost > 0) {
        println!("# IBS fetch lost {} samples", lost);
    }
    if let Some(lost) = ibs_op.as_ref().map(|ibs| ibs.lost).filter(|&lost| lost > 0) {
        println!("# IBS op lost {} samples", lost);
    }
}

/// Summarises the IBS fetch `samples` of the step `table` by step and by L1i set.
fn report_ibs(samples: &[perf::FetchSample], symbols: &[Symbol]) {
    #[derive(Default)]
    struct Stats {
        n: u64,
        ic_miss: u64,
        l2_miss: u64,
        oc_miss: u64,
        l1tlb_miss: u64,
        l2tlb_miss: u64,
        latency: u64,
        itlb_latency: u64,
    }
    impl Stats {
        fn add(&mut self, s: &perf::FetchSample) {
            self.n += 1;
            self.ic_miss += s.ic_miss() as u64;
            self.l2_miss += (s.ic_miss() && s.l2_miss()) as u64;
            self.oc_miss += s.oc_miss() as u64;
            self.l1tlb_miss += s.l1tlb_miss() as u64;
            self.l2tlb_miss += s.l2tlb_miss() as u64;
            self.latency += s.latency();
            self.itlb_latency += s.itlb_refill_latency();
        }
        fn fmt(&self) -> String {
            let pct = |v: u64| 100.0 * v as f64 / self.n.max(1) as f64;
            format!(
                "n={:<6} ic_miss={:5.1}% (l2_miss {:4.1}%) oc_miss={:5.1}% l1tlb_miss={:5.1}% l2tlb_miss={:4.1}% fetch_lat={:5.1} itlb_lat={:5.1}",
                self.n,
                pct(self.ic_miss),
                pct(self.l2_miss),
                pct(self.oc_miss),
                pct(self.l1tlb_miss),
                pct(self.l2tlb_miss),
                self.latency as f64 / self.n.max(1) as f64,
                self.itlb_latency as f64 / self.l1tlb_miss.max(1) as f64,
            )
        }
    }
    let mut steps: Stats = Stats::default();
    let mut dispatches: Stats = Stats::default();
    let mut supports: Stats = Stats::default();
    let mut other: Stats = Stats::default();
    let mut by_symbol: BTreeMap<(usize, &str, usize), Stats> = BTreeMap::new();
    let mut by_set: BTreeMap<u64, Stats> = BTreeMap::new();
    for s in samples.iter().filter(|s| s.complete()) {
        match symbolize(symbols, s.linear) {
            Some(symbol) => {
                if symbol.name.starts_with("step_") {
                    steps.add(s);
                } else if symbol.name.starts_with("dispatch_") {
                    dispatches.add(s);
                } else {
                    supports.add(s);
                }
                by_symbol
                    .entry((symbol.address, &symbol.name, symbol.size))
                    .or_default()
                    .add(s);
                by_set.entry(s.linear >> 6 & 63).or_default().add(s);
            }
            None => other.add(s),
        }
    }
    println!("    ibs steps     {}", steps.fmt());
    println!("    ibs dispatch  {}", dispatches.fmt());
    println!("    ibs supports  {}", supports.fmt());
    println!("    ibs other     {}", other.fmt());
    for (set, stats) in &by_set {
        println!("      set {set:2}  {}", stats.fmt());
    }
    if std::env::var_os("IBS_BY_SYMBOL").is_some() {
        for ((address, name, size), stats) in &by_symbol {
            println!(
                "      {name:<28} {size:3}B sets {:2}..={:2}  {}",
                address >> 6 & 63,
                (address + size - 1) >> 6 & 63,
                stats.fmt()
            );
        }
    }
}

/// The distribution of the lengths of the steps of each interpreter, and the code of an `add64`
/// step written to `add64_step.bin` for disassembly.
fn step_sizes() {
    for version in [SBPFVersion::V0, SBPFVersion::V2, SBPFVersion::V3] {
        let region = debug::interpreter(version);
        let steps: Vec<&Symbol> = region
            .symbols
            .iter()
            .filter(|symbol| {
                // Valid instructions only: a known mnemonic and registers up to r10.
                let registers = |field: &str| -> Option<u8> {
                    let at = symbol.name.find(field)? + field.len();
                    let digits: String = symbol.name[at..]
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect();
                    digits.parse().ok()
                };
                symbol.name.starts_with("step_")
                    && !symbol.name.contains("_op")
                    && registers("_dst").is_some_and(|r| r <= 10)
                    && registers("_src").is_some_and(|r| r <= 10)
            })
            .collect();
        let mut histogram: BTreeMap<usize, usize> = BTreeMap::new();
        for step in &steps {
            *histogram.entry(step.size.div_ceil(8) * 8).or_default() += 1;
        }
        let over = |limit: usize| steps.iter().filter(|step| step.size > limit).count();
        println!(
            "{version:?}: {} valid steps, {} over 64 B, {} over 56 B, {} over 48 B",
            steps.len(),
            over(64),
            over(56),
            over(48)
        );
        println!("  size (rounded up to 8) -> steps: {histogram:?}");
        // The steps that end with a `jmp rel32` to a copy of the dispatch, rather than with the
        // dispatch's `jmp rcx`, possibly followed by its `jmp rel32` to the island.
        let jumps_out = |step: &Symbol| {
            let end = step.address - region.start + step.size;
            step.size >= 5
                && region.bytes[end - 5] == 0xe9
                && !(step.size >= 7 && region.bytes[end - 7..end - 5] == [0xff, 0xe1])
        };
        let mut out_of_line: BTreeMap<String, usize> = BTreeMap::new();
        for step in steps.iter().filter(|step| jumps_out(step)) {
            let kind = step.name.split("_dst").next().unwrap().to_string();
            *out_of_line.entry(kind).or_default() += 1;
        }
        println!(
            "  {} with the dispatch inline, {} jumping to their copy of it: {out_of_line:?}",
            steps.iter().filter(|step| !jumps_out(step)).count(),
            out_of_line.values().sum::<usize>(),
        );
        let mut long: Vec<&&Symbol> = steps.iter().filter(|step| step.size > 64).collect();
        long.sort_by_key(|step| std::cmp::Reverse(step.size));
        let mut kinds: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for step in long {
            let kind = step.name.split("_dst").next().unwrap().to_string();
            let entry = kinds.entry(kind).or_default();
            entry.0 += 1;
            entry.1 = entry.1.max(step.size);
        }
        println!("  over 64 B, by opcode (count, max size): {kinds:?}");
        if version == SBPFVersion::V3 {
            let add = steps
                .iter()
                .find(|step| step.name == "step_0f_add64_dst2_src3")
                .unwrap();
            let offset = add.address - region.start;
            std::fs::write("add64_step.bin", &region.bytes[offset..offset + add.size]).unwrap();
        }
    }
}

/// Prints how much of the mapping at `address` is backed by huge pages, from `/proc/self/smaps`.
fn print_huge_pages(address: usize) {
    let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
    let mut inside = false;
    for line in smaps.lines() {
        if let Some((range, _)) = line
            .split_once(' ')
            .filter(|(range, _)| range.contains('-'))
        {
            let (start, end) = range.split_once('-').unwrap();
            let (start, end) = (
                usize::from_str_radix(start, 16).unwrap_or(0),
                usize::from_str_radix(end, 16).unwrap_or(0),
            );
            inside = (start..end).contains(&address);
            if inside {
                println!("# step table mapping {line}");
            }
        } else if inside
            && ["Size", "AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"]
                .iter()
                .any(|field| line.starts_with(&format!("{field}:")))
        {
            println!(
                "#   {}",
                line.split_whitespace().collect::<Vec<_>>().join(" ")
            );
        }
    }
}

/// Summarises the IBS op `samples` by the instruction they are of: the kind of symbol (the steps of
/// the same operation together, whatever their registers) and the offset in it. With
/// `IBS_OP_DUMP` set, writes the code of a symbol of each kind into `ibs_<kind>.bin` for
/// disassembly.
fn report_ibs_ops(samples: &[perf::OpSample], symbols: &[Symbol]) {
    #[derive(Default)]
    struct Stats {
        n: u64,
        tag_to_retire: u64,
        completion_to_retire: u64,
        mispredicted: u64,
        loads: u64,
        dc_misses: u64,
    }
    let kind_of = |name: &str| name.split("_dst").next().unwrap().to_string();
    let mut total = 0u64;
    let mut by_kind: BTreeMap<String, (u64, &Symbol)> = BTreeMap::new();
    let mut by_insn: BTreeMap<(String, usize), Stats> = BTreeMap::new();
    for s in samples.iter().filter(|s| s.rip_valid()) {
        total += 1;
        let (kind, offset) = match symbolize(symbols, s.rip) {
            Some(symbol) => {
                let kind = kind_of(&symbol.name);
                by_kind.entry(kind.clone()).or_insert((0, symbol)).0 += 1;
                (kind, s.rip as usize - symbol.address)
            }
            None => ("other".to_string(), 0),
        };
        let stats = by_insn.entry((kind, offset)).or_default();
        stats.n += 1;
        stats.tag_to_retire += s.tag_to_retire();
        stats.completion_to_retire += s.completion_to_retire();
        stats.mispredicted += s.mispredicted() as u64;
        stats.loads += s.load() as u64;
        stats.dc_misses += (s.load() && s.dc_miss()) as u64;
    }
    println!("    ibs op samples {total}");
    for ((kind, offset), stats) in &by_insn {
        let share = 100.0 * stats.n as f64 / total.max(1) as f64;
        if share < std::env::var("IBS_OP_MIN").map_or(0.5, |m| m.parse().unwrap()) {
            continue;
        }
        println!(
            "      {kind:<24} +{offset:#04x}  {share:5.1}%  tag_to_retire={:6.1}  completion_to_retire={:6.1}  misp={}  dc_miss={}",
            stats.tag_to_retire as f64 / stats.n as f64,
            stats.completion_to_retire as f64 / stats.n as f64,
            stats.mispredicted,
            if stats.loads > 0 { format!("{:.0}%", 100.0 * stats.dc_misses as f64 / stats.loads as f64) } else { "-".to_string() },
        );
    }
    if std::env::var_os("IBS_OP_DUMP").is_some() {
        let region = debug::interpreter(SBPFVersion::V3);
        for (kind, (_, symbol)) in &by_kind {
            let offset = symbol.address - region.start;
            let bytes = &region.bytes[offset..offset + symbol.size];
            std::fs::write(format!("ibs_{kind}.bin"), bytes).unwrap();
        }
    }
}
