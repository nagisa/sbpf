// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Pooled JIT memory must stay sound across `fork`.
//!
//! This is the only test in this binary on purpose: forking a process with other threads that use
//! the pool could leave the child with a pool mutex nobody will ever unlock.

#![cfg(target_os = "linux")]

extern crate libc;

use solana_sbpf::program::JitProgram;

const PC: usize = 1024;
const CODE: usize = 64 * 1024;

/// Compile-and-touch every page of a fresh program, as a stand-in for real use.
fn exercise(program: &mut JitProgram) {
    program.pc_section_mut().fill(0x5a5a5a5a);
    program.text_section_mut().fill(0xc3);
    program.seal(CODE / 2).unwrap();
    assert!(program.text_section().iter().all(|&b| b == 0xc3));
}

/// Runs `body` in a forked child, reporting whether it exited cleanly.
fn in_child(body: impl FnOnce()) -> bool {
    unsafe {
        let pid = libc::fork();
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_ok();
            libc::_exit(if ok { 0 } else { 1 });
        }
        let mut status = 0;
        assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
    }
}

#[test]
fn pool_survives_fork() {
    // Fill the pool: these blocks are what a child must not be handed.
    let pooled = (0..8)
        .map(|_| JitProgram::new(PC, CODE))
        .collect::<Vec<_>>();
    drop(pooled);
    let mut live = JitProgram::new(PC, CODE);
    exercise(&mut live);
    assert!(live.is_valid());

    let ok = in_child(|| {
        // Programs from before the fork are not mapped in the child any more.
        assert!(!live.is_valid());

        // Allocating draws on the (stale) pool. Touching every page is how a hole is noticed.
        let fresh = (0..16)
            .map(|_| {
                let mut p = JitProgram::new(PC, CODE);
                assert!(p.is_valid());
                exercise(&mut p);
                p
            })
            .collect::<Vec<_>>();

        // Dropping a stale program must neither panic (`mprotect` of a hole fails) nor put the
        // stale block in the pool. This is the child's own copy: the child exits right after.
        drop(unsafe { std::ptr::read(&live) });
        for _ in 0..16 {
            exercise(&mut JitProgram::new(PC, CODE));
        }

        // A fork of the fork starts a new generation again.
        assert!(in_child(|| {
            assert!(fresh.iter().all(|p| !p.is_valid()));
            for _ in 0..16 {
                exercise(&mut JitProgram::new(PC, CODE));
            }
        }));
    });
    assert!(ok, "child failed");

    // The parent's view is unaffected by whatever the child did.
    assert!(live.is_valid());
    assert!(live.text_section().iter().all(|&b| b == 0xc3));
    for _ in 0..16 {
        exercise(&mut JitProgram::new(PC, CODE));
    }
}
