// Copyright 2022 Solana Maintainers <maintainers@solana.com>
//
// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![cfg_attr(target_os = "windows", allow(dead_code))]

#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::{
    ptr::NonNull,
    sync::{LazyLock, Mutex},
};

use crate::error::EbpfError;

#[cfg(not(target_os = "windows"))]
extern crate libc;
#[cfg(not(target_os = "windows"))]
use libc::c_void;

#[cfg(target_os = "windows")]
use {
    core::ffi::c_void,
    windows_sys::Win32::{
        Foundation::GetLastError,
        System::{
            Memory::{
                VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
                MEM_RESET, PAGE_EXECUTE_READ, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE,
            },
            SystemInformation::{GetSystemInfo, SYSTEM_INFO},
        },
    },
};

/// Detects `fork` by way of a `MADV_WIPEONFORK` page, and counts forks as a *generation*.
///
/// Pool memory is `MAP_SHARED` (so that `mprotect` can be cheap) and `MADV_DONTFORK`, which means
/// that after a `fork` the child has a hole where every mapping used to be. Anything the child
/// inherited that points into such a hole is dangling, and worse, the kernel is free to place a
/// new unrelated mapping into the hole. Each mapping is therefore stamped with the generation it
/// was created in, and mappings of an older generation are never touched again: not read, not
/// `mprotect`ed, not `munmap`ed and not handed out of the pool.
///
/// The kernel zeroes a `MADV_WIPEONFORK` page in the child, so noticing a fork costs a single
/// load in the common case and no syscalls or locks.
#[cfg(target_os = "linux")]
struct ForkTracker {
    /// Lives in a `MADV_WIPEONFORK` page; the page is never unmapped.
    state: &'static AtomicU32,
    /// The current generation. Starts at 1 and is bumped exactly once per observed fork.
    generation: AtomicU64,
}

#[cfg(target_os = "linux")]
impl ForkTracker {
    /// The page was wiped by a fork (or never initialized).
    const WIPED: u32 = 0;
    /// A thread is bumping the generation.
    const REFRESHING: u32 = 1;
    /// The generation is current.
    const CLEAN: u32 = 2;

    /// Returns `None` when the kernel does not support `MADV_WIPEONFORK` (Linux < 4.14).
    fn new() -> Option<Self> {
        unsafe {
            let len = get_system_page_size();
            let page = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            );
            if page == libc::MAP_FAILED {
                return None;
            }
            // Only private anonymous mappings may be wiped on fork, hence this is not part of
            // the (shared) pool memory.
            if libc::madvise(page, len, libc::MADV_WIPEONFORK) != 0 {
                libc::munmap(page, len);
                return None;
            }
            let state = &*page.cast::<AtomicU32>();
            state.store(Self::CLEAN, Ordering::Release);
            Some(Self {
                state,
                generation: AtomicU64::new(1),
            })
        }
    }

    /// The current generation, noticing any fork that happened since the last call.
    #[inline]
    fn generation(&self) -> u64 {
        if self.state.load(Ordering::Acquire) != Self::CLEAN {
            self.refresh();
        }
        self.generation.load(Ordering::Acquire)
    }

    #[cold]
    fn refresh(&self) {
        loop {
            match self.state.compare_exchange(
                Self::WIPED,
                Self::REFRESHING,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // The bump must be visible before `CLEAN` is, so that no thread observes
                    // `CLEAN` and goes on to use the previous generation.
                    self.generation.fetch_add(1, Ordering::Relaxed);
                    self.state.store(Self::CLEAN, Ordering::Release);
                    return;
                }
                Err(Self::CLEAN) => return,
                // Another thread is bumping the generation, which only takes an instant.
                Err(_) => std::hint::spin_loop(),
            }
        }
    }
}

#[cfg(target_os = "linux")]
static FORK_TRACKER: LazyLock<Option<ForkTracker>> = LazyLock::new(ForkTracker::new);

/// Identifies which address space (as in: before or after which `fork`) a mapping belongs to.
///
/// Always `0` on targets that cannot track forks. Those targets use `MAP_PRIVATE` mappings,
/// which are already fork-safe.
#[inline]
fn current_generation() -> u64 {
    cfg_select! {
        target_os = "linux" => {
            match &*FORK_TRACKER {
                Some(tracker) => tracker.generation(),
                None => 0,
            }
        }
        _ => 0
    }
}

/// A free list for managing memory allocations of a fixed size.
struct FreeList {
    /// Pool of free blocks awaiting reuse, each with the generation it was allocated in.
    mem: Mutex<Vec<(*mut u8, u64)>>,
    /// The size of each memory block.
    size: usize,
}

// Safety: FreeList only stores mmap allocation base addresses. The pointed-to
// memory is not accessed through FreeList without holding the FreeList
// mutex, and ownership of each allocation is transferred into/out of the pool.
unsafe impl Sync for FreeList {}
unsafe impl Send for FreeList {}

impl FreeList {
    /// Create a new free list with the specified size.
    ///
    /// This does not allocate any memory blocks; they are allocated lazily as needed.
    fn new(size: usize) -> Self {
        Self {
            mem: Mutex::new(Vec::new()),
            size,
        }
    }

    /// Allocate a memory block of the configured size.
    ///
    /// If a free block is available, it is reused; otherwise, a new block is allocated.
    ///
    /// Returns a pointer to the allocated memory, the size of the allocation and the generation
    /// the allocation belongs to.
    /// Returned memory has read-write permissions and may contain arbitrary
    /// bytes left over from a previous owner; the caller should not assume
    /// any particular contents.
    fn alloc(&self) -> (*mut u8, usize, u64) {
        // Notice a fork before looking at the pool.
        let generation = current_generation();
        let ptr = {
            let mut mem = self.mem.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                match mem.pop() {
                    Some((ptr, g)) if g == generation => break Some(ptr),
                    // The block predates a fork and is not mapped in this process anymore. Forget
                    // it: the address may have been reused by something else, so it must not be
                    // unmapped either.
                    Some(_) => continue,
                    None => break None,
                }
            }
        };
        let ptr = match ptr {
            Some(ptr) => ptr,
            None => unsafe { allocate_pages(self.size) }.expect("allocation failed"),
        };

        (ptr, self.size, generation)
    }

    /// Free the given allocation, returning it to the pool.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by [`FreeList::alloc`] on this same
    ///   instance and not already returned to the pool.
    /// - `size` must equal the size configured at construction.
    /// - `generation` must be the one `alloc` returned alongside `ptr`.
    /// - The caller must not retain any reference into the block after calling
    ///   `free`; subsequent `alloc` calls may hand the same memory to another
    ///   owner.
    unsafe fn free(&self, ptr: *mut u8, size: usize, generation: u64) {
        /// The threshold for discarding physical backing from returned memory.
        ///
        /// Allocations at or above 128 MiB are uncommon, so drop their
        /// resident pages when they are returned to the pool.
        const MADV_DONTNEED_THRESHOLD: usize = 1024 * 1024 * 128; // 128 MiB

        if size != self.size {
            panic!("free size mismatch: expected {}, got {}", self.size, size);
        }

        // The block was allocated before a fork. It is not mapped in this process anymore (or,
        // worse, something unrelated is mapped there), so there is nothing to give back.
        if generation != current_generation() {
            return;
        }

        unsafe { protect_pages(ptr, self.size, PagePermissions::ReadWrite) }
            .expect("failed to protect pages");

        if self.size >= MADV_DONTNEED_THRESHOLD {
            if let Err(e) = unsafe { madvise(ptr, self.size, Advice::DontNeed) } {
                log::error!("FreeList: unable to advise returned allocation: {e}");
            }
        }

        self.mem
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((ptr, generation));
    }
}

impl Drop for FreeList {
    fn drop(&mut self) {
        let generation = current_generation();
        for (ptr, _) in self
            .mem
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .filter(|&(_, g)| g == generation)
        {
            if let Err(e) = unsafe { free_pages(ptr, self.size) } {
                log::error!("FreeList: unable to free {e}");
            }
        }
    }
}

/// Minimum allocation size for a bucket.
const BUCKET_MIN: usize = 1024 * 128; // 128 KiB
/// Maximum allocation size for a bucket.
const BUCKET_MAX: usize = 1024 * 1024 * 256; // 256 MiB
/// Number of buckets in the free list.
const BUCKET_COUNT: usize =
    (BUCKET_MAX.trailing_zeros() - BUCKET_MIN.trailing_zeros()) as usize + 1;

const _: () = assert!(BUCKET_MIN.is_power_of_two());
const _: () = assert!(BUCKET_MAX.is_power_of_two());
const _: () = assert!(BUCKET_MIN <= BUCKET_MAX);
const _: () = assert!(BUCKET_MAX == BUCKET_MIN * (1 << (BUCKET_COUNT - 1)));

/// A free list that uses a bucketed strategy to manage memory
/// allocations of varying sizes.
///
/// Buckets are organized by power-of-two size, with the smallest
/// bucket being [`BUCKET_MIN`] and the largest being [`BUCKET_MAX`].
///
/// Allocations will be rounded up to the nearest power-of-two size
/// and stored in the corresponding bucket.
///
/// Returned blocks remain cached in the process-global pool and are not
/// released back to the OS during normal operation. This intentionally trades
/// higher retained RSS after peak load for fewer mmap/munmap calls during JIT
/// churn.
///
/// This is safe to use in a multi-threaded context -- locks are
/// sharded per bucket.
struct BucketedFreeList {
    buckets: [FreeList; BUCKET_COUNT],
}

impl BucketedFreeList {
    /// Construct an empty pool with one bucket per power-of-two size class.
    #[expect(clippy::arithmetic_side_effects)]
    fn new() -> Self {
        Self {
            buckets: core::array::from_fn(|i| FreeList::new(BUCKET_MIN * (1 << i))),
        }
    }

    /// Round up the requested size to the nearest power-of-two
    /// and determine the corresponding bucket index.
    #[inline]
    #[expect(clippy::arithmetic_side_effects)]
    fn bucket_idx(size: usize) -> usize {
        let bucket_bits = usize::BITS - (size.max(BUCKET_MIN) - 1).leading_zeros();
        bucket_bits as usize - const { BUCKET_MIN.trailing_zeros() as usize }
    }

    /// Allocate memory of at least the given size, returning a pointer to the allocation,
    /// the actual size allocated and the generation it belongs to.
    fn alloc(&self, size: usize) -> (*mut u8, usize, u64) {
        self.buckets[Self::bucket_idx(size)].alloc()
    }

    /// Free the given allocation, returning it to the pool.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by [`BucketedFreeList::alloc`] on this same
    ///   instance and not already returned to the pool.
    /// - `generation` must be the one `alloc` returned alongside `ptr`.
    /// - The caller must not retain any reference into the block after calling
    ///   `free`; subsequent `alloc` calls may hand the same memory to another
    ///   owner.
    unsafe fn free(&self, ptr: *mut u8, size: usize, generation: u64) {
        unsafe { self.buckets[Self::bucket_idx(size)].free(ptr, size, generation) }
    }
}

static ALLOCATOR: LazyLock<BucketedFreeList> = LazyLock::new(BucketedFreeList::new);

/// An owned block of pooled pages, returned to the pool when dropped.
///
/// The pages are read-write when handed out and may contain arbitrary bytes left over from a
/// previous owner.
///
/// # Forking
///
/// On Linux the pages are shared (rather than private) mappings, which are not inherited by a
/// `fork`ed child process. In a child, a `PooledPages` created before the `fork` therefore
/// refers to memory that does not exist: accessing it faults or, worse, hits whatever has been
/// mapped at the address since. [`PooledPages::is_valid`] tells these apart. Dropping such a
/// block is fine, it is simply forgotten.
pub struct PooledPages {
    ptr: NonNull<u8>,
    size: usize,
    generation: u64,
}

// Safety: this uniquely owns the memory it points to.
unsafe impl Send for PooledPages {}
unsafe impl Sync for PooledPages {}

impl PooledPages {
    /// Allocate pages with room for at least `size` bytes.
    pub fn new(size: usize) -> Self {
        let (ptr, size, generation) = ALLOCATOR.alloc(size);
        Self {
            ptr: NonNull::new(ptr).expect("the pooled allocation is never null"),
            size,
            generation,
        }
    }

    /// The start of the pages.
    pub fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    /// The actual size of the allocation, which is at least what was asked for.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the pages are still mapped in this process.
    ///
    /// This is `false` in a forked child for pages created before the fork. The pages must not
    /// be accessed then.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.generation == current_generation()
    }
}

impl Drop for PooledPages {
    fn drop(&mut self) {
        unsafe { ALLOCATOR.free(self.ptr.as_ptr(), self.size, self.generation) }
    }
}

#[cfg(not(target_os = "windows"))]
macro_rules! libc_error_guard {
    (succeeded?, mmap, $addr:expr, $($arg:expr),*) => {{
        *$addr = libc::mmap(*$addr, $($arg),*);
        *$addr != libc::MAP_FAILED
    }};
    (succeeded?, $function:ident, $($arg:expr),*) => {
        libc::$function($($arg),*) == 0
    };
    ($function:ident, $($arg:expr),* $(,)?) => {{
        const RETRY_COUNT: usize = 3;
        for i in 0..RETRY_COUNT {
            if libc_error_guard!(succeeded?, $function, $($arg),*) {
                break;
            } else if i.saturating_add(1) == RETRY_COUNT {
                let args = vec![$(format!("{:?}", $arg)),*];
                #[cfg(any(target_os = "freebsd", target_os = "ios", target_os = "macos"))]
                let errno = *libc::__error();
                #[cfg(any(target_os = "android", target_os = "netbsd", target_os = "openbsd"))]
                let errno = *libc::__errno();
                #[cfg(target_os = "linux")]
                let errno = *libc::__errno_location();
                return Err(EbpfError::LibcInvocationFailed(stringify!($function), args, errno));
            }
        }
    }};
}

#[cfg(target_os = "windows")]
macro_rules! winapi_error_guard {
    (succeeded?, VirtualAlloc, $addr:expr, $($arg:expr),*) => {{
        *$addr = VirtualAlloc(*$addr, $($arg),*);
        !(*$addr).is_null()
    }};
    (succeeded?, $function:ident, $($arg:expr),*) => {
        $function($($arg),*) != 0
    };
    ($function:ident, $($arg:expr),* $(,)?) => {{
        if !winapi_error_guard!(succeeded?, $function, $($arg),*) {
            let args = vec![$(format!("{:?}", $arg)),*];
            let errno = GetLastError();
            return Err(EbpfError::LibcInvocationFailed(stringify!($function), args, errno as i32));
        }
    }};
}

pub fn get_system_page_size() -> usize {
    #[cfg(not(target_os = "windows"))]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
    #[cfg(target_os = "windows")]
    unsafe {
        let mut system_info: SYSTEM_INFO = std::mem::zeroed();
        GetSystemInfo(&mut system_info);
        system_info.dwPageSize as usize
    }
}

pub fn round_to_page_size(value: usize, page_size: usize) -> usize {
    value
        .saturating_add(page_size)
        .saturating_sub(1)
        .checked_div(page_size)
        .unwrap()
        .saturating_mul(page_size)
}

pub unsafe fn allocate_pages(size_in_bytes: usize) -> Result<*mut u8, EbpfError> {
    let mut raw: *mut c_void = std::ptr::null_mut();
    cfg_select! {
        windows => {
            winapi_error_guard!(
                VirtualAlloc,
                &mut raw,
                size_in_bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
        }
        target_os = "linux" => {
            // Shared mappings make `mprotect` cheaper, but are only sound to use if forks can be
            // told apart (see `ForkTracker`).
            let shared = FORK_TRACKER.is_some();
            let flags = if shared {
                libc::MAP_ANONYMOUS | libc::MAP_SHARED
            } else {
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE
            };
            libc_error_guard!(
                mmap,
                &mut raw,
                size_in_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            );
            if shared {
                libc_error_guard!(madvise, raw, size_in_bytes, libc::MADV_DONTFORK);
            }
        }
        _ => {
            libc_error_guard!(
                mmap,
                &mut raw,
                size_in_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            );
        }
    }
    Ok(raw.cast())
}

pub unsafe fn free_pages(raw: *mut u8, size_in_bytes: usize) -> Result<(), EbpfError> {
    #[cfg(not(target_os = "windows"))]
    libc_error_guard!(munmap, raw.cast::<c_void>(), size_in_bytes);
    #[cfg(target_os = "windows")]
    winapi_error_guard!(
        VirtualFree,
        raw.cast::<c_void>(),
        size_in_bytes,
        MEM_RELEASE,
    );
    Ok(())
}

#[derive(Copy, Clone)]
pub enum PagePermissions {
    Read,
    ReadWrite,
    ReadExecute,
}

pub unsafe fn protect_pages(
    raw: *mut u8,
    size_in_bytes: usize,
    permissions: PagePermissions,
) -> Result<(), EbpfError> {
    #[cfg(not(target_os = "windows"))]
    {
        let prot = match permissions {
            PagePermissions::Read => libc::PROT_READ,
            PagePermissions::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
            PagePermissions::ReadExecute => libc::PROT_READ | libc::PROT_EXEC,
        };
        libc_error_guard!(mprotect, raw.cast::<c_void>(), size_in_bytes, prot);
    }
    #[cfg(target_os = "windows")]
    {
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        let ptr_old: *mut PAGE_PROTECTION_FLAGS = &mut old;
        let prot = match permissions {
            PagePermissions::Read => PAGE_READONLY,
            PagePermissions::ReadWrite => PAGE_READWRITE,
            PagePermissions::ReadExecute => PAGE_EXECUTE_READ,
        };
        winapi_error_guard!(
            VirtualProtect,
            raw.cast::<c_void>(),
            size_in_bytes,
            prot,
            ptr_old,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub enum Advice {
    DontNeed,
}

pub unsafe fn madvise(raw: *mut u8, size_in_bytes: usize, advice: Advice) -> Result<(), EbpfError> {
    #[cfg(not(target_os = "windows"))]
    {
        let advice = match advice {
            Advice::DontNeed => libc::MADV_DONTNEED,
        };
        libc_error_guard!(madvise, raw.cast::<c_void>(), size_in_bytes, advice);
    }

    #[cfg(target_os = "windows")]
    {
        let mut ptr = raw.cast::<c_void>();
        let advice = match advice {
            Advice::DontNeed => MEM_RESET,
        };
        winapi_error_guard!(
            VirtualAlloc,
            &mut ptr,
            size_in_bytes,
            advice,
            PAGE_READWRITE,
        );
    }

    Ok(())
}
