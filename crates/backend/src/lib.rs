mod actor;
mod context;
mod db;
mod embedding;
mod rerank;
mod server;
mod service;

mod domain;
pub use domain::{config, project};

pub mod dirs;
pub mod ipc;

mod daemon;
pub use daemon::{Daemon, RuntimeConfig};
//
// --- all the different lovely profiling tools ---
//

//
#[cfg(all(not(target_env = "msvc"), feature = "jemalloc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(all(not(target_env = "msvc"), feature = "jemalloc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(all(not(target_env = "msvc"), feature = "jemalloc-pprof"))]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "malloc_conf")]
pub static malloc_conf: &[u8] = b"prof:true,prof_active:true,lg_prof_sample:19\0";

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
