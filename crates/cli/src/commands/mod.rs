//! CLI command implementations

mod admin;
mod agent;
mod context;
mod daemon;
mod hook;
mod index;
mod logs;
mod memory;
mod migrate;
mod projects;
mod search;
mod update;
mod watch;

#[cfg(all(unix, feature = "jemalloc-pprof"))]
mod pprof;

pub use admin::{cmd_archive, cmd_config_init, cmd_config_reset, cmd_config_show, cmd_health, cmd_stats};
pub use agent::{cmd_agent, cmd_tui};
pub use context::cmd_context;
pub use daemon::cmd_daemon;
pub use hook::cmd_hook;
pub use index::cmd_index;
pub use logs::{cmd_logs, cmd_logs_list};
pub use memory::{cmd_delete, cmd_deleted, cmd_restore, cmd_show};
pub use migrate::cmd_migrate;
#[cfg(all(unix, feature = "jemalloc-pprof"))]
pub use pprof::cmd_pprof;
pub use projects::{cmd_projects_clean, cmd_projects_clean_all, cmd_projects_list, cmd_projects_show};
pub use search::{cmd_search, cmd_search_code, cmd_search_docs};
pub use update::cmd_update;
pub use watch::cmd_watch;
