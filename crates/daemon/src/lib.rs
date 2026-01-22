pub mod hooks;
pub mod lifecycle;
pub mod projects;
pub mod router;
pub mod scheduler;
pub mod server;
pub mod tools;

pub use db::{default_cache_dir, default_config_dir, default_data_dir, default_port};
pub use hooks::{HookError, HookEvent, HookHandler};
pub use lifecycle::{Daemon, DaemonConfig, LifecycleError, is_running};
pub use projects::{ProjectError, ProjectInfo, ProjectRegistry};
pub use router::{Request, Response, Router, RpcError};
pub use scheduler::{Scheduler, SchedulerConfig, spawn_scheduler};
pub use server::{Client, Server, ServerError, ShutdownHandle, default_socket_path};
pub use tools::ToolHandler;
