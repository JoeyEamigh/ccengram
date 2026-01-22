pub mod chunker;
pub mod coordination;
pub mod debounce;
pub mod gitignore;
pub mod parser;
pub mod scanner;
pub mod watcher;

pub use chunker::{Chunker, ChunkerConfig};
pub use coordination::{CoordinationError, WatcherCoordinator, WatcherLock};
pub use debounce::{BatchProcessor, DebounceConfig, DebouncedWatcher};
pub use gitignore::{GitignoreState, compute_gitignore_hash, should_ignore};
pub use parser::{detect_language, is_indexable, supported_extensions};
pub use scanner::{ScanError, ScanProgress, ScanResult, ScannedFile, Scanner};
pub use watcher::{ChangeKind, FileChange, FileWatcher, WatchError};
