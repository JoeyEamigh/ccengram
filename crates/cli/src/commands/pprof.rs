//! Pprof heap profile download command (jemalloc-pprof feature only)

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use tokio::{io::AsyncReadExt, net::UnixStream};

/// Get the pprof socket path (mirrors server logic)
fn get_pprof_socket_path() -> PathBuf {
  if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
    PathBuf::from(runtime_dir).join("ccengram-pprof.sock")
  } else {
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/{}-pprof.sock", uid))
  }
}

/// Download pprof heap profile from the daemon to the current directory.
pub async fn cmd_pprof(output: Option<&str>) -> Result<()> {
  let socket_path = get_pprof_socket_path();

  if !socket_path.exists() {
    bail!(
      "Pprof socket not found at {:?}. Is the daemon running with jemalloc-pprof enabled?",
      socket_path
    );
  }

  let mut stream = UnixStream::connect(&socket_path)
    .await
    .context("Failed to connect to pprof socket")?;

  let mut data = Vec::new();
  stream
    .read_to_end(&mut data)
    .await
    .context("Failed to read pprof data")?;

  if data.is_empty() {
    bail!("Received empty pprof data from daemon");
  }

  let output_path = if let Some(path) = output {
    PathBuf::from(path)
  } else {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    PathBuf::from(format!("heap_{}.pb.gz", timestamp))
  };

  tokio::fs::write(&output_path, &data)
    .await
    .context("Failed to write pprof file")?;

  println!("Wrote {} bytes to {}", data.len(), output_path.display());
  println!("View with: go tool pprof {}", output_path.display());

  Ok(())
}
