import { parseArgs } from "util";
import { join, dirname } from "node:path";
import { log } from "../../utils/log.js";
import { getPaths } from "../../utils/paths.js";

const REPO = "JoeyEamigh/ccmemory";
const UPDATE_CHECK_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

type ReleaseInfo = {
  tag_name: string;
  assets: Array<{ name: string; browser_download_url: string }>;
};

function detectPlatform(): string {
  const os = process.platform;
  const arch = process.arch;

  let osName: string;
  switch (os) {
    case "linux":
      osName = "linux";
      break;
    case "darwin":
      osName = "darwin";
      break;
    case "win32":
      osName = "windows";
      break;
    default:
      throw new Error(`Unsupported OS: ${os}`);
  }

  let archName: string;
  switch (arch) {
    case "x64":
      archName = "x64";
      break;
    case "arm64":
      archName = "arm64";
      break;
    default:
      throw new Error(`Unsupported architecture: ${arch}`);
  }

  return `${osName}-${archName}`;
}

async function getLatestRelease(): Promise<ReleaseInfo | null> {
  try {
    const res = await fetch(
      `https://api.github.com/repos/${REPO}/releases/latest`,
      {
        headers: { Accept: "application/vnd.github.v3+json" },
        signal: AbortSignal.timeout(10000),
      }
    );

    if (!res.ok) {
      log.warn("update", "Failed to fetch latest release", { status: res.status });
      return null;
    }

    return (await res.json()) as ReleaseInfo;
  } catch (err) {
    log.warn("update", "Failed to check for updates", {
      error: err instanceof Error ? err.message : String(err),
    });
    return null;
  }
}

function getCurrentVersion(): string {
  // In production, this would be replaced at build time
  // For now, try to read from adjacent version file or return unknown
  const binaryPath = process.argv[0] ?? "";
  const versionFile = join(dirname(binaryPath), ".ccmemory-version");

  try {
    const version = Bun.file(versionFile);
    // This is sync-ish in Bun
    return version.toString().trim() || "unknown";
  } catch {
    return "unknown";
  }
}

async function getLastUpdateCheck(): Promise<number> {
  const paths = getPaths();
  const checkFile = join(paths.cache, "last-update-check");

  try {
    const file = Bun.file(checkFile);
    if (await file.exists()) {
      const content = await file.text();
      return parseInt(content, 10) || 0;
    }
  } catch {
    // Ignore errors
  }
  return 0;
}

async function setLastUpdateCheck(): Promise<void> {
  const paths = getPaths();
  const checkFile = join(paths.cache, "last-update-check");

  try {
    await Bun.$`mkdir -p ${paths.cache}`.quiet();
    await Bun.write(checkFile, String(Date.now()));
  } catch {
    // Ignore errors
  }
}

async function downloadBinary(url: string, destPath: string): Promise<void> {
  const res = await fetch(url, { signal: AbortSignal.timeout(60000) });

  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }

  const buffer = await res.arrayBuffer();
  const tempPath = `${destPath}.tmp`;

  await Bun.write(tempPath, buffer);
  await Bun.$`chmod +x ${tempPath}`.quiet();
  await Bun.$`mv ${tempPath} ${destPath}`.quiet();
}

export async function updateCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      check: { type: "boolean", short: "c" },
      force: { type: "boolean", short: "f" },
    },
  });

  const checkOnly = values.check ?? false;
  const force = values.force ?? false;

  log.info("update", "Checking for updates", { checkOnly, force });

  const release = await getLatestRelease();

  if (!release) {
    console.log("Could not check for updates. Try again later.");
    return;
  }

  const latestVersion = release.tag_name;
  const currentVersion = getCurrentVersion();

  console.log(`Current version: ${currentVersion}`);
  console.log(`Latest version:  ${latestVersion}`);

  if (currentVersion === latestVersion && !force) {
    console.log("\nYou're running the latest version.");
    await setLastUpdateCheck();
    return;
  }

  if (checkOnly) {
    if (currentVersion !== latestVersion) {
      console.log(`\nUpdate available! Run 'ccmemory update' to install.`);
    }
    return;
  }

  // Find the right binary for this platform
  const platform = detectPlatform();
  const assetName = `ccmemory-${platform}${platform.startsWith("windows") ? ".exe" : ""}`;
  const asset = release.assets.find((a) => a.name === assetName);

  if (!asset) {
    console.error(`No binary available for ${platform}`);
    console.error(`Available assets: ${release.assets.map((a) => a.name).join(", ")}`);
    process.exit(1);
  }

  console.log(`\nDownloading ${assetName}...`);

  const binaryPath = process.argv[0] ?? "";

  if (!binaryPath || binaryPath.includes("bun")) {
    console.error("Cannot determine binary path. Running in development mode?");
    console.error("Use the install script instead:");
    console.error("  curl -fsSL https://raw.githubusercontent.com/JoeyEamigh/ccmemory/main/scripts/install.sh | bash");
    process.exit(1);
  }

  try {
    await downloadBinary(asset.browser_download_url, binaryPath);

    // Update version file
    const versionFile = join(dirname(binaryPath), ".ccmemory-version");
    await Bun.write(versionFile, latestVersion);

    await setLastUpdateCheck();

    console.log(`\nSuccessfully updated to ${latestVersion}!`);
  } catch (err) {
    log.error("update", "Update failed", {
      error: err instanceof Error ? err.message : String(err),
    });
    console.error(`\nUpdate failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }
}

export async function checkForUpdatesInBackground(): Promise<void> {
  const lastCheck = await getLastUpdateCheck();
  const now = Date.now();

  if (now - lastCheck < UPDATE_CHECK_INTERVAL) {
    log.debug("update", "Skipping update check (checked recently)", {
      lastCheck: new Date(lastCheck).toISOString(),
    });
    return;
  }

  log.debug("update", "Running background update check");

  const release = await getLatestRelease();
  if (!release) return;

  const latestVersion = release.tag_name;
  const currentVersion = getCurrentVersion();

  await setLastUpdateCheck();

  if (currentVersion !== latestVersion && currentVersion !== "unknown") {
    log.info("update", "Update available", { current: currentVersion, latest: latestVersion });
    // In the future, could write to a notification file that the WebUI shows
  }
}
