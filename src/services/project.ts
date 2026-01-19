import { getDatabase } from "../db/database.js";
import { log } from "../utils/log.js";
import { basename } from "path";

export type Project = {
  id: string;
  path: string;
  name: string;
  settingsJson?: string;
  createdAt: number;
  updatedAt: number;
};

function rowToProject(row: Record<string, unknown>): Project {
  return {
    id: String(row["id"]),
    path: String(row["path"]),
    name: String(row["name"]),
    settingsJson: row["settings_json"]
      ? String(row["settings_json"])
      : undefined,
    createdAt: Number(row["created_at"]),
    updatedAt: Number(row["updated_at"]),
  };
}

export async function getOrCreateProject(cwd: string): Promise<Project> {
  const db = await getDatabase();

  const existing = await db.execute(
    "SELECT * FROM projects WHERE path = ?",
    [cwd]
  );

  if (existing.rows.length > 0 && existing.rows[0]) {
    return rowToProject(existing.rows[0]);
  }

  const now = Date.now();
  const id = crypto.randomUUID();
  const name = basename(cwd);

  await db.execute(
    `INSERT INTO projects (id, path, name, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)`,
    [id, cwd, name, now, now]
  );

  log.info("project", "Created project", { id, path: cwd, name });

  return {
    id,
    path: cwd,
    name,
    createdAt: now,
    updatedAt: now,
  };
}

export async function getProjectById(id: string): Promise<Project | null> {
  const db = await getDatabase();
  const result = await db.execute("SELECT * FROM projects WHERE id = ?", [id]);

  if (result.rows.length === 0) return null;
  const row = result.rows[0];
  if (!row) return null;

  return rowToProject(row);
}

export async function getProjectByPath(path: string): Promise<Project | null> {
  const db = await getDatabase();
  const result = await db.execute("SELECT * FROM projects WHERE path = ?", [path]);

  if (result.rows.length === 0) return null;
  const row = result.rows[0];
  if (!row) return null;

  return rowToProject(row);
}

export async function listProjects(): Promise<Project[]> {
  const db = await getDatabase();
  const result = await db.execute(
    "SELECT * FROM projects ORDER BY updated_at DESC"
  );
  return result.rows.map(rowToProject);
}

export async function updateProject(
  id: string,
  updates: { name?: string; settingsJson?: string }
): Promise<Project> {
  const db = await getDatabase();
  const now = Date.now();

  const setClauses: string[] = ["updated_at = ?"];
  const args: (string | number)[] = [now];

  if (updates.name !== undefined) {
    setClauses.push("name = ?");
    args.push(updates.name);
  }

  if (updates.settingsJson !== undefined) {
    setClauses.push("settings_json = ?");
    args.push(updates.settingsJson);
  }

  args.push(id);

  await db.execute(
    `UPDATE projects SET ${setClauses.join(", ")} WHERE id = ?`,
    args
  );

  const project = await getProjectById(id);
  if (!project) throw new Error("Project not found after update");

  return project;
}
