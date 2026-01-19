import { createDatabase, setDatabase } from "../src/db/database.js";
import { createMemoryStore } from "../src/services/memory/store.js";
import { getOrCreateProject } from "../src/services/project.js";
import { getOrCreateSession } from "../src/services/memory/sessions.js";
import { join } from "node:path";

const dataDir = process.env["CCMEMORY_DATA_DIR"] ?? `${process.env.HOME}/.local/share/ccmemory`;
const dbPath = join(dataDir, "ccmemory.db");

async function seed() {
  const db = await createDatabase(dbPath);
  setDatabase(db);
  
  const project1 = await getOrCreateProject("/home/joey/Documents/ccmemory");
  const project2 = await getOrCreateProject("/home/joey/Documents/other-project");
  
  const session1 = await getOrCreateSession("test-session-1", project1.id);
  const session2 = await getOrCreateSession("test-session-2", project2.id);
  
  const store = createMemoryStore();
  
  // Create memories for project 1
  await store.create(
    { content: "The database schema uses libSQL with migrations for version control", sector: "semantic", tier: "project" },
    project1.id,
    session1.id
  );
  await store.create(
    { content: "Memory search uses hybrid FTS5 and vector search for best results", sector: "semantic", tier: "project" },
    project1.id,
    session1.id
  );
  await store.create(
    { content: "User ran bun test and all 332 tests passed successfully", sector: "episodic", tier: "session" },
    project1.id,
    session1.id
  );
  await store.create(
    { content: "To build the project run bun run build:all from the root directory", sector: "procedural", tier: "project" },
    project1.id,
    session1.id
  );
  
  // Create memories for project 2
  await store.create(
    { content: "This is a test memory in the other project about React components", sector: "semantic", tier: "project" },
    project2.id,
    session2.id
  );
  
  console.log("Seeded test data successfully!");
  process.exit(0);
}

seed().catch(console.error);
