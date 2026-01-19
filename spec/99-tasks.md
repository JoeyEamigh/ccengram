# CCMemory Master Task List

**⚠️ CRITICAL: UPDATE THIS FILE AS YOU WORK ⚠️**

Before starting ANY task:
1. Find the next `[ ]` task
2. Change it to `[~]` IMMEDIATELY
3. Save and commit: `git commit -am "WIP: T0XX in progress"`

After completing a task:
1. Change `[~]` to `[x]`
2. Save and commit: `git commit -am "Complete T0XX: description"`

**If you don't mark progress, the next session won't know what's done!**

---

## Status Legend
- [ ] Not started
- [~] In progress (MARK THIS BEFORE YOU START)
- [x] Completed (MARK THIS WHEN TESTS PASS)

---

## Phase 1: Core Infrastructure

### P1.1: Project Setup
- [ ] **T001** Create package.json with dependencies
- [ ] **T002** Configure TypeScript (tsconfig.json)
- [ ] **T003** Set up Bun test configuration
- [ ] **T004** Create directory structure

### P1.2: XDG Paths
- [ ] **T005** Implement `src/utils/paths.ts`
- [ ] **T006** Platform detection (Linux, macOS, Windows)
- [ ] **T007** Directory creation utility
- [ ] **T008** Write tests for paths (`src/utils/__test__/paths.test.ts`)

### P1.3: Logging
- [ ] **T009** Implement `src/utils/log.ts`
- [ ] **T010** Log levels (debug, info, warn, error)
- [ ] **T011** Structured logging with module + context
- [ ] **T012** LOG_LEVEL env var support
- [ ] **T013** Write tests for logging (`src/utils/__test__/log.test.ts`)

### P1.4: Database Setup
- [ ] **T014** Implement `src/db/database.ts` (libSQL connection)
- [ ] **T015** Enable WAL mode and pragmas
- [ ] **T016** Implement `src/db/schema.ts` (table definitions)
- [ ] **T017** Implement `src/db/migrations.ts`
- [ ] **T018** Create initial migration (v1)
- [ ] **T019** Write database tests (`src/db/__test__/*.test.ts`)

---

## Phase 2: Embedding Service

### P2.1: Ollama Provider
- [ ] **T020** Implement `src/services/embedding/ollama.ts`
- [ ] **T021** Availability check (model detection)
- [ ] **T022** Dimension detection
- [ ] **T023** Batch embedding support
- [ ] **T024** Write Ollama provider tests (`src/services/embedding/__test__/ollama.test.ts`)

### P2.2: OpenRouter Provider
- [ ] **T025** Implement `src/services/embedding/openrouter.ts`
- [ ] **T026** API key management
- [ ] **T027** Model dimension mapping
- [ ] **T028** Batch embedding support
- [ ] **T029** Write OpenRouter provider tests (`src/services/embedding/__test__/openrouter.test.ts`)

### P2.3: Embedding Service
- [ ] **T030** Implement `src/services/embedding/index.ts`
- [ ] **T031** Provider fallback logic
- [ ] **T032** Model registration in database
- [ ] **T033** Provider switching
- [ ] **T034** Write service integration tests (`src/services/embedding/__test__/service.test.ts`)

---

## Phase 3: Memory System

### P3.1: Memory Sectors
- [ ] **T035** Implement `src/services/memory/types.ts`
- [ ] **T036** Five-sector model (episodic, semantic, procedural, emotional, reflective)
- [ ] **T037** Sector classification patterns
- [ ] **T038** Decay rate constants per sector
- [ ] **T039** Write sector classification tests (`src/services/memory/__test__/types.test.ts`)

### P3.2: Deduplication
- [ ] **T040** Implement `src/services/memory/dedup.ts`
- [ ] **T041** Simhash computation (64-bit)
- [ ] **T042** Hamming distance calculation
- [ ] **T043** Duplicate detection with threshold
- [ ] **T044** Write deduplication tests (`src/services/memory/__test__/dedup.test.ts`)

### P3.3: Memory Relationships
- [ ] **T045** Implement `src/services/memory/relationships.ts`
- [ ] **T046** Relationship types (SUPERSEDES, CONTRADICTS, RELATED_TO, BUILDS_ON)
- [ ] **T047** Create relationship with validation
- [ ] **T048** Get related memories
- [ ] **T049** Write relationship tests (`src/services/memory/__test__/relationships.test.ts`)

### P3.4: Memory Store
- [ ] **T050** Implement `src/services/memory/store.ts`
- [ ] **T051** Create memory with auto-classification
- [ ] **T052** Bi-temporal timestamps (valid_from, valid_until)
- [ ] **T053** Deduplication on create (boost existing)
- [ ] **T054** Concept extraction
- [ ] **T055** Get, update, soft delete operations
- [ ] **T056** List with filtering (sector, tier, salience)
- [ ] **T057** Touch (access tracking)
- [ ] **T058** Reinforce with diminishing returns
- [ ] **T059** De-emphasize (reduce salience)
- [ ] **T060** Write memory store tests (`src/services/memory/__test__/store.test.ts`)

### P3.5: Session Tracking
- [ ] **T061** Implement `src/services/memory/sessions.ts`
- [ ] **T062** Session creation with metadata
- [ ] **T063** Track memory usage (created, recalled, updated, reinforced)
- [ ] **T064** End session with summary
- [ ] **T065** Promote session tier memories
- [ ] **T066** Write session tests (`src/services/memory/__test__/sessions.test.ts`)

### P3.6: Salience Decay
- [ ] **T067** Implement `src/services/memory/decay.ts`
- [ ] **T068** Decay calculation by sector
- [ ] **T069** Access count protection
- [ ] **T070** Salience boost function
- [ ] **T071** Background decay process
- [ ] **T072** Write decay tests (`src/services/memory/__test__/decay.test.ts`)

---

## Phase 4: Search System

### P4.1: FTS5 Search
- [ ] **T073** Implement `src/services/search/fts.ts`
- [ ] **T074** Query preparation (prefix matching)
- [ ] **T075** Snippet extraction
- [ ] **T076** Project filtering
- [ ] **T077** Write FTS tests (`src/services/search/__test__/fts.test.ts`)

### P4.2: Vector Search
- [ ] **T078** Implement `src/services/search/vector.ts`
- [ ] **T079** Query embedding
- [ ] **T080** vector_top_k usage
- [ ] **T081** Model-aware search
- [ ] **T082** Write vector search tests (`src/services/search/__test__/vector.test.ts`)

### P4.3: Hybrid Search & Ranking
- [ ] **T083** Implement `src/services/search/ranking.ts`
- [ ] **T084** Score computation with weights
- [ ] **T085** Sector-specific boosts
- [ ] **T086** Implement `src/services/search/hybrid.ts`
- [ ] **T087** Result merging
- [ ] **T088** Filtering (sector, tier, salience)
- [ ] **T089** Salience boost on retrieval
- [ ] **T090** Session context in results (session_id, agent_name)
- [ ] **T091** Timeline function with session grouping
- [ ] **T092** Write hybrid search tests (`src/services/search/__test__/hybrid.test.ts`)

---

## Phase 5: Documents

### P5.1: Chunking
- [ ] **T093** Implement `src/services/documents/chunk.ts`
- [ ] **T094** Sentence/paragraph aware splitting
- [ ] **T095** Overlap handling
- [ ] **T096** Offset tracking
- [ ] **T097** Write chunking tests (`src/services/documents/__test__/chunk.test.ts`)

### P5.2: Document Service
- [ ] **T098** Implement `src/services/documents/ingest.ts`
- [ ] **T099** File path ingestion
- [ ] **T100** URL fetching
- [ ] **T101** Raw content ingestion
- [ ] **T102** Title extraction (markdown H1)
- [ ] **T103** Checksum for change detection
- [ ] **T104** Chunk embedding
- [ ] **T105** Document search
- [ ] **T106** Update detection
- [ ] **T107** Write document tests (`src/services/documents/__test__/ingest.test.ts`)

---

## Phase 6: Claude Code Plugin

### P6.1: Plugin Configuration
- [ ] **T108** Create `plugin/.claude-plugin/plugin.json`
- [ ] **T109** Create `plugin/hooks/hooks.json`
- [ ] **T110** Create `plugin/.mcp.json`

### P6.2: Hook Scripts
- [ ] **T111** Implement `scripts/capture.ts` (PostToolUse)
- [ ] **T112** Tool observation formatting (sector: episodic)
- [ ] **T113** File path extraction
- [ ] **T114** Size limit handling
- [ ] **T115** Implement `scripts/summarize.ts` (Stop)
- [ ] **T116** SDK agent integration
- [ ] **T117** Summary prompt (sector: reflective)
- [ ] **T118** AbortController handling
- [ ] **T119** Implement `scripts/cleanup.ts` (SessionEnd)
- [ ] **T120** Session tier promotion
- [ ] **T121** Write hook tests (`scripts/*.test.ts`)

### P6.3: MCP Server - Core Tools
- [ ] **T122** Implement `src/mcp/server.ts`
- [ ] **T123** memory_search tool (sector/tier filtering)
- [ ] **T124** memory_timeline tool (session context)
- [ ] **T125** memory_add tool (sector classification)
- [ ] **T126** docs_search tool
- [ ] **T127** docs_ingest tool
- [ ] **T128** Project detection (CLAUDE_PROJECT_DIR)

### P6.4: MCP Server - Memory Management Tools
- [ ] **T129** memory_reinforce tool (increase salience)
- [ ] **T130** memory_deemphasize tool (reduce salience)
- [ ] **T131** memory_delete tool (soft delete)
- [ ] **T132** memory_supersede tool (create relationship + invalidate)
- [ ] **T133** Write MCP server tests (`src/mcp/__test__/server.test.ts`)

---

## Phase 7: CLI

### P7.1: Commands
- [ ] **T134** Implement `src/cli/index.ts` (entry point)
- [ ] **T135** search command (sector filtering)
- [ ] **T136** show command (with relationships)
- [ ] **T137** delete command (soft delete)
- [ ] **T138** archive command
- [ ] **T139** import command
- [ ] **T140** export command (JSON, CSV)
- [ ] **T141** config command
- [ ] **T142** health command
- [ ] **T143** stats command (per sector/tier)
- [ ] **T144** serve command
- [ ] **T145** Write CLI tests (`src/cli/commands/__test__/*.test.ts`)

### P7.2: Build
- [ ] **T146** CLI build script
- [ ] **T147** Executable configuration (bin)

---

## Phase 8: WebUI

### P8.1: Server Core
- [ ] **T148** Implement `src/webui/server.ts` (Bun.serve)
- [ ] **T149** HTTP request routing
- [ ] **T150** WebSocket handler
- [ ] **T151** React SSR setup
- [ ] **T152** Client hydration script bundle

### P8.2: Instance Coordination
- [ ] **T153** Implement `src/webui/coordinator.ts`
- [ ] **T154** Lock file management
- [ ] **T155** Client registration/deregistration
- [ ] **T156** Auto-start with first Claude Code instance
- [ ] **T157** Auto-stop with last instance

### P8.3: API Routes
- [ ] **T158** Implement `src/webui/routes.ts`
- [ ] **T159** Search API (sector, tier filtering)
- [ ] **T160** Memory CRUD APIs (reinforce, deemphasize, delete)
- [ ] **T161** Timeline API (session grouping)
- [ ] **T162** Stats API (per sector/tier)
- [ ] **T163** Config API
- [ ] **T164** Projects API
- [ ] **T165** Active agents API

### P8.4: WebSocket Events
- [ ] **T166** Implement `src/webui/websocket.ts`
- [ ] **T167** memory:created event
- [ ] **T168** memory:updated event
- [ ] **T169** memory:deleted event
- [ ] **T170** session:started event
- [ ] **T171** session:ended event
- [ ] **T172** agent:activity event

### P8.5: React Components
- [ ] **T173** Create `src/webui/components/App.tsx`
- [ ] **T174** Create `src/webui/components/Search.tsx`
- [ ] **T175** Create `src/webui/components/Timeline.tsx`
- [ ] **T176** Create `src/webui/components/AgentView.tsx` (multi-agent)
- [ ] **T177** Create `src/webui/components/SessionCard.tsx`
- [ ] **T178** Create `src/webui/components/MemoryDetail.tsx`
- [ ] **T179** Create `src/webui/components/Settings.tsx`
- [ ] **T180** Create `src/webui/hooks/useWebSocket.ts`
- [ ] **T181** Create `src/webui/hooks/useSearch.ts`
- [ ] **T182** Create `src/webui/styles.css`
- [ ] **T183** Write WebUI tests (`src/webui/__test__/*.test.ts`)

---

## Phase 9: Polish

### P9.1: Integration Testing
- [ ] **T184** Full capture flow test (`tests/integration/capture.test.ts`)
- [ ] **T185** Search quality test (`tests/integration/search.test.ts`)
- [ ] **T186** Concurrent instance test (`tests/integration/concurrent.test.ts`)
- [ ] **T187** Model switching test (`tests/integration/model-switch.test.ts`)
- [ ] **T188** Multi-agent WebSocket test (`tests/integration/websocket.test.ts`)

### P9.2: Documentation
- [ ] **T189** Update CLAUDE.md for project
- [ ] **T190** README.md
- [ ] **T191** Installation instructions

### P9.3: Error Handling
- [ ] **T192** Graceful degradation (no Ollama)
- [ ] **T193** Database recovery
- [ ] **T194** Hook failure isolation
- [ ] **T195** WebSocket reconnection

---

## Task Dependencies

```
T001-T004 (setup) → T005-T013 (paths+log) → T014-T019 (db) → T020-T034 (embedding)
                                                           ↓
T035-T072 (memory) ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
         ↓
T073-T092 (search) → T093-T107 (docs)
         ↓
T108-T133 (plugin) → T134-T147 (cli) → T148-T183 (webui)
                                               ↓
                                      T184-T195 (polish)
```

## Quick Reference

| Phase | Tasks | Spec File |
|-------|-------|-----------|
| 1: Infrastructure | T001-T019 | 01-database.md |
| 2: Embedding | T020-T034 | 02-embedding.md |
| 3: Memory | T035-T072 | 03-memory.md |
| 4: Search | T073-T092 | 04-search.md |
| 5: Documents | T093-T107 | 05-documents.md |
| 6: Plugin | T108-T133 | 06-plugin.md |
| 7: CLI | T134-T147 | 07-cli.md |
| 8: WebUI | T148-T183 | 08-webui.md |
| 9: Polish | T184-T195 | - |

## Loop Execution Notes

**When context is cleared and you're resuming work:**

1. **Read this file FIRST** to find current progress
2. **Find task marked `[~]`** - that's what was in progress
3. **If no `[~]`, find first `[ ]`** - start there
4. **IMMEDIATELY mark `[~]`** before reading specs
5. **Read the relevant spec** for implementation details
6. **Write tests first**, then implement
7. **Run `bun test`** to verify
8. **Mark `[x]` when tests pass**
9. **Commit with task ID**: `git commit -am "Complete T0XX: description"`

**REMEMBER:**
- Mark `[~]` BEFORE you start coding
- Mark `[x]` AFTER tests pass
- Never leave without updating this file
- One task at a time

Each task should be:
- Implementable in one context window
- Testable in isolation
- Not dependent on incomplete tasks

## Type Safety Reminder

**NO `any` TYPES. EVER.**

- Use `unknown` + type guards
- Prefer `type` over `interface` (use interface only for declaration merging or extending)
- Use `as Type` only with defined types
- All exports need return types

## Test Structure

**Colocated unit tests** - next to source files:
- `src/services/memory/store.ts` → `src/services/memory/__test__/store.test.ts`
- `src/mcp/server.ts` → `src/mcp/__test__/server.test.ts`

**Integration tests** - in `tests/` directory:
- `tests/integration/capture.test.ts`
- `tests/integration/search.test.ts`
