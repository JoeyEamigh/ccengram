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
- [x] **T001** Create package.json with dependencies
- [x] **T002** Configure TypeScript (tsconfig.json)
- [x] **T003** Set up Bun test configuration
- [x] **T004** Create directory structure

### P1.2: XDG Paths
- [x] **T005** Implement `src/utils/paths.ts`
- [x] **T006** Platform detection (Linux, macOS, Windows)
- [x] **T007** Directory creation utility
- [x] **T008** Write tests for paths (`src/utils/__test__/paths.test.ts`)

### P1.3: Logging
- [x] **T009** Implement `src/utils/log.ts`
- [x] **T010** Log levels (debug, info, warn, error)
- [x] **T011** Structured logging with module + context
- [x] **T012** LOG_LEVEL env var support
- [x] **T013** Write tests for logging (`src/utils/__test__/log.test.ts`)

### P1.4: Database Setup
- [x] **T014** Implement `src/db/database.ts` (libSQL connection)
- [x] **T015** Enable WAL mode and pragmas
- [x] **T016** Implement `src/db/schema.ts` (table definitions)
- [x] **T017** Implement `src/db/migrations.ts`
- [x] **T018** Create initial migration (v1)
- [x] **T019** Write database tests (`src/db/__test__/*.test.ts`)

---

## Phase 2: Embedding Service

### P2.1: Ollama Provider
- [x] **T020** Implement `src/services/embedding/ollama.ts`
- [x] **T021** Availability check (model detection)
- [x] **T022** Dimension detection
- [x] **T023** Batch embedding support
- [x] **T024** Write Ollama provider tests (`src/services/embedding/__test__/ollama.test.ts`)

### P2.2: OpenRouter Provider
- [x] **T025** Implement `src/services/embedding/openrouter.ts`
- [x] **T026** API key management
- [x] **T027** Model dimension mapping
- [x] **T028** Batch embedding support
- [x] **T029** Write OpenRouter provider tests (`src/services/embedding/__test__/openrouter.test.ts`)

### P2.3: Embedding Service
- [x] **T030** Implement `src/services/embedding/index.ts`
- [x] **T031** Provider fallback logic
- [x] **T032** Model registration in database
- [x] **T033** Provider switching
- [x] **T034** Write service integration tests (`src/services/embedding/__test__/service.test.ts`)

---

## Phase 3: Memory System

### P3.1: Memory Sectors
- [x] **T035** Implement `src/services/memory/types.ts`
- [x] **T036** Five-sector model (episodic, semantic, procedural, emotional, reflective)
- [x] **T037** Sector classification patterns
- [x] **T038** Decay rate constants per sector
- [x] **T039** Write sector classification tests (`src/services/memory/__test__/types.test.ts`)

### P3.2: Deduplication
- [x] **T040** Implement `src/services/memory/dedup.ts`
- [x] **T041** Simhash computation (64-bit)
- [x] **T042** Hamming distance calculation
- [x] **T043** Duplicate detection with threshold
- [x] **T044** Write deduplication tests (`src/services/memory/__test__/dedup.test.ts`)

### P3.3: Memory Relationships
- [x] **T045** Implement `src/services/memory/relationships.ts`
- [x] **T046** Relationship types (SUPERSEDES, CONTRADICTS, RELATED_TO, BUILDS_ON)
- [x] **T047** Create relationship with validation
- [x] **T048** Get related memories
- [x] **T049** Write relationship tests (`src/services/memory/__test__/relationships.test.ts`)

### P3.4: Memory Store
- [x] **T050** Implement `src/services/memory/store.ts`
- [x] **T051** Create memory with auto-classification
- [x] **T052** Bi-temporal timestamps (valid_from, valid_until)
- [x] **T053** Deduplication on create (boost existing)
- [x] **T054** Concept extraction
- [x] **T055** Get, update, soft delete operations
- [x] **T056** List with filtering (sector, tier, salience)
- [x] **T057** Touch (access tracking)
- [x] **T058** Reinforce with diminishing returns
- [x] **T059** De-emphasize (reduce salience)
- [x] **T060** Write memory store tests (`src/services/memory/__test__/store.test.ts`)

### P3.5: Session Tracking
- [x] **T061** Implement `src/services/memory/sessions.ts`
- [x] **T062** Session creation with metadata
- [x] **T063** Track memory usage (created, recalled, updated, reinforced)
- [x] **T064** End session with summary
- [x] **T065** Promote session tier memories
- [x] **T066** Write session tests (`src/services/memory/__test__/sessions.test.ts`)

### P3.6: Salience Decay
- [x] **T067** Implement `src/services/memory/decay.ts`
- [x] **T068** Decay calculation by sector
- [x] **T069** Access count protection
- [x] **T070** Salience boost function
- [x] **T071** Background decay process
- [x] **T072** Write decay tests (`src/services/memory/__test__/decay.test.ts`)

---

## Phase 4: Search System

### P4.1: FTS5 Search
- [x] **T073** Implement `src/services/search/fts.ts`
- [x] **T074** Query preparation (prefix matching)
- [x] **T075** Snippet extraction
- [x] **T076** Project filtering
- [x] **T077** Write FTS tests (`src/services/search/__test__/fts.test.ts`)

### P4.2: Vector Search
- [x] **T078** Implement `src/services/search/vector.ts`
- [x] **T079** Query embedding
- [x] **T080** vector_top_k usage
- [x] **T081** Model-aware search
- [x] **T082** Write vector search tests (`src/services/search/__test__/vector.test.ts`)

### P4.3: Hybrid Search & Ranking
- [x] **T083** Implement `src/services/search/ranking.ts`
- [x] **T084** Score computation with weights
- [x] **T085** Sector-specific boosts
- [x] **T086** Implement `src/services/search/hybrid.ts`
- [x] **T087** Result merging
- [x] **T088** Filtering (sector, tier, salience)
- [x] **T089** Salience boost on retrieval
- [x] **T090** Session context in results (session_id, agent_name)
- [x] **T091** Timeline function with session grouping
- [x] **T092** Write hybrid search tests (`src/services/search/__test__/hybrid.test.ts`)

---

## Phase 5: Documents

### P5.1: Chunking
- [x] **T093** Implement `src/services/documents/chunk.ts`
- [x] **T094** Sentence/paragraph aware splitting
- [x] **T095** Overlap handling
- [x] **T096** Offset tracking
- [x] **T097** Write chunking tests (`src/services/documents/__test__/chunk.test.ts`)

### P5.2: Document Service
- [x] **T098** Implement `src/services/documents/ingest.ts`
- [x] **T099** File path ingestion
- [x] **T100** URL fetching
- [x] **T101** Raw content ingestion
- [x] **T102** Title extraction (markdown H1)
- [x] **T103** Checksum for change detection
- [x] **T104** Chunk embedding
- [x] **T105** Document search
- [x] **T106** Update detection
- [x] **T107** Write document tests (`src/services/documents/__test__/ingest.test.ts`)

---

## Phase 6: Claude Code Plugin

### P6.1: Plugin Configuration
- [x] **T108** Create `plugin/.claude-plugin/plugin.json`
- [x] **T109** Create `plugin/hooks/hooks.json`
- [x] **T110** Create `plugin/.mcp.json`

### P6.2: Hook Scripts
- [x] **T111** Implement `scripts/capture.ts` (PostToolUse)
- [x] **T112** Tool observation formatting (sector: episodic)
- [x] **T113** File path extraction
- [x] **T114** Size limit handling
- [x] **T115** Implement `scripts/summarize.ts` (Stop)
- [x] **T116** SDK agent integration
- [x] **T117** Summary prompt (sector: reflective)
- [x] **T118** AbortController handling
- [x] **T119** Implement `scripts/cleanup.ts` (SessionEnd)
- [x] **T120** Session tier promotion
- [x] **T121** Write hook tests (`scripts/*.test.ts`)

### P6.3: MCP Server - Core Tools
- [x] **T122** Implement `src/mcp/server.ts`
- [x] **T123** memory_search tool (sector/tier filtering)
- [x] **T124** memory_timeline tool (session context)
- [x] **T125** memory_add tool (sector classification)
- [x] **T126** docs_search tool
- [x] **T127** docs_ingest tool
- [x] **T128** Project detection (CLAUDE_PROJECT_DIR)

### P6.4: MCP Server - Memory Management Tools
- [x] **T129** memory_reinforce tool (increase salience)
- [x] **T130** memory_deemphasize tool (reduce salience)
- [x] **T131** memory_delete tool (soft delete)
- [x] **T132** memory_supersede tool (create relationship + invalidate)
- [x] **T133** Write MCP server tests (`src/mcp/__test__/server.test.ts`)

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
