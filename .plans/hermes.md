# Plan: Replace Custom Chatbot Agent with Hermes Agent

## Context

The project has a custom chatbot/agent subsystem spanning ~11,700 lines (backend + frontend). It implements LLM provider routing (5 providers), a 20-tool ReAct loop, deep research pipeline, SSE streaming, and a full React chat UI. This is too much effort to maintain. The plan is to replace it with **Hermes Agent** (NousResearch), which provides agent orchestration, LLM routing, tool calling, and learning capabilities out of the box.

**Decisions:**
- **UI**: Both full-page route + floating popup (FAB button) on all pages
- **Research**: Drop the deep research pipeline entirely
- **Tools**: Extend the existing MCP server to expose all 20 tools
- **LLMService**: KEEP — used by theme extraction, merging, and taxonomy (not chatbot-only)

---

## Phase 0: Delete Dead Code (Quick Win)

Delete legacy files that are not called anywhere in the current code path. Risk-free, reduces noise for subsequent phases.

**Delete:**
- `backend/app/services/chatbot/planning_agent.py` (228 lines — dead code)
- `backend/app/services/chatbot/action_agent.py` (242 lines — dead code)
- `backend/app/services/chatbot/validation_agent.py` (252 lines — dead code)
- `backend/app/services/chatbot/answer_agent.py` (235 lines — dead code)
- `backend/app/services/chatbot/research/groq_retry.py` (271 lines — redundant with LLMService)

**Verify:** Check for any remaining imports of these modules and remove them. The only live reference is `ActionAgent` in the `GET /chatbot/tools` endpoint — update that endpoint to source tool listings from `tool_definitions.py` directly, or remove it (it will be deleted in Phase 3 anyway).

**~1,228 lines removed, standalone PR.**

---

## Phase 1: Extend MCP Server (Backend)

### 1a. Add chatbot tool implementations to the MCP server

**File**: `backend/app/interfaces/mcp/market_copilot.py`

The existing `MarketCopilotService` has 12 tools using `ToolSpec` dataclass pattern (name, description, args_model, handler). Add the remaining chatbot tools following the same **thin adapter handler** pattern:

**Adapter pattern** (existing, follow exactly):
1. Create a Pydantic args model in `models.py`
2. Write a handler method in `MarketCopilotService` that:
   - Instantiates the existing tool class (e.g., `YFinanceTools()`)
   - Calls the class method with validated args
   - Wraps the result in a `ToolEnvelope` (summary, facts, citations, freshness)
3. Register as a `ToolSpec(name, description, args_model, handler)`

Each adapter is ~15-25 lines. This is NOT a reimplementation — the existing tool classes do the real work.

**Tools to add** (import tool classes from existing code):
- `yfinance_quote`, `yfinance_fundamentals`, `yfinance_history`, `yfinance_earnings`, `compare_stocks` — via `YFinanceTools` from `services/chatbot/tools/yfinance_tools.py`
- `web_search`, `search_news`, `search_finance` — via `WebSearchTool` from `services/chatbot/tools/web_search.py`
- `get_sec_10k`, `read_ir_pdf` — via `DocumentTools` from `services/chatbot/tools/document_tools.py`
- `research_theme`, `discover_themes` — via `DatabaseTools` from `services/chatbot/tools/database_tools.py`

**Schema translation** — existing OpenAI format → MCP format:
```python
# OpenAI format (in tool_definitions.py):
{"type": "function", "function": {"name": "X", "parameters": {...}}}

# MCP format (target — Pydantic args model):
class XArgs(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
```
Reference: `backend/app/services/chatbot/tool_definitions.py`

### 1b. Organize Pydantic args models

**File**: `backend/app/interfaces/mcp/models.py`

Adding ~12 new args models to the existing ~12. Keep in a single file but add clear section headers:

```python
# === Market Copilot Tools (existing) ===
class MarketOverviewArgs(...): ...

# === Market Data Tools (yfinance, web search) ===
class YFinanceQuoteArgs(...): ...

# === Database Query Tools (scans, themes, breadth) ===
class ScanResultsArgs(...): ...

# === Document Tools (SEC, PDF) ===
class SecFilingArgs(...): ...
```

**Additional DB models required** by the new tools (already exist, just need imports in market_copilot.py):
- `Scan`, `ScanResult` from `app.models.scan_result`
- `StockPrice`, `StockFundamental` from `app.models.stock`
- `ThemeMention`, `ContentItem` from `app.models.theme`
- `DocumentCache`, `DocumentChunk` from `app.models.document_cache`

### 1c. Extend MCP test suites

**Files**:
- Extend: `backend/tests/unit/test_mcp_market_copilot.py` — add test cases for each new tool
- Extend: `backend/tests/integration/test_mcp_server_integration.py` — update `tools/list` count assertion
- Update: golden snapshots via `make golden-update`

### 1d. Update Hermes config and skill

**File**: `integrations/hermes/README.md` (update docs with new tool list)
**File**: `integrations/hermes/skills/market-copilot/SKILL.md` (add workflow guidance for new tools)

Ensure both stdio and HTTP transport configurations are documented:
```yaml
# Local dev (stdio — same machine):
mcp_servers:
  stockscreen:
    command: "/path/to/venv/bin/python"
    args: ["-m", "app.interfaces.mcp.server"]

# Docker (HTTP — cross-container):
mcp_servers:
  stockscreen:
    url: "http://backend:8000/mcp"
```

**Key files**:
- Extend: `backend/app/interfaces/mcp/market_copilot.py`
- Extend: `backend/app/interfaces/mcp/models.py`
- Already exists: `backend/app/interfaces/mcp/http_transport.py` (HTTP MCP transport for Docker)
- Reuse: `backend/app/services/chatbot/tools/yfinance_tools.py`
- Reuse: `backend/app/services/chatbot/tools/web_search.py`
- Reuse: `backend/app/services/chatbot/tools/document_tools.py`
- Reuse: `backend/app/services/chatbot/tools/database_tools.py`
- Reference: `backend/app/services/chatbot/tool_definitions.py`

---

## Phase 2: Frontend — Hermes Chat Component

### 2a. Create Hermes API adapter

**New file**: `frontend/src/api/hermes.js`

A thin adapter (~150-200 lines) that:
- Talks to hermes gateway at configurable URL (env var `VITE_HERMES_URL`, default `http://localhost:8642`)
- Sends `POST /v1/chat/completions` with `stream: true`
- Parses OpenAI delta SSE format (`choices[0].delta.content`)
- Exposes same callback interface as existing `sendMessageStream` (onChunk, onError, onDone)
- Handles bearer token auth (`API_SERVER_KEY`)
- Includes a `checkHealth()` function: `GET /health` on the gateway

**Reuse pattern from**: `frontend/src/api/chatbot.js:65-145` (fetch + ReadableStream + SSE parsing)

### 2b. Create session context for shared state

**New file**: `frontend/src/contexts/HermesChatContext.jsx`

A React context (~50-80 lines) providing:
- Active session ID (stored in localStorage for persistence across page reloads)
- Message history for the current session
- Gateway health status (via React Query polling of `GET /health`)
- Shared between popup drawer and full-page chat — user sees same conversation in both

Provider wraps children in `Layout.jsx` (same level as existing `RuntimeContext`).

### 2c. Create shared HermesChat component

**New file**: `frontend/src/components/HermesChat/HermesChat.jsx`

A self-contained chat component (~300-400 lines) usable in both full-page and popup contexts:
- Props: `fullPage: boolean` (controls height/layout), `onClose?: () => void`
- Reads session from `HermesChatContext`
- Calls `hermes.js` adapter for streaming
- Renders message bubbles with markdown (simplified from existing `MessageBubble.jsx` — keep markdown + code blocks + tool call accordion, drop citation system)
- Shows tool call indicators (Hermes gateway streams tool calls in OpenAI format)

**Supporting files** (keep minimal):
- `frontend/src/components/HermesChat/MessageBubble.jsx` — simplified message renderer
- `frontend/src/components/HermesChat/index.js` — re-export

### 2d. Full page route

**Modify**: `frontend/src/pages/ChatbotPage.jsx`

Replace entire contents. The new page is thin (~50-80 lines):
```jsx
<Box sx={{ height: 'calc(100vh - 64px)', display: 'flex' }}>
  <HermesChat fullPage />
</Box>
```

Keep the `/chatbot` route in `App.jsx` unchanged.

### 2e. Floating popup (FAB + Drawer)

**Modify**: `frontend/src/components/Layout/Layout.jsx`

Add after line 197 (before closing `</Box>`):
- A `Fab` button (bottom-right, position: fixed) with chat icon
- A MUI `Drawer` (anchor: right, ~400px wide) containing `<HermesChat onClose={...} />`
- Local state: `chatOpen` boolean
- **Only render when:**
  - `features.chatbot` is enabled (existing feature flag)
  - Gateway is healthy (from `HermesChatContext`)
  - Current route is NOT `/chatbot` (avoid duplicate UI)

```jsx
{features.chatbot && isGatewayHealthy && location.pathname !== '/chatbot' && (
  <>
    <Fab onClick={() => setChatOpen(true)} .../>
    <Drawer open={chatOpen} onClose={() => setChatOpen(false)} anchor="right">
      <HermesChat onClose={() => setChatOpen(false)} />
    </Drawer>
  </>
)}
```

This follows the existing `TaskSettingsModal` pattern in Layout.jsx (line 197).

---

## Phase 3: Delete Old Chatbot Code

### 3a. Backend deletions (~7,000 lines)

Delete entirely:
- `backend/app/services/chatbot/agent_orchestrator.py` (324 lines)
- `backend/app/services/chatbot/tool_agent.py` (403 lines)
- `backend/app/services/chatbot/tool_executor.py` (304 lines)
- `backend/app/services/chatbot/prompts.py` (302 lines)
- `backend/app/services/chatbot/research/` entire directory (remaining files after Phase 0, ~2,330 lines)

Delete API routes (Hermes handles the chat API now):
- `backend/app/api/v1/chatbot.py` (368 lines)
- `backend/app/api/v1/chatbot_folders.py`

**DO NOT delete** (still needed):
- `backend/app/services/llm/` — used by theme extraction, merging, and taxonomy services
- `backend/app/services/chatbot/tools/` — tool implementations used by MCP server (relocated in Phase 3c)
- `backend/app/services/chatbot/tool_definitions.py` — reference schemas
- `backend/app/models/chatbot.py` — DB models (defer table drop)
- `backend/app/schemas/chatbot.py` — Pydantic schemas (defer cleanup)

Update imports:
- `backend/app/api/v1/__init__.py` or router registration in `main.py` — remove chatbot router
- `backend/app/main.py` — remove chatbot route includes

### 3b. Frontend deletions (~2,500 lines)

Delete entirely:
- `frontend/src/components/Chatbot/` — all 10 files (ChatWindow, MessageBubble, ToolSelector, PromptLibrary, ResearchProgressIndicator, ResearchModeToggle, SavePromptDialog, TickerInputDialog, RenameDialog, MessageList)
- `frontend/src/api/chatbot.js` (260 lines)

### 3c. Relocate tool implementations

Move tool files out of the now-gutted `services/chatbot/` directory:

```
backend/app/services/chatbot/tools/database_tools.py  →  backend/app/interfaces/mcp/tools/database_tools.py
backend/app/services/chatbot/tools/yfinance_tools.py  →  backend/app/interfaces/mcp/tools/yfinance_tools.py
backend/app/services/chatbot/tools/web_search.py      →  backend/app/interfaces/mcp/tools/web_search.py
backend/app/services/chatbot/tools/document_tools.py  →  backend/app/interfaces/mcp/tools/document_tools.py
backend/app/services/chatbot/tools/read_url.py        →  backend/app/interfaces/mcp/tools/read_url.py
```

Update imports in `market_copilot.py` to point to new locations.

After relocation, delete `backend/app/services/chatbot/` entirely — clean break, no orphans.

### 3d. Database migration (deferred)

The `chatbot_conversations`, `chatbot_messages`, `chatbot_agent_executions`, `chatbot_folders` tables can be dropped via an Alembic migration. Defer this until we're confident the Hermes setup is stable.

---

## Phase 4: Docker / Infrastructure

### 4a. Add Hermes to Docker Compose

Add a `hermes` service to `docker-compose.yml`:
```yaml
hermes:
  image: nousresearch/hermes-agent:latest
  volumes:
    - ./integrations/hermes:/root/.hermes
    - ./data/hermes:/root/.hermes/data
  environment:
    - API_SERVER_ENABLED=true
    - API_SERVER_PORT=8642
    - API_SERVER_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
  ports:
    - "8642:8642"
  depends_on:
    - backend
```

### 4b. MCP transport: HTTP for Docker

In Docker, Hermes and backend are separate containers — stdio MCP won't work. Use the existing HTTP transport at `backend/app/interfaces/mcp/http_transport.py`:

Hermes config for Docker:
```yaml
mcp_servers:
  stockscreen:
    url: "http://backend:8000/mcp"
```

Ensure `http_transport.py` router is mounted in `main.py` (verify — may already be mounted).

### 4c. Frontend environment variable

Add `VITE_HERMES_URL` to:
- `frontend/.env.development` → `http://localhost:8642`
- `docker-compose.yml` frontend build args
- `.env.docker.example` template

---

## Execution Order

1. **Phase 0** (dead code deletion) — standalone PR, risk-free
2. **Phase 1** (MCP tools + tests) — can be done independently, testable with `hermes` CLI
3. **Phase 2** (Frontend) — depends on Hermes gateway being available
4. **Phase 3** (Deletions + relocations) — only after Phase 2 is validated
5. **Phase 4** (Docker) — parallel with Phase 2

---

## Verification

### Phase 0 verification
```bash
cd backend && source venv/bin/activate
pytest tests/unit/  # no import errors from deleted modules
grep -r "planning_agent\|action_agent\|validation_agent\|answer_agent\|groq_retry" backend/app/ --include="*.py"  # should return nothing
```

### Phase 1: MCP server testing
```bash
cd backend && source venv/bin/activate

# Test tool listing — should return 20+ tools
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python -m app.interfaces.mcp.server

# Test new tool call
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"yfinance_quote","arguments":{"symbol":"AAPL"}}}' | python -m app.interfaces.mcp.server

# Run test suite
pytest tests/unit/test_mcp_market_copilot.py -v
pytest tests/integration/test_mcp_server_integration.py -v
make golden-update  # regenerate snapshots
make gate-5         # golden regression gate
```

### Phase 1: Hermes integration testing
```bash
hermes  # interactive CLI
> what is AAPL trading at?  # triggers mcp_stockscreen_yfinance_quote

hermes gateway
curl http://localhost:8642/health  # verify gateway
```

### Phase 2: Frontend testing
```bash
cd frontend
npm run dev
# 1. Navigate to /chatbot — verify full page chat works
# 2. Navigate to /scan — verify FAB button appears (bottom-right)
# 3. Click FAB — verify drawer opens with chat
# 4. Send a message — verify SSE streaming from hermes gateway
# 5. Start conversation in popup, navigate to /chatbot — verify same session
# 6. Stop hermes gateway — verify FAB disappears gracefully
```

### Phase 3: Cleanup verification
```bash
cd backend && pytest tests/unit/  # no import errors
cd frontend && npm run build      # no missing imports
cd frontend && npm run lint       # clean

# Verify tool files relocated correctly
ls backend/app/interfaces/mcp/tools/  # should contain 5 files
ls backend/app/services/chatbot/      # should not exist
```

---

## Net Impact

| Metric | Before | After |
|---|---|---|
| Backend chatbot code | ~7,900 lines | ~0 (tools relocated to MCP layer) |
| Frontend chatbot code | ~2,500 lines | ~600 lines (HermesChat + adapter + context) |
| LLM provider management | Custom (5 providers, key rotation) | Hermes handles chatbot LLM; theme pipeline keeps LLMService |
| Agent orchestration | Custom ReAct loop | Hermes agent with learning |
| Deep research | Custom 4-phase pipeline | Dropped |
| MCP tools | 12 | 20+ (all chatbot tools exposed) |
| Maintenance burden | High | Low (Hermes community maintains agent layer) |
| New dependency | — | Hermes Agent (gateway process) |
