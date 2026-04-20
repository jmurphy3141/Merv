# Merv — Jarvis-like Personal AI Companion

**v0.1.0 — Phase 0: Family Assistant + Telegram Bot**

Merv is a proactive, context-aware AI companion for busy families. It prioritizes family logistics above everything else, then project management, then engineering. Built on LangGraph for stateful multi-agent orchestration.

---

## Priority Order (Locked)

1. **Family Assistant** — calendars, email triage, rain-out alerts, Henry's scheduling constraints, morning brief
2. **Project Manager** — task tracking, reminders, status reports *(Phase 1)*
3. **Engineering Team** — autonomous multi-agent coding *(Phase 2)*

---

## Phase 0 Features

- **Morning Brief via Telegram** — sent automatically at 7 AM (configurable):
  - Email roundup (urgent + family-related first)
  - All family calendars for the day
  - Weather + rain-out risk for soccer/baseball/lacrosse
  - Henry's scheduling constraint violations flagged proactively
- **LangGraph orchestration** — stateful, cyclical, debuggable workflow
- **Hybrid skill model** — fast keyword routing + LLM classification fallback
- **Jarvis personality** — proactive, helpful, witty when appropriate
- **Google Calendar + Gmail integration** (mocked in dev)
- **Open-Meteo weather** (no API key needed)
- **JSON-backed memory** (family profile, rain-out history, notes)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/jmurphy3141/Merv.git
cd Merv
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and TELEGRAM_BOT_TOKEN at minimum

# 3. Run tests (all pass without any credentials)
pytest tests/ -v

# 4. Start the Telegram bot
merv bot

# 5. Or generate a morning brief in the terminal
merv brief

# 6. Interactive CLI chat
merv chat
```

---

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key (claude-sonnet-4-6 default) |
| `TELEGRAM_BOT_TOKEN` | Yes (for bot) | From @BotFather |
| `TELEGRAM_ALLOWED_CHAT_IDS` | Recommended | Comma-separated chat IDs (security) |
| `GOOGLE_CREDENTIALS_FILE` | For real data | OAuth2 credentials JSON |
| `MORNING_ROUTINE_CHAT_ID` | For auto-brief | Telegram chat to send morning brief |
| `FAMILY_TIMEZONE` | Optional | Default: `America/New_York` |

**Without credentials**: all services run in mock mode with realistic sample data.

---

## Architecture

```
[Telegram / CLI]
       ↓
[LangGraph Supervisor]
       ↓
[Intent Router] ──keyword fast path──> classification in ~1ms
       ↓               ↓ LLM fallback for ambiguous input
       ├── morning_routine  → MorningRoutineSkill
       ├── family_skill     → FamilyAssistantSkill
       ├── project_skill    → placeholder (Phase 1)
       ├── engineering_skill→ placeholder (Phase 2)
       └── conversation     → direct LLM response
```

**Key modules:**
- `merv/core/graph.py` — LangGraph workflow assembly
- `merv/core/state.py` — `MervState` (single shared state object)
- `merv/core/intent_router.py` — fast keyword + LLM classification
- `merv/core/personality.py` — Jarvis system prompt
- `merv/skills/morning_routine.py` — flagship feature
- `merv/skills/calendar_service.py` — Google Calendar + Henry constraints
- `merv/skills/email_service.py` — Gmail triage
- `merv/skills/weather_service.py` — Open-Meteo rain-out detection
- `merv/interfaces/telegram_bot.py` — Telegram bot with APScheduler
- `merv/memory/store.py` — JSON-backed persistent memory

---

## Testing

```bash
# All tests (115 total, no credentials required)
pytest tests/ -v

# Unit tests only (94 tests, ~1s)
pytest tests/unit/ -v

# Integration tests (21 tests)
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=merv --cov-report=term-missing
```

---

## Sample Morning Brief (no-LLM fallback)

```
☀️ Good morning! Here's your brief for Monday, April 20.

🌤️ Weather: Light rain expected
   High: 54°F  Low: 46°F
⚠️ Rain expected today — outdoor activities at risk!

⚠️ Rain-out alerts:
   • Henry — Soccer Practice — may be canceled

🔴 Urgent emails:
   • Soccer Practice CANCELED — Rain

👨‍👩‍👦 Family emails:
   • Henry appointment reminder — Dr. Kim

📅 Today's schedule (3 events):
   • Henry — Pediatrician Appointment @ 09:00 AM [TENTATIVE]
   • Henry — Soccer Practice @ 04:00 PM
   • Family Dinner @ 06:30 PM

— Merv 🤖
```

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| **0** | ✅ Complete | Family assistant, Telegram bot, morning routine, LangGraph core, 115 tests |
| **1** | Planned | Project manager, task tracking, multi-calendar coordination |
| **2** | Planned | Engineering team sub-agents, LangGraph worker pool, self-improvement |
| **3** | Planned | Voice layer (Hudson option), smart home, vector memory |

---

*Merv v2.0 Specification is the single source of truth. All agents reference it.*
