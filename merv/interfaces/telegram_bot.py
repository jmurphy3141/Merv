"""Telegram Bot interface — primary typing interface for Merv.

Handles:
- Receiving messages and routing through MervGraph
- Sending the morning brief (triggered by scheduler)
- /start, /brief, /help, /status commands
- Chat ID allowlist enforcement (security)
- Graceful error handling with user-friendly messages
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from merv.config.settings import get_settings
from merv.core.graph import MervGraph, build_merv_graph

logger = logging.getLogger(__name__)


class MervTelegramBot:
    """
    Telegram bot that wraps MervGraph.

    Usage:
        bot = MervTelegramBot()
        await bot.run()
    """

    def __init__(self, graph: Optional[MervGraph] = None):
        self._settings = get_settings()
        self._graph = graph  # injected for testing; built lazily otherwise
        self._app: Optional[Application] = None

    @property
    def graph(self) -> MervGraph:
        if self._graph is None:
            self._graph = build_merv_graph()
        return self._graph

    def _build_app(self) -> Application:
        token = self._settings.telegram_bot_token
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")

        app = Application.builder().token(token).build()

        # Commands
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("brief", self._cmd_brief))
        app.add_handler(CommandHandler("status", self._cmd_status))

        # All other text messages
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        # Error handler
        app.add_error_handler(self._error_handler)

        return app

    # ── Auth guard ─────────────────────────────────────────────────────────────

    def _is_allowed(self, chat_id: int) -> bool:
        allowed = self._settings.allowed_chat_ids
        if not allowed:
            logger.warning("No TELEGRAM_ALLOWED_CHAT_IDS set — allowing all chats (insecure!)")
            return True
        return chat_id in allowed

    async def _check_auth(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            logger.warning("Unauthorized access attempt from chat_id=%s", chat_id)
            await update.message.reply_text(
                "Sorry, I don't recognize you. Ask the family admin to add your chat ID."
            )
            return False
        return True

    # ── Command handlers ───────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update, context):
            return
        name = update.effective_user.first_name or "there"
        await update.message.reply_text(
            f"Hello {name}! I'm Merv, your personal AI companion.\n\n"
            "I can help you with:\n"
            "• 📅 Family schedule management\n"
            "• 📧 Email triage and summaries\n"
            "• ☀️ Morning briefings\n"
            "• ⚽ Rain-out alerts for outdoor sports\n\n"
            "Just talk to me naturally, or use /brief for your morning summary.\n"
            "Type /help to see all commands.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update, context):
            return
        await update.message.reply_text(
            "*Merv Commands*\n\n"
            "/brief — Morning brief: emails + schedule + weather\n"
            "/status — Check Merv's system status\n"
            "/help — This message\n\n"
            "*Just talk to me naturally:*\n"
            "• \"What's on the schedule today?\"\n"
            "• \"Are Henry's appointments OK this week?\"\n"
            "• \"Any rain-outs coming up?\"\n"
            "• \"What's in my inbox?\"\n\n"
            "*Merv note:* I prioritize family first, projects second, "
            "engineering tasks third.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_brief(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update, context):
            return
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        try:
            state = await self.graph.process(
                "Give me my morning brief",
                user_id=str(update.effective_user.id),
            )
            response = state.current_response or "I couldn't assemble the brief — check my logs."
        except Exception as e:
            logger.exception("Error generating brief")
            response = f"⚠️ Error generating brief: {e}"

        await self._send_long_message(update, response)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update, context):
            return
        settings = self._settings
        status_lines = [
            "✅ *Merv Status*",
            f"• Version: 0.1.0 (Phase 0)",
            f"• LLM: {'✅ Connected' if settings.anthropic_api_key else '❌ No API key'}",
            f"• Google Calendar: {'✅ Configured' if settings.google_calendar_ids != 'primary' else '⚠️ Using primary only'}",
            f"• Voice: {'✅ Enabled' if settings.voice_enabled else '⏸ Disabled (Phase 2)'}",
            f"• Morning brief: {settings.morning_routine_hour:02d}:{settings.morning_routine_minute:02d} {settings.family_timezone}",
        ]
        await update.message.reply_text("\n".join(status_lines), parse_mode=ParseMode.MARKDOWN)

    # ── Message handler ────────────────────────────────────────────────────────

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update, context):
            return

        user_text = update.message.text
        chat_id = update.effective_chat.id
        logger.info("Message from chat_id=%s: %s", chat_id, user_text[:80])

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        try:
            state = await self.graph.process(
                user_text,
                user_id=str(update.effective_user.id),
            )
            response = state.current_response or "Hmm, I didn't get a response. Try again?"
        except Exception as e:
            logger.exception("Error processing message")
            response = (
                f"⚠️ Something went wrong on my end: {type(e).__name__}. "
                "I've logged it — please try again."
            )

        await self._send_long_message(update, response)

    # ── Utility ────────────────────────────────────────────────────────────────

    async def _send_long_message(self, update: Update, text: str) -> None:
        """Split messages >4096 chars (Telegram limit) into chunks."""
        MAX_LEN = 4096
        if len(text) <= MAX_LEN:
            await update.message.reply_text(text)
            return
        chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]
        for chunk in chunks:
            await update.message.reply_text(chunk)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Telegram error: %s", context.error, exc_info=context.error)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def send_brief_to_chat(self, chat_id: int) -> None:
        """
        Push the morning brief to a specific chat.
        Called by the APScheduler morning routine job.
        """
        if self._app is None:
            logger.error("Bot app not initialized — cannot send brief")
            return
        try:
            state = await self.graph.process("Give me my morning brief")
            response = state.current_response or "Could not generate morning brief."
            MAX_LEN = 4096
            chunks = [response[i:i + MAX_LEN] for i in range(0, len(response), MAX_LEN)]
            for chunk in chunks:
                await self._app.bot.send_message(chat_id=chat_id, text=chunk)
            logger.info("Morning brief sent to chat_id=%s", chat_id)
        except Exception as e:
            logger.exception("Failed to send morning brief to chat_id=%s", chat_id)

    def run(self) -> None:
        """Start the bot with polling (blocking). Use in production."""
        self._app = self._build_app()
        self._setup_scheduler()
        logger.info("Starting Merv Telegram bot (polling)...")
        self._app.run_polling(allowed_updates=Update.ALL_TYPES)

    def _setup_scheduler(self) -> None:
        """Register the morning routine job with APScheduler via post_init hook."""
        settings = self._settings
        if not settings.morning_routine_chat_id:
            logger.info("MORNING_ROUTINE_CHAT_ID not set — morning brief scheduler disabled")
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._morning_brief_job,
            CronTrigger(
                hour=settings.morning_routine_hour,
                minute=settings.morning_routine_minute,
                timezone=settings.family_timezone,
            ),
            id="morning_brief",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "Morning brief scheduled at %02d:%02d %s → chat_id=%s",
            settings.morning_routine_hour,
            settings.morning_routine_minute,
            settings.family_timezone,
            settings.morning_routine_chat_id,
        )

    async def _morning_brief_job(self) -> None:
        settings = self._settings
        if settings.morning_routine_chat_id:
            await self.send_brief_to_chat(settings.morning_routine_chat_id)
