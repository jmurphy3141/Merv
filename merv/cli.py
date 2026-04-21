"""Merv CLI — run the bot, trigger morning brief, or start an interactive session."""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from merv.config.settings import get_settings


def _setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def main() -> None:
    """Entry point for the `merv` CLI command."""
    _setup_logging()
    args = sys.argv[1:]

    if not args or args[0] == "bot":
        _run_bot()
    elif args[0] == "brief":
        asyncio.run(_run_brief())
    elif args[0] == "chat":
        asyncio.run(_run_chat())
    else:
        print(
            "Usage:\n"
            "  merv bot      — Start the Telegram bot (default)\n"
            "  merv brief    — Generate and print morning brief\n"
            "  merv chat     — Interactive CLI chat\n"
        )
        sys.exit(1)


def _run_bot() -> None:
    from merv.interfaces.telegram_bot import MervTelegramBot
    bot = MervTelegramBot()
    bot.run()


async def _run_brief() -> None:
    from merv.core.graph import build_merv_graph
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]Generating morning brief...[/bold cyan]")
    graph = build_merv_graph()
    state = await graph.process("Give me my morning brief")
    console.print(state.current_response)


async def _run_chat() -> None:
    from merv.core.graph import build_merv_graph
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]Merv Interactive Chat (type 'quit' to exit)[/bold cyan]\n")
    graph = build_merv_graph()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        state = await graph.process(user_input)
        console.print(f"\n[bold green]Merv:[/bold green] {state.current_response}\n")


if __name__ == "__main__":
    main()
