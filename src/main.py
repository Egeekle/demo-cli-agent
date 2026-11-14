import asyncio
from typing import List

import typer
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent import appointment_agent
from src.approvals import build_results, summarize_approvals
from src.database import create_db_and_tables, get_session
from src.negotiator import negotiator_agent
from src.tools import Deps

app = typer.Typer(help="AI-powered Appointment Assistant CLI")
console = Console()


async def _run_turn(user_input: str, messages: List[ModelMessage]) -> List[ModelMessage]:
    """Run one user turn, handling any approval round-trip. Returns updated history."""
    async with get_session() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent)
        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            result = await appointment_agent.run(
                user_input, message_history=messages, deps=deps
            )

        while isinstance(result.output, DeferredToolRequests):
            requests = result.output
            console.print("\n[bold yellow]Approval required:[/bold yellow]")
            console.print(summarize_approvals(requests))
            answer = console.input("[bold green]Proceed? (yes/no) > [/bold green]").strip().lower()
            approved = answer in ("yes", "y", "confirm")
            results = build_results(requests, approved=approved)
            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                result = await appointment_agent.run(
                    message_history=result.all_messages(),
                    deferred_tool_results=results,
                    deps=deps,
                )

        console.print("\n[bold blue]Assistant >[/bold blue]")
        console.print(Markdown(result.output))
        return result.all_messages()


async def chat_loop():
    console.print(Panel.fit(
        "[bold blue]Personal Appointment Assistant[/bold blue]\n"
        "Ask about your schedule, create, update, or delete appointments.\n"
        "\n"
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to stop.",
        title="Welcome",
    ))

    await create_db_and_tables()
    messages: List[ModelMessage] = []

    while True:
        try:
            user_input = console.input("\n[bold green]You > [/bold green]")
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            if not user_input.strip():
                continue
            messages = await _run_turn(user_input, messages)
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[bold red]An error occurred:[/bold red] {e}")


@app.command()
def chat():
    """Start an interactive chat session with the AI assistant."""
    asyncio.run(chat_loop())


if __name__ == "__main__":
    app()
