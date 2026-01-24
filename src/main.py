import asyncio
from typing import List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pydantic_ai.messages import ModelMessage

from src.agent import appointment_agent
from src.database import create_db_and_tables, get_session
from src.tools import Deps

app = typer.Typer(help="AI-powered Appointment Assistant CLI")
console = Console()


async def chat_loop():
    """
    Main REPL loop for the appointment assistant.
    """
    console.print(
        Panel.fit(
            "[bold blue]Personal Appointment Assistant[/bold blue]\n"
            "I can help you manage your appointments. Ask me about your schedule,\n"
            "create new ones, or delete them.\n"
            "\n"
            "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to stop.",
            title="Welcome",
        )
    )

    # Ensure database is initialized
    create_db_and_tables()

    # Store conversation history
    messages: List[ModelMessage] = []

    # Use a single session for the entire chat session
    with get_session() as session:
        deps = Deps(session=session)

        while True:
            try:
                user_input = console.input("\n[bold green]You > [/bold green]")

                if user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if not user_input.strip():
                    continue

                with console.status("[bold yellow]Thinking...[/bold yellow]"):
                    # Run the agent with current history and dependencies
                    result = await appointment_agent.run(
                        user_input, message_history=messages, deps=deps
                    )

                    # Update history for the next turn
                    messages = result.all_messages()

                    # Render the response
                    console.print("\n[bold blue]Assistant >[/bold blue]")
                    console.print(Markdown(result.output))

            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[bold red]An error occurred:[/bold red] {e}")
                # Optional: log the traceback for debugging
                # import traceback
                # console.print(traceback.format_exc())


@app.command()
def chat():
    """
    Start an interactive chat session with the AI assistant.
    """
    asyncio.run(chat_loop())


if __name__ == "__main__":
    app()
