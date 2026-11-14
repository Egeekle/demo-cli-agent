from src import main


async def _noop() -> None:
    return None


async def test_chat_loop_exits_on_end_of_input(monkeypatch):
    # Ctrl-D (or piped input running out) makes console.input raise EOFError; the
    # loop must end instead of printing the error and prompting again forever.
    monkeypatch.setattr(main, "create_db_and_tables", _noop)
    prompts = 0

    def fake_input(prompt: str = "") -> str:
        nonlocal prompts
        prompts += 1
        if prompts > 1:
            # Not an Exception subclass, so the loop's catch-all can't swallow it.
            raise KeyboardInterrupt
        raise EOFError

    monkeypatch.setattr(main.console, "input", fake_input)

    await main.chat_loop()

    assert prompts == 1
