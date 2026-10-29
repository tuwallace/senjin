"""Keep the bot in the call: rejoin after a dropped connection, stop when the interview is over."""

from collections.abc import Callable
from typing import Protocol


class CallBot(Protocol):
    async def wait_until_ended(self) -> None: ...
    async def join(self, url: str, admit_timeout: float, log) -> None: ...


async def watch_call(
    bot: CallBot,
    url: str,
    finished: Callable[[], bool],
    max_rejoins: int = 2,
    rejoin_timeout: float = 120.0,
    log=print,
) -> str:
    """Return "finished" if the interview ended normally, "ended" if the call is gone for good."""
    rejoins = 0
    while True:
        await bot.wait_until_ended()
        if finished():
            return "finished"
        if rejoins >= max_rejoins:
            return "ended"
        rejoins += 1
        log(f"The call dropped; rejoining ({rejoins}/{max_rejoins})...")
        try:
            await bot.join(url, rejoin_timeout, log)
        except Exception as exc:
            log(f"Could not rejoin: {exc}")
            return "ended"
