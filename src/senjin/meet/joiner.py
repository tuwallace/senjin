"""Join a Google Meet with Playwright and report call state.

Runs a real, visible Chrome with a persistent profile: Google often blocks or challenges
headless and fresh automated browsers. Sign in once with `senjin-meet login` (you type the
credentials; this code never sees them). Selectors live in selectors.py.
"""

import asyncio
import time
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

from senjin.meet import selectors as sel
from senjin.meet.browser_js import init_script

PROFILE_DIR = Path("data/meet_profile")


class JoinError(RuntimeError):
    pass


class MeetBot:
    def __init__(
        self,
        profile_dir: Path = PROFILE_DIR,
        mic_name: str | None = None,
        speaker_name: str | None = None,
        headless: bool = False,
        channel: str | None = "chrome",
        guest_name: str = "Senjin (AI interviewer)",
        extra_args: list[str] | None = None,
    ) -> None:
        self.extra_args = extra_args or []
        self.profile_dir = profile_dir
        self.mic_name, self.speaker_name = mic_name, speaker_name
        self.headless, self.channel, self.guest_name = headless, channel, guest_name
        self._pw = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self) -> Page:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        self.context = await self._pw.chromium.launch_persistent_context(
            str(self.profile_dir),
            channel=self.channel,
            headless=self.headless,
            args=[
                "--use-fake-ui-for-media-stream",  # grant mic without the permission bubble
                "--autoplay-policy=no-user-gesture-required",
                "--disable-blink-features=AutomationControlled",
                *self.extra_args,
            ],
            ignore_default_args=["--enable-automation"],
            permissions=["microphone"],
            viewport={"width": 1280, "height": 800},
        )
        await self.context.add_init_script(init_script(self.mic_name, self.speaker_name))
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        return self.page

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self._pw:
            await self._pw.stop()
        self.context = self._pw = self.page = None

    async def signed_in(self) -> bool:
        assert self.context
        cookies = await self.context.cookies("https://accounts.google.com")
        return any(c["name"] in {"SID", "SAPISID", "__Secure-1PSID"} for c in cookies)

    async def login(self, timeout: float = 600.0) -> bool:
        """Open Google sign-in and wait while the person signs in by hand."""
        assert self.page
        await self.page.goto("https://accounts.google.com/")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.signed_in():
                return True
            await asyncio.sleep(2)
        return False

    async def _visible(self, locator) -> bool:
        try:
            return await locator.first.is_visible(timeout=300)
        except Exception:
            return False

    def _button(self, pattern):
        assert self.page
        return self.page.get_by_role("button", name=pattern)

    async def in_call(self) -> bool:
        return await self._visible(self._button(sel.LEAVE_BUTTON))

    async def _text_present(self, pattern) -> bool:
        assert self.page
        return await self._visible(self.page.get_by_text(pattern))

    async def join(self, url: str, admit_timeout: float = 300.0, log=print) -> None:
        """Open the meeting, set camera off and mic on, ask to join, wait to be admitted."""
        assert self.page
        await self.page.goto(url, wait_until="domcontentloaded")
        join = self._button(sel.JOIN_BUTTON)
        await join.first.wait_for(state="visible", timeout=45_000)

        for _ in range(3):  # optional popups ("Got it", device tips)
            if not await self._visible(self._button(sel.DISMISS_BUTTONS)):
                break
            await self._button(sel.DISMISS_BUTTONS).first.click()

        name = self.page.get_by_label(sel.NAME_FIELD)
        if await self._visible(name):  # guest (not signed in)
            await name.first.fill(self.guest_name)
        if await self._visible(self._button(sel.CAMERA_ON_BUTTON)):
            await self._button(sel.CAMERA_ON_BUTTON).first.click()
        if await self._visible(self._button(sel.MIC_OFF_BUTTON)):
            await self._button(sel.MIC_OFF_BUTTON).first.click()

        await join.first.click()
        log("Asked to join; waiting for the host to admit the bot...")
        deadline = time.monotonic() + admit_timeout
        while time.monotonic() < deadline:
            if await self.in_call():
                return
            if await self._text_present(sel.DENIED_TEXT):
                raise JoinError("the host denied the request or nobody responded")
            await asyncio.sleep(1)
        raise JoinError(f"not admitted within {admit_timeout:.0f}s")

    async def participant_count(self) -> int | None:
        """Best effort: the number shown on the people button, or None if not readable."""
        import re

        button = self._button(sel.PEOPLE_BUTTON)
        if not await self._visible(button):
            return None
        label = (await button.first.get_attribute("aria-label")) or ""
        text = await button.first.inner_text()
        match = re.search(r"\d+", f"{label} {text}")
        return int(match.group()) if match else None

    async def wait_for_participants(self, minimum: int = 2, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            count = await self.participant_count()
            if count is not None and count >= minimum:
                return True
            if not await self.in_call():
                return False
            await asyncio.sleep(2)
        return False

    async def wait_until_ended(self, poll: float = 2.0, misses: int = 3) -> None:
        """Return once the call UI has been gone for `misses` consecutive checks."""
        gone = 0
        while gone < misses:
            gone = 0 if await self.in_call() else gone + 1
            await asyncio.sleep(poll)

    async def leave(self, attempts: int = 3) -> bool:
        """Hang up. True once the call is gone; drops the page if the button does not work."""
        if not self.page:
            return True
        for _ in range(attempts):
            if not await self.in_call():
                return True
            try:
                await self._button(sel.LEAVE_BUTTON).first.click(timeout=3000)
            except Exception:
                pass
            for _ in range(8):  # the call screen takes a moment to go away
                if not await self.in_call():
                    return True
                await asyncio.sleep(0.5)
        try:
            await self.page.goto("about:blank")  # last resort: leaving the page ends the call
        except Exception:
            return False
        return True
