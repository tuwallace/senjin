import asyncio
import http.server
import tempfile
import threading
from pathlib import Path

import pytest

from senjin.meet.audio import DeviceError, find_bridge, find_device
from senjin.meet.browser_js import init_script
from senjin.meet.joiner import JoinError, MeetBot

FIXTURES = Path(__file__).parent / "fixtures"
HAS_CHROME = Path("/Applications/Google Chrome.app").exists()
needs_chrome = pytest.mark.skipif(not HAS_CHROME, reason="Google Chrome not installed")


def dev(name, i=0, o=0):
    return {"name": name, "max_input_channels": i, "max_output_channels": o}


def test_find_bridge_picks_blackhole_devices_by_direction():
    devices = [
        dev("MacBook Pro Microphone", i=1),
        dev("BlackHole 2ch", i=2, o=2),
        dev("BlackHole 16ch", i=16, o=16),
    ]
    bridge = find_bridge(devices)
    assert (bridge.speak_index, bridge.hear_index) == (1, 2)


def test_missing_driver_gives_install_hint():
    with pytest.raises(DeviceError, match="brew install --cask blackhole"):
        find_bridge([dev("MacBook Pro Microphone", i=1)])
    with pytest.raises(DeviceError):
        find_device(
            [dev("BlackHole 2ch", i=2)], "BlackHole 2ch", need_input=False, need_output=True
        )


def test_init_script_embeds_names_safely():
    js = init_script('Black"Hole', None)
    assert '"Black\\"Hole"' in js and "SPK = null" in js


@pytest.fixture(scope="module")
def server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(FIXTURES), **kw)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def make_bot(**kw) -> MeetBot:
    return MeetBot(Path(tempfile.mkdtemp()), headless=True, **kw)


@needs_chrome
async def test_join_flow_against_mock_meet(server):
    bot = make_bot()
    page = await bot.start()
    try:
        await bot.join(f"{server}/mock_meet.html", admit_timeout=10, log=lambda *_: None)
        assert await bot.in_call()
        assert await page.evaluate("window.__clicked") == ["cam", "mic", "join"]
        assert await page.evaluate("window.__name") == "Sen (AI interviewer)"
        await asyncio.sleep(0.5)
        assert await bot.participant_count() == 2
        assert await bot.wait_for_participants(2, timeout=5)
        await bot.leave()
        await asyncio.wait_for(bot.wait_until_ended(poll=0.1, misses=2), 5)
        assert not await bot.in_call()
    finally:
        await bot.close()


@needs_chrome
async def test_denied_request_raises(server):
    bot = make_bot()
    await bot.start()
    try:
        with pytest.raises(JoinError, match="denied"):
            await bot.join(f"{server}/mock_meet.html?deny=1", admit_timeout=10, log=lambda *_: None)
    finally:
        await bot.close()


@needs_chrome
async def test_microphone_and_speaker_are_pinned_by_label(server):
    bot = make_bot(
        mic_name="Fake Audio Input 2",
        speaker_name="Fake Audio Output 2",
        extra_args=["--use-fake-device-for-media-stream"],
    )
    page = await bot.start()
    try:
        await page.goto(f"{server}/mock_meet.html")
        result = await page.evaluate(
            """async () => {
              const s = await navigator.mediaDevices.getUserMedia({audio: true});
              const used = s.getAudioTracks()[0].getSettings().deviceId;
              s.getTracks().forEach(t => t.stop());
              const a = document.createElement('audio');
              document.body.appendChild(a);
              await new Promise(r => setTimeout(r, 1500));   // the sink router runs every second
              return {used, ids: window.__senjin.ids, sink: a.sinkId};
            }"""
        )
        assert result["ids"]["mic"] and result["used"] == result["ids"]["mic"]
        assert result["ids"]["spk"] and result["sink"] == result["ids"]["spk"]
    finally:
        await bot.close()
