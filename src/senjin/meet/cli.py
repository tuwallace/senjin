"""`senjin-meet devices | selftest | login | run`."""

import argparse
import asyncio
from pathlib import Path

from senjin.audio.stt import DEFAULT_WHISPER_MODEL
from senjin.llm.client import DEFAULT_MODEL
from senjin.meet.audio import HEAR_DEVICE, SPEAK_DEVICE, DeviceError, find_bridge


async def login() -> None:
    from senjin.meet.joiner import MeetBot

    bot = MeetBot()
    await bot.start()
    print("A Chrome window opened. Sign in to the Google account the bot should use.")
    print(
        "Type your own credentials in that window; this tool never sees them. "
        "Waiting up to 10 minutes..."
    )
    try:
        ok = await bot.login()
    finally:
        await bot.close()
    print(
        "Signed in. The session is saved in data/meet_profile."
        if ok
        else "Timed out; not signed in."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="senjin-meet")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "run", "devices"):
        p = sub.add_parser(name)
        p.add_argument("--speak-device", default=SPEAK_DEVICE, help="virtual mic the browser reads")
        p.add_argument(
            "--hear-device", default=HEAR_DEVICE, help="virtual speaker the browser plays to"
        )
    for name in ("selftest", "run"):
        sub.choices[name].add_argument("--stt-model", default=DEFAULT_WHISPER_MODEL)
    sub.add_parser("login", help="sign the bot's Google account in (by hand, once)")
    run = sub.choices["run"]
    run.add_argument("url", help="https://meet.google.com/xxx-xxxx-xxx")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--exemplars", type=Path)
    run.add_argument("--out", type=Path)
    run.add_argument("--admit-timeout", type=float, default=300)
    run.add_argument("--wait-candidate", type=float, default=600)
    run.add_argument("--silence-ms", type=int, default=800)
    run.add_argument("--guest-name", default="Sen (AI interviewer)")
    args = parser.parse_args()

    try:
        if args.cmd == "devices":
            bridge = find_bridge(speak=args.speak_device, hear=args.hear_device)
            print(
                f"speak: {bridge.speak_name} (#{bridge.speak_index})\n"
                f"hear:  {bridge.hear_name} (#{bridge.hear_index})"
            )
        elif args.cmd == "login":
            asyncio.run(login())
        elif args.cmd == "selftest":
            from senjin.meet.selftest import run_selftest

            ok = asyncio.run(run_selftest(args.speak_device, args.hear_device, args.stt_model))
            raise SystemExit(0 if ok else 1)
        else:
            from senjin.meet.runner import default_out, run_meet_interview

            out = args.out or default_out()
            asyncio.run(
                run_meet_interview(
                    args.url, model=args.model, stt_model=args.stt_model,
                    speak=args.speak_device, hear=args.hear_device, out=out,
                    admit_timeout=args.admit_timeout, wait_candidate=args.wait_candidate,
                    silence_ms=args.silence_ms, guest_name=args.guest_name,
                    exemplars=args.exemplars,
                )
            )  # fmt: skip
            print(f"transcript: {out}")
    except DeviceError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
