"""`senjin-meet devices | selftest | login | debug | run`."""

import argparse
import asyncio
from pathlib import Path

from senjin.audio.stt import DEFAULT_WHISPER_MODEL
from senjin.config import InterviewConfig, exit_on_config_error
from senjin.llm.client import DEFAULT_MODEL
from senjin.meet.accounts import ProfileError, keep_account, logout
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


@exit_on_config_error
def main() -> None:
    parser = argparse.ArgumentParser(prog="senjin-meet")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "run", "devices", "debug"):
        p = sub.add_parser(name)
        p.add_argument("--speak-device", default=SPEAK_DEVICE, help="virtual mic the browser reads")
        p.add_argument(
            "--hear-device", default=HEAR_DEVICE, help="virtual speaker the browser plays to"
        )
    for name in ("selftest", "run"):
        sub.choices[name].add_argument("--stt-model", default=DEFAULT_WHISPER_MODEL)
    dbg = sub.choices["debug"]
    dbg.add_argument("url")
    dbg.add_argument("--seconds", type=float, default=60)
    sub.add_parser("login", help="sign the bot's Google account in (by hand, once)")
    sub.add_parser("logout", help="delete the bot's saved Google sign-in")
    keep = sub.add_parser(
        "keep-account", help="set the current sign-in aside, then sign in with another account"
    )
    keep.add_argument("--label", help="name for the saved folder (default: a timestamp)")
    run = sub.choices["run"]
    run.add_argument("url", help="https://meet.google.com/xxx-xxxx-xxx")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--exemplars", type=Path)
    run.add_argument(
        "--candidate-dir", type=Path, help="folder with resume.md and job_description.md"
    )
    run.add_argument("--out", type=Path)
    run.add_argument("--admit-timeout", type=float, default=300)
    run.add_argument("--wait-candidate", type=float, default=600)
    run.add_argument("--silence-ms", type=int, help="env SENJIN_SILENCE_MS, default 1000")
    run.add_argument("--filler-ms", type=int, help="env SENJIN_FILLER_MS, default 1100 (0 = off)")
    run.add_argument("--max-questions", type=int, help="env SENJIN_MAX_QUESTIONS, default 5")
    run.add_argument("--max-minutes", type=float, help="env SENJIN_MAX_MINUTES, default 45")
    run.add_argument("--idle-seconds", type=float, help="env SENJIN_IDLE_SECONDS, default 25")
    run.add_argument(
        "--assess", action="store_true", help="score the transcript when the call ends"
    )
    run.add_argument("--judge", default=None, help="judge model for --assess")
    run.add_argument("--guest-name", default="Sen (AI interviewer)")
    args = parser.parse_args()

    try:
        if args.cmd == "devices":
            bridge = find_bridge(speak=args.speak_device, hear=args.hear_device)
            print(
                f"speak: {bridge.speak_name} (#{bridge.speak_index})\n"
                f"hear:  {bridge.hear_name} (#{bridge.hear_index})"
            )
        elif args.cmd == "debug":
            from senjin.meet.debug import run_debug

            asyncio.run(run_debug(args.url, args.speak_device, args.hear_device, args.seconds))
        elif args.cmd == "logout":
            removed = logout()
            print("Signed out: deleted data/meet_profile." if removed else "Already signed out.")
        elif args.cmd == "keep-account":
            saved = keep_account(label=args.label)
            if saved:
                print(f"Kept the current account in {saved}.")
            asyncio.run(login())
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
                    guest_name=args.guest_name,
                    config=InterviewConfig.from_env().override(
                        args.max_questions,
                        args.max_minutes,
                        args.idle_seconds,
                        args.silence_ms,
                        args.filler_ms,
                    ),
                    exemplars=args.exemplars, candidate_dir=args.candidate_dir,
                )
            )  # fmt: skip
            print(f"transcript: {out}")
            if args.assess:
                from senjin.assess.cli import ROOT
                from senjin.assess.cli import run as assess_run
                from senjin.assess.judge import DEFAULT_JUDGE

                asyncio.run(
                    assess_run(
                        out,
                        args.judge or DEFAULT_JUDGE,
                        Path("data/reports"),
                        ROOT / "question_bank",
                    )
                )
    except (DeviceError, ProfileError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
