# ADR 0004: Google Meet bot

Date: 2026-10-29. Status: implemented; verified against a mock page and Chrome fake devices only.

## Choices
- **Real Chrome, visible window, persistent profile** (`data/meet_profile`), not headless Chromium. Google often challenges fresh or headless automated browsers. `--enable-automation` is dropped and `AutomationControlled` is disabled. The account is signed in by the person (`senjin-meet login`); this code never handles credentials. If Google still blocks it, the roadmap fallbacks apply (human-admitted real profile, a tab-capture extension, or Jitsi).
- **Audio routing at the API level.** macOS has no per-app audio routing and changing the system default would hijack the user's sound. An init script forces `getUserMedia` onto the device whose label contains `BlackHole 2ch`, and routes every `audio`/`video` element to `BlackHole 16ch` with `setSinkId` (on play and once a second for autoplay elements). Two different drivers are used so the bot's speech and the call audio do not share a channel. Verified: with Chrome's fake devices the mic and sink ids resolve by label and the constraint is applied.
- **Join logic** uses role and name regexes kept in `selectors.py`: dismiss popups, fill the guest name, camera off, mic on, press Join or Ask to join, wait for the Leave call button, detect denial. Participant count is read from the people button and is best effort; the interview starts anyway after a timeout.
- **Runner** reuses the Phase 5 voice loop with BlackHole devices, and stops on sign-off or when the call UI disappears for three polls.

## Verified here
- Join, leave, denial and participant-count flow against `tests/fixtures/mock_meet.html`.
- Device pinning with Chrome's `--use-fake-device-for-media-stream`.
- Missing-driver error with the install command.

## Not verified (needs you)
1. BlackHole is not installed here (it needs `sudo`), so `senjin-meet selftest` has not run.
2. The Meet labels in `selectors.py` come from memory of the Meet UI, not from a live meeting. The mock page encodes the same assumption, so passing tests do not prove they match.
3. Meet applies echo cancellation and noise suppression to the bot's microphone; speech quality over that path is unknown.
4. Google may ask for extra verification when a new profile signs in.

## Risks
Meet UI changes break `selectors.py` silently; automated joining may violate Meet terms for some account types, so use a dedicated account on meetings you host.
