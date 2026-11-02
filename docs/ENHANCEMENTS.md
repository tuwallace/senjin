# Potential enhancements

Ideas that came out of design discussions. Nothing here is committed to the roadmap
(`docs/ROADMAP.md`); each item says what is done, what is only proposed, and what has to be
checked first. Claims about Google, Ollama and third-party models come from general knowledge or
a vendor blog post and were not tested unless the item says so.

## 1. Decision models for turn-level decisions

Ollama 0.35+ serves small "decision" models (`nimble` 9B, `tev1` 4B and `tev1:0.8b`) on
`/v1/systemone`: send a state and named questions (choice, yes/no probability, score), get typed
answers with probabilities and confidence.

**Done (opt-in).** `SENJIN_DECISION_MODEL=nimble` adds a second opinion on "has the candidate
finished the Q&A?". The rules decide first. Only when they say "not done" is the model asked, and
only a confident yes (0.8 or higher) ends the Q&A. A missing model, a timeout (3 s) or a bad reply
falls back to the rules. Code: `src/senjin/llm/decide.py`, `InterviewSession._is_done`.

**Measured** (70 hand-labelled turns, about 30 real and the rest written to break the regexes):

| Decision | Rules | nimble |
|---|---|---|
| Q&A finished (43) | 40 right | 43 right |
| Answer complete, not cut off (14) | 13 right | 14 right |
| Repeat request (13) | 13 right | 12 right |

Latency was a median of 632 ms per call on this Mac, not the roughly 91 ms the vendor reports for
an M5 Max. The regexes were tuned on the real turns, which favours the rules. The question name
matters to the model (`done` worked, `q` did not), so every question needs a descriptive name.

**Candidates, not built**

| Idea | Why it might help | Why it is not built |
|---|---|---|
| Answer complete (hold or reply) | Fixes answers with no closing period | Runs on every turn at about 600 ms for one gain |
| Follow-up or move on | Replaces the fixed follow-up count with a judgement of how complete the answer was | Needs a labelled set and a latency budget; changes interview shape |
| Repeat, clarify, don't-know, answer | One typed call instead of several regexes | The repeat regex already scored 13 of 13 |
| Profile choice and coverage mapping in `senjin-review` | Classification and scoring that vary between runs | Offline, so latency is free, but needs a comparison against the 32B judge first |
| Judge scoring as typed scores | Consistent per-question scores | Quote verification must stay; compare on saved transcripts first |

**Before widening use**
- Warm the decision model before the call starts. A cold first call can exceed the 3 s limit and
  fall back to the rules.
- Build a larger labelled set from real calls, with false stops (ending the Q&A early) counted
  separately from missed sign-offs.
- Re-measure latency with the interview model, speech-to-text and voice running together.
- Run a full live Meet interview with the option on.

## 2. Scheduling and hosting the Meet

**Goal.** After `senjin-review`, create a Google Meet invite with human approval, host the call,
wait for the candidate, then interview. Today the bot joins a link someone else created, a person
admits it, and `--wait-candidate` already waits for a second participant.

**Proposed flow**
1. `senjin-review` finishes as it does now.
2. `senjin-schedule --candidate-dir DIR --email ADDRESS --start TIME` builds a Google Calendar
   event with a Meet link.
3. It prints the title, time, recipient, link and plan summary, and creates or sends nothing until
   a person types yes. The result is saved as `invite.yaml` in the candidate folder.
4. A scheduler (cron or launchd) starts the bot about five minutes before the start time. The bot's
   Google account is the organizer.
5. The bot waits for the candidate up to a limit. On no-show it leaves, records "no-show" and
   notifies the human.
6. The existing interview and judge run.

**Feasibility** (not tested)

| Part | Feasibility | Notes |
|---|---|---|
| Create the event and link | High | Google Calendar API with a Meet conference request; needs a one-time OAuth consent and a Google Cloud project. Token kept in git-ignored `data/`. The alternative, opening `meet.google.com/new` in the signed-in Chrome profile, is fragile and sends no email |
| Human approval | High | Terminal confirmation plus a saved record |
| Starting on time | High | The Mac must be awake with Ollama and BlackHole running |
| Waiting and no-show | High | The wait exists; add a timeout, notification and record |
| Admitting the candidate | Medium, needs a real test | Whether a candidate must knock depends on the account type and the meeting's access setting. Needs new Meet selectors in `src/senjin/meet/selectors.py` |
| Running unattended | Medium | Google can challenge automated Chrome |

**Risks and rules**
- Meet shows only a display name, so someone could join under another name. Admit automatically
  only when the name roughly matches the candidate's first name, otherwise notify the human.
- The invite must carry no questions, scores or resume details. It should say that an AI conducts
  and transcribes the interview.
- Calendar is a cloud call. It carries only the candidate's email address and the time, not
  interview data. Confirm that this fits the "everything runs locally" rule.
- Some places regulate automated interviews and require notice or consent. Have that checked for
  the hiring locations. This document is not legal advice.

**Open questions.** Organizer account type (Workspace or personal). Whether invited guests can
join without knocking. How to notify the human (macOS notification, email, chat). What happens if
the Mac sleeps or Chrome is challenged.

**Suggested order.** Prototype with a throwaway Google account to test the admit step first, then
the approved-invite command, then the scheduler and no-show handling.

## 3. A voice that is easier to understand

**Done.** Senjin now speaks 20% slower by default (`SENJIN_VOICE_PACE=1.2`, Piper
`en_US-lessac-medium`). Eight voices were compared by ear from the same four lines, in clean and
call-quality (Opus) versions, and the slower Piper voice was chosen. Synthesis time does not
change; the audio is about 11% longer.

**Open**
- The time box assumes about 2 minutes per interviewer turn. Slower speech makes turns a little
  longer, so the budgets may need to grow. Check on the first live run.
- Other options, none tried: Piper `-high` voices, macOS Premium or Enhanced voices (a free
  download in System Settings), Kokoro-82M (a new dependency), and larger models such as F5-TTS or
  Orpheus (much heavier, competing with the interview model for memory).
- Quick wins regardless of voice: shorter sentences, and spelling hard terms the way they are
  said ("fed ramp").
- Speech-to-text mishearing candidates, such as names, has been a bigger problem so far than the
  candidate's difficulty hearing Senjin. Ask for candidate feedback on the voice before spending
  more here.

## 4. Report upload

**Idea.** Send a finished report (`data/reports/*.md` and `.json`) to a Google Drive folder or a
Slack channel, only when a person asks for it. This was an optional item in Phase 7 of the roadmap
and was moved here because the MVP does not need it. Nothing is built.

**Rules if built**
- A person starts every upload, naming the report and the destination. Nothing uploads
  automatically after an interview.
- It sends interview content to a cloud service, so it is outward-facing and should show the file
  and destination, then wait for a yes, as the invite flow in section 2 does.
- Reports hold candidate answers and scores. Keep names and contact details out, as the judge input
  already does, and say plainly that the report is advisory.
- Credentials stay in git-ignored `data/`, never in the repo.

**Alternative.** Share the file by hand from `data/reports/`; this needs no code.
