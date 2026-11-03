# Recording guide for the Senjin video

Companion to `script.md`. Two things are recorded live: the terminal segments (3 and 6) and the Meet call (1 and 4).
Segments 2, 5, 7, 8 and 9 are slides; any slide tool works. Record the screen silent and lay your voice (or a synthetic
voice, which YouTube may require you to label) under it.

## 1. Before you record

```bash
cd ~/source/senjin
ollama list | grep -E "qwen3:8b-16k|qwen2.5:32b"   # models present
pgrep -fl "senjin-meet|whisper-server" || echo clean      # nothing left from an earlier run
caffeinate -dis &                                         # keep the Mac awake
```

1. Do Not Disturb on. Close mail and chat.
2. Terminal: font 20 pt or larger, about 1280 x 720, dark background, short prompt (`PROMPT='$ '`, then `clear`).
3. Privacy: the resume shows the author's name and work history. If you would rather not show it, show only
   `interview.yaml`, the coverage table and the report (the report holds a redacted summary and quotes). Do not
   open `.env`, `data/meet_profile` or anything under `~/.config`.
4. Use a **second person or a second device** as the candidate, or answer from your phone, so the call has two
   participants. Senjin waits for one (`--wait-candidate`, 600 s by default).
5. Create the Meet from the Google account the bot is signed in to, and check the notice: tell the candidate first that
   an AI will interview them and the video is recorded.
6. Record with `Cmd+Shift+5`, Microphone None, a 16:9 region. One file per segment.

## 2. Segment 3: planning (about 40 s of screen)

Type the first command, press record, then press Enter. Do not run `senjin-review` on camera: it calls the 32B model and
rewrites the files. The files are already written for `jane`.

```bash
cat data/candidates/jane/interview.yaml
```

```bash
head -12 data/candidates/jane/questions.coverage.md
```

| Narration says | On screen |
|---|---|
| One command reads the job description and picks a shape | `interview.yaml`: profile `senior_fullstack`, 3 background, 5 technical, 1 follow-up |
| Then it prints a coverage report | The first lines of the coverage file: "5/13 reached by the first 5 questions" |
| That gap is visible before anyone is interviewed | Scroll the table slowly; hold on a few `gap` rows and the `asked` rows |

Optional (a dry run, about a minute of model time; speed it up in editing): `uv run senjin-review --candidate-dir data/candidates/jane --dry-run`.

## 3. Segments 1 and 4: the call

Start it, admit the bot, then act as the candidate. The script uses these moments, so make sure each happens:

1. Answer "Yes." after the greeting. Keep the first background answer to about 20 seconds.
2. In a technical question, ask: "Can you repeat that question, please?" (the repeat handling).
3. Cut in once while Senjin is speaking (barge-in). Say one short sentence and let it carry on.
4. In the final part, ask: "Are we using AWS as the cloud service?" and "Is there a clearance involved?" (the Q&A
   answers from the job description).
5. Say "That was my last question." The call ends by itself.

```bash
uv run senjin-meet run https://meet.google.com/xxx-xxxx-xxx --candidate-dir data/candidates/jane --assess
```

A full interview took about 20 minutes on 2026-11-03. To keep the footage short, use
`--max-questions 4` for a take, or cut it in editing. Keep the terminal output beside the Meet window: the transcript
lines scroll as the call goes, which makes a good lower-third. Record the Meet window and the terminal as one region.

| Cue | What the viewer should see |
|---|---|
| First 15 s | Senjin's tile; the printed greeting ending "Are you ready to begin?" |
| A technical question | The question text printed in the terminal while it is spoken |
| The repeat request | The same question in simpler words |
| The end | `Interview over.`, `Senjin has left the call.`, then the latency summary |

What the last live run printed (use these as expectations, yours will differ): 19 turns measured; end of speech to
first sound mean 2.09 s; the "I see" filler played on 14 of 19 turns; judge `No Hire`, technical 6, communication 6.

## 4. Segment 6: the report (about 40 s of screen)

```bash
sed -n 1,16p data/reports/meet-20261103-132204.md
```

| Narration says | On screen |
|---|---|
| The judge writes strengths and weaknesses | The Strengths and Weaknesses headings |
| Every finding quotes the candidate | One bullet with its `[11] "..."` evidence line |
| The judge said Hire; the bar said No Hire | The italic hiring-bar line near the top |
| It is advisory | The last line: "Advisory only: a human makes the hiring decision." |

If you record a fresh interview, replace the filename with the new `data/reports/meet-*.md` (`ls -t data/reports | head`).
The wording of strengths and scores changes from run to run with the transcript, so read it before you narrate.

## 5. Honesty checks

- The latency (about 2 s) is one interview with one human candidate on this Mac. Say "in one live call".
- The fairness result is a check on a few paired transcripts, not proof that the system is unbiased.
- Slides 7 and 8 name real bugs from this repository's history; do not add numbers that are not in the repository.
- The coverage numbers (5 of 13) belong to the `jane` job description only.
- The voice is synthetic. Say so on the slides and use the YouTube "altered or synthetic content" label if it asks.

## 6. Things that can go wrong

- **The bot cannot join.** Open `https://meet.google.com/xxx-xxxx-xxx` (not `/meet/xxx`) and check that Chrome is signed in
  (`uv run senjin-meet login`).
- **No sound from Senjin.** `uv run senjin-meet devices` and `uv run senjin-meet selftest`; BlackHole must be present.
- **A stale process holds the audio.** `pkill -f senjin-meet; pkill -f whisper-server`, then try again.
- **Slow replies.** Quit other memory-heavy apps; the interview and judge models are large.
