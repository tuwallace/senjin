# Video script: "I built an AI interviewer that runs entirely on my Mac"

Target length: about 7 minutes (about 950 spoken words at a relaxed pace). Everything on screen uses the
`data/candidates/jane` folder (the candidate is the author, who agreed to be shown). Narration is written to be read
aloud; change anything that does not sound like you. Recording steps are in `live-recording-guide.md`; the upload text
is in `upload/`.

Every number below comes from this repository's own runs. The latency numbers are from one live Meet interview on
2026-11-03 (19 measured turns). Nothing here is a benchmark.

## Segment plan

| # | Time | Segment | Type |
|---|---|---|---|
| 1 | 0:00 | Hook | LIVE (the call) |
| 2 | 0:30 | What Senjin is | SLIDE |
| 3 | 1:10 | Planning from a resume and a job description | LIVE (terminal) |
| 4 | 2:10 | The call: the greeting, then the interview | LIVE (Meet) |
| 5 | 3:30 | Code asks, the model only phrases | SLIDE |
| 6 | 4:10 | The report: quotes it can prove | LIVE (terminal) |
| 7 | 5:00 | Checking it honestly | SLIDE |
| 8 | 5:45 | What went wrong | SLIDE |
| 9 | 6:25 | Limits and takeaways | SLIDE |

---

## 1. Hook (0:00)

**Visual (LIVE):** a 15 second clip from the Meet recording: Senjin's tile speaking the greeting, the candidate
answering "Yes." Caption: "A real call. The interviewer is a local AI."

**Narration:**
> This is a job interview. The interviewer is an AI running on my laptop. No cloud models, no API keys. It joined a
> Google Meet, introduced itself, asked questions drawn from the candidate's resume and the job posting, listened,
> asked follow-ups, and afterwards wrote an assessment. A person still makes the hiring decision. Here is how it works,
> and what broke along the way.

## 2. What Senjin is (0:30)

**Visual (SLIDE):** pipeline: resume + job description, plan, questions, Meet call, transcript, report. Stack row
underneath: Ollama (qwen3 8B for the interview, a 32B judge for scoring), whisper.cpp, Piper, Silero VAD, BlackHole,
Playwright.

**Narration:**
> Senjin has two halves. Before the call, it plans: it reads the resume and the job description and decides what to ask.
> During the call, it listens with whisper.cpp, thinks with a local Ollama model, and speaks with Piper, through virtual
> audio devices into Google Meet. After the call, a bigger local model scores the transcript. All of it runs on one
> Mac.

## 3. Planning from a resume and a job description (1:10)

**Visual (LIVE terminal):** `cat data/candidates/jane/interview.yaml`, then the top of `questions.coverage.md`: the
requirement table with `asked` and `gap` statuses.

**Narration:**
> The plan comes first. One command reads the job description, picks an interview shape, here a senior full stack
> profile with three background questions and five technical ones, and writes the questions. Half come from a shared
> question bank, half are written for this job. Then it prints a coverage report: which of the job's requirements the
> questions actually reach, and which they don't. For this posting, five of thirteen must-haves are covered by the default
> five questions. That gap is visible before anyone is interviewed, and you can edit the questions file by hand. The
> candidate's name and contact details are removed before any model sees the resume.

## 4. The call (2:10)

**Visual (LIVE):** the Meet recording, cut to about 80 seconds: the greeting, a background answer and follow-up, one
technical question, the candidate saying "Can you repeat that question, please?" and Senjin rephrasing, then the end.
Show the live transcript lines in a terminal beside it if the layout allows.

**Narration:**
> In the call, the first thing it says is fixed text: it is an AI, the conversation is transcribed, a person decides,
> you can ask to stop, here is the format, are you ready? That text is not left to the model, because early on the model
> kept dropping parts of it. Then it asks. It waits for you to finish, replies in about two seconds, and plays a short
> "I see" when a reply is slow. If you cut in, it stops talking. If you ask it to repeat, it says the question again in
> simpler words. And in the last part, you can ask about the role, and it answers only from the job description.

## 5. Code asks, the model only phrases (3:30)

**Visual (SLIDE):** two columns. "Code decides": which question, when to follow up, when to move on, when to stop.
"Model does": say it naturally, acknowledge, answer from the job description. Small note: "never praise, never hint".

**Narration:**
> The key design choice: the code runs the interview, not the model. Code picks the next question and decides when to
> follow up or move on. The model only phrases it. That keeps every candidate on the same plan, keeps the questions
> from being invented, and lets me test the flow with fakes instead of real models. The interviewer is also told never
> to praise or criticize, and a filter removes praise if the model slips.

## 6. The report (4:10)

**Visual (LIVE terminal):** `senjin-assess` output or `data/reports/meet-20261103-132204.md`: the recommendation line,
the hiring-bar note, then one weakness with its quoted evidence.

**Narration:**
> After the call, the judge writes strengths, weaknesses and a score for each question. Every finding has to quote the
> candidate, and the code checks that the quote is really in the transcript. A finding it cannot prove is dropped. Then a
> fixed hiring bar applies. In this run the judge said Hire, the bar said technical six is below seven, and the report
> says so in plain words: No Hire, low confidence, and why it changed. It is advisory. Someone has to read the transcript.

## 7. Checking it honestly (5:00)

**Visual (SLIDE):** four lines. "Same inputs, same report: 0 variance (temperature 0)". "302 offline tests, no model
calls". "Fairness check: names and cues swapped, scores held". "Live call: 19 turns, about 2.1 s to first sound".

**Narration:**
> I checked three things. Same transcript in, same report out: with temperature zero there was no variation. Then a
> fairness check: the same answers with different names and cues attached, with and without the redaction step. The
> redaction version stayed level. And speed: in the last live call it took about two seconds from the candidate
> stopping to the first sound. The tests do not call any model. They use fakes, so they run anywhere.

## 8. What went wrong (5:45)

**Visual (SLIDE):** a short list, one line each, with a small "fixed" tick.

**Narration:**
> Plenty broke. A follow-up was lost when the candidate cut in before the model spoke. Praise leaked through when the
> model used a curly apostrophe. A quote check failed because the judge wrote a period where the candidate had a comma.
> The coverage report called questions a gap when they were only cut for time. And my own prompt change made the greeting
> drop the notice, which is why it is fixed text now. I also tried a small decision model to detect when the candidate
> is finished. It was more accurate on my small test set, but slower, so it is off by default.

## 9. Limits and takeaways (6:25)

**Visual (SLIDE):** "Advisory only. Local only. Not legal advice." Then three takeaways.

**Narration:**
> Limits. Speech recognition mishears names and terms. The voice is still obviously synthetic. Rules about automated
> interviews differ by place, so check yours. Three things I would pass on. One: let code run the process and keep the
> model to language. Two: make every claim in a report point to evidence you can check. Three: read the actual
> transcripts, because the averages never showed me any of these bugs. The code and the docs are on GitHub.
