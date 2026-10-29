# ADR 0006: Code-driven questions in voice and Meet interviews

Date: 2026-10-29. Status: implemented (`prefetch=True` in `InterviewSession`); text chat and `senjin-sim` keep the model-driven mode unless `--orchestrate` is passed.

## Problem
The first real Meet interview (transcript `meet-20261029-160454`) showed that letting the model drive the question bank failed in four ways: it answered its own question ("Okay, I need to design a rate limiter… Let me start by thinking…"), it read the bank's follow-up list aloud ("as mentioned in the follow-up question") and asked the shared-store question three times, it kept asking technical questions during the candidate Q&A phase so the candidate never got to ask anything, and it ended the sign-off on a question.

## Change
- The session fetches what each phase needs itself: the resume (background), the next question (technical), the job description (Q&A). The resume and job description go into the phase's system message; no tool messages enter the history and the model is given no tools, which also removes tool round trips.
- Each turn, code decides what to say and adds a short `[Interviewer note]` to the candidate's last message (never stored in history): ask this question, ask this one follow-up, move to the next question, probe, invite questions, answer from the job description, or say goodbye without a question. Follow-ups are consumed one at a time, so none repeats and none is visible to the model in advance.
- In Q&A, a short "no, I'm good" moves straight to sign-off.
- Bank questions are written as spoken, second-person questions ("Imagine you are building…") instead of task statements, which invited the model to solve them.
- The system message stays stable within a phase, so the Ollama prefix cache still works; only the last user message changes each turn.

## Evidence
Unit tests cover the flow with fakes. A live run with the real model and a simulated candidate (`senjin-sim --persona average --quick --orchestrate`) asked the rate-limiter question without answering it, used the two follow-ups in order, invited questions, and said goodbye without a question.

## Not verified
Not yet re-run in a real Meet call. In the live simulation the interviewer still opened one reply with "Great insights", despite the no-praise rule, and the quick budget gave the Q&A phase one turn, so the candidate's questions were not answered in that run. Latency in orchestrated mode has not been re-measured on real calls (the previous run averaged 3.5 s to first audio, worst 11.9 s).
