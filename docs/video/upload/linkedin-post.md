I built an AI technical interviewer that runs entirely on my Mac.

Senjin joins a Google Meet, tells the candidate it is an AI, asks questions drawn from their resume and the job description, listens, follows up, answers the candidate's own questions, and writes an advisory report afterward. A person reads the transcript and makes the hiring decision.

No cloud models and no API keys. Speech recognition (whisper.cpp), the interview model (Ollama), the voice (Piper) and the scoring model all run locally.

Three design choices I'd pass on:

1. Code runs the interview, the model only phrases it. Questions and follow-ups come from a plan, so every candidate gets the same structure and nothing is invented.
2. Every finding in the report has to quote the candidate, and the code checks the quote against the transcript. A finding it can't prove is dropped.
3. A fixed hiring bar sits on top of the scores, and the report says when it changed the model's answer and why.

It also broke in plenty of ways: a lost follow-up when a candidate cut in, praise leaking through a curly apostrophe, a prompt change that quietly removed the AI disclosure from the greeting. The video covers those too.

In the video: how a reply moves through the call, the report, what I checked (small samples, not benchmarks), and what could come next.

Watch: https://youtu.be/-vwdgV50Ewc
Code and docs: https://github.com/tuwallace/senjin

The interview shown in the video is a re-creation from a real transcript, with synthetic voices. Rules about automated interviews differ by place, so check yours before using anything like this.

#LocalLLM #Ollama #AIEngineering #Python #OpenSource
