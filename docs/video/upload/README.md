# YouTube upload checklist

The video and the other upload files are built by `docs/video/make_video.py` (see `docs/video/script.md`). The
video itself is large and git-ignored; it lives in `data/video/upload/` beside copies of these files.

| File | Use |
|---|---|
| `senjin-ai-interviewer.mp4` (in `data/video/upload/`) | The video: 1080p, about 6 minutes |
| `title.txt` | The title. Two alternatives are in `titles.txt` |
| `description.txt` | Paste into the description box. The chapter list (`0:00 ...`) makes YouTube chapters |
| `tags.txt` | Paste into the Tags field, comma separated |
| `captions.srt` | Upload under Subtitles, English. Timed to the narrated parts |
| `thumbnail.png` | 1280 x 720 custom thumbnail |

## Settings

- **Audience:** not made for kids.
- **Altered or synthetic content:** answer yes. The narration is a synthetic voice, and the opening clip re-creates a
  real call with two synthetic voices. The description says so.
- **Category:** Science and Technology. **Language:** English.
- **Visibility:** upload as Unlisted first, watch it once on YouTube, then switch to Public.
- **Comments and end screen:** your choice. Nothing in the video needs either.

## Before you publish

1. Watch the whole video once with sound. Check the narration speed and the slide joins.
2. The report slide quotes the candidate (the author) from the real call. Make sure you are happy showing it.
3. The slides show job requirement names from the `jane` job description. Check that nothing there identifies an employer.
4. The fairness figure is from an earlier live run, not repeated for this video. Re-run `uv run python -m senjin.eval.fairness` if you want it fresh.
5. Rules about automated interviews differ by place. The video and description say it is not legal advice.
