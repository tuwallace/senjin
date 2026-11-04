"""Write one narration line as a WAV with the Kokoro-82M neural voice.

Kokoro is not a Senjin dependency, so this runs under another Python that has it installed:

    /path/to/kokoro-venv/bin/python docs/video/kokoro_say.py VOICE OUT.wav < text.txt
"""

import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline

voice, out = sys.argv[1], sys.argv[2]
pipe = KPipeline(lang_code=voice[0])  # a = American, b = British
chunks = [np.asarray(a, dtype="float32") for _, _, a in pipe(sys.stdin.read(), voice=voice)]
sf.write(out, np.concatenate(chunks), 24000, subtype="PCM_16")
