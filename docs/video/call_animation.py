"""The animated "inside a call" segment: a script timed sentence by sentence, and the page it draws.

make_video.py narrates each sentence, then screenshots `frame(stage, local, line)` at 12 frames a
second, so every highlight lands on the spoken phrase.
"""

import json

# (spoken sentence, which stage is lit). Stage 0 = none, 1-6 = the pipeline, 7 = filler and
# barge-in, 8 = the timing badge. The third value is how many transcript lines are visible.
CALL_SCRIPT = [
    ("Here is what happens inside the call, one reply at a time.", 0, 0),
    (
        "The candidate speaks, and the call's audio reaches Senjin through a virtual audio device.",
        1,
        1,
    ),
    (
        "A voice detector waits for them to stop, about a second of quiet, and holds a little "
        "longer after a short answer in case they carry on.",
        2,
        1,
    ),
    ("Whisper turns the speech into text, in well under a second.", 3, 1),
    (
        "Then the code, not the model, decides what happens next: a follow-up, the next question, "
        "or the end.",
        4,
        1,
    ),
    ("The local model only puts that into natural words.", 5, 1),
    ("Piper speaks the reply back into the call, a sentence at a time.", 6, 2),
    (
        "If a reply is slow, a short I see fills the gap. If the candidate cuts in, Senjin stops "
        "talking.",
        7,
        2,
    ),
    (
        "In the last live call, that whole loop took about two seconds, from the end of speech to "
        "the first sound.",
        8,
        2,
    ),
]

STAGES = [
    ["Call audio", "candidate's voice, via BlackHole"],
    ["Voice detector", "Silero VAD: waits for quiet"],
    ["Speech to text", "whisper.cpp, on this Mac"],
    ["Interview code", "picks the question or follow-up"],
    ["Local model", "Ollama qwen3 8B words it"],
    ["Voice", "Piper speaks into the call"],
]

_STYLE = """<style>
.stage{position:absolute;top:250px;width:250px;height:230px;border-radius:18px;background:#1f2630;
border:4px solid #2a313c;padding:26px 22px}
.stage b{display:block;font-size:34px;margin-bottom:10px}
.stage span{font-size:25px;color:#9aa3af;line-height:1.3}
.arr{position:absolute;top:345px;font-size:44px;color:#4a5361;width:36px;text-align:center}
.dot{position:absolute;width:30px;height:30px;border-radius:50%;background:#8ab4f8;top:350px;
opacity:0;box-shadow:0 0 24px #8ab4f8}
.extra{position:absolute;top:560px;height:120px;border-radius:16px;background:#1f2630;
border:4px solid #2a313c;padding:20px 28px;font-size:30px;line-height:1.3}
.badge{position:absolute;top:545px;right:120px;width:540px;height:160px;border-radius:16px;
background:#1d3a28;border:4px solid #7ee787;padding:14px 28px;font-size:28px;line-height:1.25;
opacity:0}
.badge b{font-size:46px;color:#7ee787;display:block}
.tr{position:absolute;left:120px;right:120px;top:730px;height:260px;background:#0c0f14;
border-radius:16px;padding:26px 40px;font-size:32px;line-height:1.5;color:#c9d1d9;
border:2px solid #2a313c}
.tr div{opacity:0}.tr em{font-style:normal;color:#8ab4f8}.tr i{font-style:normal;color:#7ee787}
</style>"""

_BODY = """<h2>Inside a call</h2><h1 style="font-size:60px;margin-bottom:0">One reply, start to finish</h1>
<div id=stages></div><div id=dot class=dot></div>
<div class=extra id=ex1 style="left:120px;width:520px"><b>Filler clip</b><br>“I see.” if a reply is slow</div>
<div class=extra id=ex2 style="left:670px;width:540px"><b>Barge-in</b><br>stops talking if you cut in</div>
<div class=badge id=badge><b>about 2 s</b>end of speech to first sound<br>(one live call, 19 turns)</div>
<div class=tr><div id=l1><em>Candidate:</em> Can you repeat that question, please?</div>
<div id=l2><i>Senjin:</i> How would you automate the collection of evidence for NIST 800-53 or FedRAMP
controls during the development process, rather than gathering it manually just before an audit?</div></div>
"""

_SCRIPT = """<script>
const STAGES=__STAGES__, X0=120, W=250, GAP=36;
const sx=i=>X0+i*(W+GAP);
const host=document.getElementById('stages');
STAGES.forEach((s,i)=>{host.innerHTML+=`<div class=stage id=s${i+1} style="left:${sx(i)}px"><b>${s[0]}</b><span>${s[1]}</span></div>`;
 if(i<5)host.innerHTML+=`<div class=arr style="left:${sx(i)+W}px">›</div>`;});
function frame(stage,local,line){
 for(let i=1;i<=6;i++){const e=document.getElementById('s'+i);
  const on=(i==stage)||(stage==8);
  e.style.borderColor=on?'#8ab4f8':(i<stage&&stage<7?'#3d5a80':'#2a313c');
  e.style.background=on?'#25324a':'#1f2630';
  e.style.transform=(i==stage)?`scale(${1+0.025*Math.sin(local*6)})`:'scale(1)';}
 for(const id of ['ex1','ex2']){const e=document.getElementById(id);
  e.style.borderColor=stage==7?'#ffa657':'#2a313c';e.style.background=stage==7?'#3a2c1d':'#1f2630';}
 document.getElementById('badge').style.opacity=stage==8?1:0;
 const d=document.getElementById('dot');
 if(stage>=2&&stage<=6&&local<0.9){const f=local/0.9;
  d.style.opacity=1;d.style.left=(sx(stage-2)+W/2+(sx(stage-1)-sx(stage-2))*f-15)+'px';}
 else d.style.opacity=0;
 document.getElementById('l1').style.opacity=line>=1?1:0;
 document.getElementById('l2').style.opacity=line>=2?1:0;
}
</script>"""


def page(css: str) -> str:
    return css + _STYLE + _BODY + _SCRIPT.replace("__STAGES__", json.dumps(STAGES))
