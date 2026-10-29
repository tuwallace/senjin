"""Init script injected into every page: pins the browser's microphone and speaker.

Meet picks devices itself, so we override at the API level instead of clicking through its UI:
getUserMedia is forced onto the virtual microphone, and every media element's output is
routed to the virtual speaker with setSinkId. Device ids are found by label substring.
"""

import json

TEMPLATE = r"""
(() => {
  const MIC = __MIC__, SPK = __SPK__;
  const md = navigator.mediaDevices;
  if (!md) return;
  const origGUM = md.getUserMedia.bind(md);
  const origEnum = md.enumerateDevices.bind(md);
  let cached = null;

  async function resolve() {
    if (cached) return cached;
    let probe = null;
    try { probe = await origGUM({audio: true}); } catch (e) {}  // labels stay empty until a stream opens
    const devices = await origEnum();
    if (probe) probe.getTracks().forEach(t => t.stop());
    const find = (kind, name) => {
      if (!name) return undefined;
      const hit = devices.find(d => d.kind === kind && d.label.toLowerCase().includes(name.toLowerCase()));
      return hit ? hit.deviceId : undefined;
    };
    cached = {mic: find('audioinput', MIC), spk: find('audiooutput', SPK)};
    window.__senjin = {ids: cached, devices: devices.map(d => ({kind: d.kind, label: d.label}))};
    return cached;
  }

  md.getUserMedia = async (constraints) => {
    if (constraints && constraints.audio) {
      const {mic} = await resolve();
      if (mic) {
        const base = constraints.audio === true ? {} : constraints.audio;
        constraints = {...constraints, audio: {...base, deviceId: {exact: mic}}};
      }
    }
    return origGUM(constraints);
  };

  // Meet's remote audio may sit on detached elements or in an AudioContext, so track both.
  const tracked = new Set();
  const contexts = new Set();
  const routeOne = async el => {
    const {spk} = await resolve();
    if (spk && el.setSinkId && el.sinkId !== spk) { try { await el.setSinkId(spk); } catch (e) {} }
  };
  const routeContext = async ctx => {
    const {spk} = await resolve();
    if (spk && ctx.setSinkId && ctx.sinkId !== spk) { try { await ctx.setSinkId(spk); } catch (e) {} }
  };
  async function routeOutputs() {
    for (const el of document.querySelectorAll('audio, video')) tracked.add(el);
    for (const el of tracked) await routeOne(el);
    for (const ctx of contexts) await routeContext(ctx);
  }

  const srcDesc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
  if (srcDesc) {
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      configurable: true, enumerable: srcDesc.enumerable, get: srcDesc.get,
      set(value) { tracked.add(this); srcDesc.set.call(this, value); routeOne(this); },
    });
  }
  const origPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = async function (...args) {
    tracked.add(this);
    await routeOne(this);
    return origPlay.apply(this, args);
  };
  for (const name of ['AudioContext', 'webkitAudioContext']) {
    const Orig = window[name];
    if (!Orig) continue;
    window[name] = new Proxy(Orig, {
      construct(target, args, newTarget) {
        const ctx = Reflect.construct(target, args, newTarget);
        contexts.add(ctx);
        routeContext(ctx);
        return ctx;
      },
    });
  }
  window.__senjinInspect = () => ({
    ids: cached,
    elements: [...tracked].map(e => ({tag: e.tagName, sinkId: e.sinkId, paused: e.paused,
      muted: e.muted, audioTracks: e.srcObject && e.srcObject.getAudioTracks ? e.srcObject.getAudioTracks().length : 0,
      inDom: e.isConnected})),
    contexts: [...contexts].map(c => ({state: c.state, sinkId: typeof c.sinkId === 'string' ? c.sinkId : 'none-object'})),
  });
  setInterval(routeOutputs, 1000);  // catches anything created before a sink was known
})();
"""


def init_script(mic_name: str | None, speaker_name: str | None) -> str:
    """JS pinning the microphone and speaker to devices whose label contains these names."""
    return TEMPLATE.replace("__MIC__", json.dumps(mic_name)).replace(
        "__SPK__", json.dumps(speaker_name)
    )
