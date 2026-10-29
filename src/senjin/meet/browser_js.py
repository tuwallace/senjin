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

  async function routeOutputs() {
    const {spk} = await resolve();
    if (!spk) return;
    for (const el of document.querySelectorAll('audio, video')) {
      if (el.sinkId !== spk && el.setSinkId) { try { await el.setSinkId(spk); } catch (e) {} }
    }
  }
  const origPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = async function (...args) {
    const {spk} = await resolve();
    if (spk && this.setSinkId && this.sinkId !== spk) { try { await this.setSinkId(spk); } catch (e) {} }
    return origPlay.apply(this, args);
  };
  setInterval(routeOutputs, 1000);  // catches autoplay elements Meet creates for remote audio
})();
"""


def init_script(mic_name: str | None, speaker_name: str | None) -> str:
    """JS pinning the microphone and speaker to devices whose label contains these names."""
    return TEMPLATE.replace("__MIC__", json.dumps(mic_name)).replace(
        "__SPK__", json.dumps(speaker_name)
    )
