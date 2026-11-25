// Procedural WebAudio sound effects (no audio assets needed).

const SHOT: { f: number; q: number; decay: number; thump: number; gain: number }[] = [
  { f: 900, q: 0.7, decay: 0.08, thump: 0.3, gain: 0.5 }, // fists
  { f: 2600, q: 0.8, decay: 0.12, thump: 0.6, gain: 0.8 }, // pistol
  { f: 3400, q: 0.9, decay: 0.08, thump: 0.5, gain: 0.7 }, // smg
  { f: 2200, q: 0.8, decay: 0.15, thump: 0.9, gain: 0.95 }, // ar
  { f: 1300, q: 0.6, decay: 0.32, thump: 1.3, gain: 1.1 }, // shotgun
  { f: 1800, q: 0.7, decay: 0.24, thump: 1.1, gain: 1.0 }, // dmr
  { f: 1500, q: 0.6, decay: 0.45, thump: 1.4, gain: 1.15 }, // sniper
];

export class Sound {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private noise!: AudioBuffer;
  private voices = 0;
  private loops: Record<string, { src: AudioScheduledSourceNode; gain: GainNode; filter?: BiquadFilterNode; osc?: OscillatorNode }> = {};
  volume = 0.7;

  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AC) return;
      this.ctx = new AC();
      const comp = this.ctx.createDynamicsCompressor();
      comp.threshold.value = -14;
      comp.ratio.value = 6;
      this.master = this.ctx.createGain();
      this.master.gain.value = this.volume;
      this.master.connect(comp).connect(this.ctx.destination);
      const len = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
  }

  setVolume(v: number) {
    this.volume = v;
    if (this.ctx) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
  }

  suspend(s: boolean) {
    if (!this.ctx) return;
    if (s) void this.ctx.suspend();
    else void this.ctx.resume();
  }

  private ready() {
    return this.ctx && this.ctx.state === 'running' && this.voices < 40;
  }

  private out(pan: number, gain: number): GainNode {
    const c = this.ctx!;
    const g = c.createGain();
    g.gain.value = gain;
    if (c.createStereoPanner) {
      const p = c.createStereoPanner();
      p.pan.value = Math.max(-1, Math.min(1, pan));
      g.connect(p).connect(this.master);
    } else g.connect(this.master);
    return g;
  }

  private burst(dest: AudioNode, t: number, dur: number, type: BiquadFilterType, freq: number, q: number, peak: number, attack = 0.002) {
    const c = this.ctx!;
    const src = c.createBufferSource();
    src.buffer = this.noise;
    const f = c.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(f).connect(g).connect(dest);
    src.start(t, Math.random() * 0.5);
    src.stop(t + dur + 0.05);
    this.voices++;
    src.onended = () => this.voices--;
  }

  private tone(dest: AudioNode, t: number, dur: number, f0: number, f1: number, peak: number, type: OscillatorType = 'sine') {
    const c = this.ctx!;
    const o = c.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t + dur);
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(dest);
    o.start(t);
    o.stop(t + dur + 0.05);
    this.voices++;
    o.onended = () => this.voices--;
  }

  shot(weapon: number, dist: number, pan: number) {
    if (!this.ready()) return;
    const s = SHOT[weapon] ?? SHOT[3];
    const c = this.ctx!;
    const t = c.currentTime;
    const att = 1 / (1 + dist / 22);
    if (att < 0.02) return;
    const dest = this.out(pan * Math.min(1, dist / 8), s.gain * att);
    const muffle = 600 + s.f * 3 * Math.exp(-dist / 140);
    this.burst(dest, t, s.decay * (1 + dist / 300), 'lowpass', muffle, s.q, 0.9);
    this.burst(dest, t, s.decay * 0.5, 'bandpass', s.f, 1.2, 0.6);
    if (s.thump > 0 && dist < 120) this.tone(dest, t, 0.12 * s.thump, 150, 45, 0.7 * s.thump);
    if (dist > 60) this.burst(dest, t + 0.08, s.decay * 2.5, 'lowpass', 500, 0.5, 0.25);
  }

  explosion(dist: number, pan: number) {
    if (!this.ready()) return;
    const t = this.ctx!.currentTime;
    const att = 1.6 / (1 + dist / 30);
    const dest = this.out(pan, att);
    this.burst(dest, t, 1.2, 'lowpass', 900, 0.5, 1);
    this.tone(dest, t, 0.6, 90, 28, 1);
    this.burst(dest, t + 0.05, 1.6, 'lowpass', 300, 0.3, 0.5);
  }

  hit(head: boolean, kill: boolean) {
    if (!this.ready()) return;
    const t = this.ctx!.currentTime;
    const dest = this.out(0, 0.35);
    if (kill) {
      this.tone(dest, t, 0.09, 1200, 1200, 0.6, 'square');
      this.tone(dest, t + 0.08, 0.18, 1800, 1700, 0.6, 'square');
    } else if (head) {
      this.tone(dest, t, 0.12, 2400, 2100, 0.6, 'triangle');
      this.burst(dest, t, 0.05, 'highpass', 5000, 1, 0.4);
    } else {
      this.tone(dest, t, 0.05, 1600, 1500, 0.45, 'triangle');
    }
  }

  armorHit() {
    if (!this.ready()) return;
    const t = this.ctx!.currentTime;
    this.burst(this.out(0, 0.3), t, 0.08, 'bandpass', 4200, 4, 0.7);
  }

  hurt() {
    if (!this.ready()) return;
    const t = this.ctx!.currentTime;
    const dest = this.out(0, 0.5);
    this.tone(dest, t, 0.16, 180, 90, 0.8);
    this.burst(dest, t, 0.1, 'lowpass', 700, 0.5, 0.5);
  }

  footstep(dist: number, pan: number, water: boolean) {
    if (!this.ready() || dist > 45) return;
    const t = this.ctx!.currentTime;
    const dest = this.out(pan, 0.22 / (1 + dist / 6));
    if (water) this.burst(dest, t, 0.18, 'bandpass', 900, 0.8, 0.6, 0.02);
    else this.burst(dest, t, 0.07, 'bandpass', 380 + Math.random() * 200, 1.4, 0.8);
  }

  ui(kind: 'click' | 'pickup' | 'reload' | 'heal' | 'zone' | 'empty' | 'jump' | 'land' | 'chute' | 'throw' | 'win' | 'lose') {
    if (!this.ready()) return;
    const t = this.ctx!.currentTime;
    const dest = this.out(0, 0.35);
    switch (kind) {
      case 'click':
        this.tone(dest, t, 0.04, 900, 900, 0.3, 'square');
        break;
      case 'pickup':
        this.tone(dest, t, 0.06, 700, 1100, 0.4, 'triangle');
        this.tone(dest, t + 0.05, 0.08, 1100, 1500, 0.3, 'triangle');
        break;
      case 'reload':
        this.burst(dest, t, 0.05, 'bandpass', 2500, 3, 0.6);
        this.burst(dest, t + 0.35, 0.06, 'bandpass', 1800, 3, 0.7);
        break;
      case 'heal':
        this.burst(dest, t, 0.4, 'bandpass', 3000, 0.7, 0.2, 0.08);
        break;
      case 'zone':
        this.tone(dest, t, 0.12, 520, 520, 0.35, 'sine');
        break;
      case 'empty':
        this.burst(dest, t, 0.03, 'highpass', 3000, 2, 0.5);
        break;
      case 'jump':
        this.burst(dest, t, 0.5, 'lowpass', 1200, 0.5, 0.4, 0.05);
        break;
      case 'land':
        this.burst(dest, t, 0.12, 'lowpass', 400, 0.7, 0.8);
        break;
      case 'chute':
        this.burst(dest, t, 0.35, 'bandpass', 700, 0.6, 0.9, 0.02);
        break;
      case 'throw':
        this.burst(dest, t, 0.12, 'bandpass', 1500, 1, 0.4, 0.01);
        break;
      case 'win':
        [523, 659, 784, 1047].forEach((f, i) => this.tone(dest, t + i * 0.13, 0.3, f, f, 0.4, 'triangle'));
        break;
      case 'lose':
        [392, 330, 262].forEach((f, i) => this.tone(dest, t + i * 0.18, 0.35, f, f * 0.98, 0.35, 'triangle'));
        break;
    }
  }

  /** Continuous loops: plane engine / freefall wind. level 0 stops. */
  loop(name: 'plane' | 'wind' | 'engine', level: number, pitch = 0) {
    if (!this.ctx) return;
    const c = this.ctx;
    let l = this.loops[name];
    // Stopping must work even while the context is suspended/interrupted (iOS).
    if (level > 0.001 && c.state !== 'running') return;
    if (level <= 0.001) {
      if (l) {
        l.gain.gain.setTargetAtTime(0, c.currentTime, 0.2);
        const src = l.src;
        const osc = l.osc;
        setTimeout(() => {
          try {
            src.stop();
            osc?.stop();
          } catch {
            /* ignore */
          }
        }, 800);
        delete this.loops[name];
      }
      return;
    }
    if (!l) {
      const src = c.createBufferSource();
      src.buffer = this.noise;
      src.loop = true;
      const filter = c.createBiquadFilter();
      filter.type = name === 'wind' ? 'bandpass' : 'lowpass';
      filter.frequency.value = name === 'plane' ? 260 : name === 'engine' ? 180 : 800;
      filter.Q.value = name === 'plane' ? 0.7 : 0.4;
      const gain = c.createGain();
      gain.gain.value = 0;
      src.connect(filter).connect(gain).connect(this.master);
      src.start();
      let osc: OscillatorNode | undefined;
      if (name !== 'wind') {
        osc = c.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.value = name === 'engine' ? 42 : 58;
        const og = c.createGain();
        og.gain.value = name === 'engine' ? 0.5 : 0.12;
        const of = c.createBiquadFilter();
        of.type = 'lowpass';
        of.frequency.value = 300;
        osc.connect(of).connect(og).connect(gain);
        osc.start();
      }
      l = this.loops[name] = { src, gain, filter, osc };
    }
    l.gain.gain.setTargetAtTime(level * (name === 'plane' ? 0.5 : name === 'engine' ? 0.3 : 0.35), c.currentTime, 0.15);
    // engine note: idle to redline with `pitch` 0..1
    if (name === 'engine' && l.osc) l.osc.frequency.setTargetAtTime(42 + pitch * 110, c.currentTime, 0.08);
    if (name === 'wind' && l.filter) l.filter.frequency.setTargetAtTime(500 + level * 1200, c.currentTime, 0.2);
  }
}
