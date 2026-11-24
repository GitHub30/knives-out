// DOM heads-up display. Everything user-provided (names) goes through textContent.

import type { SelfState, ZoneState, PlaneState } from './protocol';

export const WEAPON_NAMES = ['素手', 'P-12 ハンドガン', 'VX-9 SMG', 'AR-7 ライフル', 'SG-12 ショットガン', 'MR-3 マークスマン', 'BR-90 スナイパー'];
export const WEAPON_SHORT = ['素手', 'P-12', 'VX-9', 'AR-7', 'SG-12', 'MR-3', 'BR-90'];
export const AMMO_NAMES = ['軽弾', '中弾', 'ショットシェル', '重弾'];
export const HEAL_NAMES = ['包帯', '救急キット', 'エナジードリンク'];
export const TOWN_NAMES = ['ポートヒカリ', '旧製材所', 'キツネ村', 'レッドロック', 'サクラ丘', 'アイアンヤード', '灯台岬', 'パインバレー', 'シロ基地', 'ミズノ湖畔', 'カゼ峠', 'ホシ町', 'アカツキ港', 'モリノ集落', 'クロガネ工場', 'ユキミ台'];
const RARITY_CSS = ['#ddd', '#e8e8e8', '#6fd35a', '#4aa3ff', '#6fd35a', '#4aa3ff', '#ffc33a'];
const LEVEL_CSS = ['#777', '#e0e0e0', '#4aa3ff', '#b06bff'];

export function itemName(kind: number, amount: number): string {
  if (kind >= 1 && kind <= 6) return WEAPON_NAMES[kind];
  if (kind >= 10 && kind <= 13) return `${AMMO_NAMES[kind - 10]} ×${amount}`;
  if (kind >= 20 && kind <= 22) return `${HEAL_NAMES[kind - 20]}${amount > 1 ? ` ×${amount}` : ''}`;
  if (kind >= 30 && kind <= 32) return `ベスト Lv${kind - 29}`;
  if (kind >= 33 && kind <= 35) return `ヘルメット Lv${kind - 32}`;
  if (kind === 40) return `グレネード${amount > 1 ? ` ×${amount}` : ''}`;
  return '???';
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, parent: HTMLElement, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}

function fmtTime(t: number) {
  t = Math.max(0, Math.ceil(t));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

export interface MapMarker {
  x: number;
  z: number;
  t: number;
}

export class Hud {
  root: HTMLDivElement;
  private alive: HTMLSpanElement;
  private kills: HTMLSpanElement;
  private compass: HTMLDivElement;
  private compassStrip: HTMLDivElement;
  private mini: HTMLCanvasElement;
  private miniCtx: CanvasRenderingContext2D;
  private zoneText: HTMLDivElement;
  private feed: HTMLDivElement;
  private cross: HTMLDivElement;
  private hitm: HTMLDivElement;
  private dmgLayer: HTMLDivElement;
  private dirLayer: HTMLDivElement;
  private toast: HTMLDivElement;
  private prompt: HTMLDivElement;
  private progress: HTMLDivElement;
  private progressLabel: HTMLDivElement;
  private hpFill: HTMLDivElement;
  private hpBoost: HTMLDivElement;
  private hpText: HTMLSpanElement;
  private vest: HTMLDivElement;
  private helm: HTMLDivElement;
  private slotEls: HTMLDivElement[] = [];
  private healEls: HTMLDivElement[] = [];
  private nadeEl: HTMLDivElement;
  private banner: HTMLDivElement;
  private scope: HTMLDivElement;
  private hurtV: HTMLDivElement;
  private zoneV: HTMLDivElement;
  private fullmap: HTMLDivElement;
  private fullCanvas: HTMLCanvasElement;
  private spect: HTMLDivElement;
  private spectName: HTMLSpanElement;
  private net: HTMLDivElement;
  private mapImg: HTMLCanvasElement | null = null;
  private cache = new Map<string, string>();
  private bannerT = 0;
  private toastT = 0;
  private hitT = 0;
  mapOpen = false;
  onSlot: (s: number) => void = () => {};
  onHeal: (k: number) => void = () => {};
  onSpectNext: () => void = () => {};
  onLeave: () => void = () => {};
  onMapClose: () => void = () => {};

  constructor(parent: HTMLElement) {
    const root = el('div', 'hud', parent);
    root.id = 'hud';
    this.root = root;
    const tl = el('div', 'hud-tl', root);
    const a = el('div', 'pill', tl);
    el('span', 'pill-k', a, '残り');
    this.alive = el('span', 'pill-v', a, '-');
    const k = el('div', 'pill', tl);
    el('span', 'pill-k', k, 'キル');
    this.kills = el('span', 'pill-v', k, '0');

    this.compass = el('div', 'compass', root);
    this.compassStrip = el('div', 'compass-strip', this.compass);
    const labels: Record<number, string> = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
    for (let d = -180; d <= 540; d += 15) {
      const n = ((d % 360) + 360) % 360;
      const m = el('span', labels[n] ? 'cmark major' : 'cmark', this.compassStrip, labels[n] ?? String(n));
      m.style.left = `${(d + 180) * 4}px`;
    }
    el('div', 'compass-needle', this.compass);

    const mm = el('div', 'minimap', root);
    this.mini = el('canvas', '', mm);
    this.mini.width = 200;
    this.mini.height = 200;
    this.miniCtx = this.mini.getContext('2d')!;
    this.zoneText = el('div', 'zone-text', root);

    this.feed = el('div', 'killfeed', root);
    this.cross = el('div', 'crosshair', root);
    for (const c of ['t', 'b', 'l', 'r', 'd']) el('i', `ch-${c}`, this.cross);
    this.hitm = el('div', 'hitmarker', root);
    for (let i = 0; i < 4; i++) el('i', '', this.hitm);
    this.dmgLayer = el('div', 'dmg-layer', root);
    this.dirLayer = el('div', 'dir-layer', root);
    this.toast = el('div', 'toast', root);
    this.prompt = el('div', 'prompt', root);
    this.progress = el('div', 'progress', root);
    this.progressLabel = el('div', 'progress-label', this.progress);
    el('div', 'progress-ring', this.progress);

    const bottom = el('div', 'hud-bottom', root);
    const slots = el('div', 'slots', bottom);
    for (let s = 0; s < 3; s++) {
      const d = el('div', 'slot', slots);
      d.dataset.slot = String(s);
      el('div', 'slot-name', d);
      el('div', 'slot-ammo', d);
      d.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        this.onSlot(s);
      });
      this.slotEls.push(d);
    }
    const hpRow = el('div', 'hp-row', bottom);
    this.vest = el('div', 'gear', hpRow, 'V');
    const hp = el('div', 'hp', hpRow);
    this.hpFill = el('div', 'hp-fill', hp);
    this.hpBoost = el('div', 'hp-boost', hp);
    this.hpText = el('span', 'hp-text', hp, '100');
    this.helm = el('div', 'gear', hpRow, 'H');
    const heals = el('div', 'heals', bottom);
    for (let h = 0; h < 3; h++) {
      const d = el('div', `heal heal-${h}`, heals);
      el('span', 'heal-icon', d, ['✚', '⛑', '⚡'][h]);
      el('span', 'heal-n', d, '0');
      d.title = HEAL_NAMES[h];
      d.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        this.onHeal(h + 1);
      });
      this.healEls.push(d);
    }
    this.nadeEl = el('div', 'heal nade', heals);
    el('span', 'heal-icon', this.nadeEl, '●');
    el('span', 'heal-n', this.nadeEl, '0');

    this.banner = el('div', 'banner', root);
    this.scope = el('div', 'scope', root);
    this.hurtV = el('div', 'hurt-vignette', root);
    this.zoneV = el('div', 'zone-vignette', root);
    this.fullmap = el('div', 'fullmap', root);
    this.fullCanvas = el('canvas', '', this.fullmap);
    const closeBtn = el('button', 'fullmap-close', this.fullmap, '✕');
    closeBtn.addEventListener('click', () => this.onMapClose());
    this.spect = el('div', 'spect', root);
    el('span', 'spect-k', this.spect, '観戦中');
    this.spectName = el('span', 'spect-name', this.spect);
    const nx = el('button', 'btn small', this.spect, '次のプレイヤー');
    nx.addEventListener('click', () => this.onSpectNext());
    const lv = el('button', 'btn small ghost', this.spect, 'メニューへ');
    lv.addEventListener('click', () => this.onLeave());
    this.net = el('div', 'netinfo', root);
  }

  show(v: boolean) {
    this.root.style.display = v ? 'block' : 'none';
  }

  private set(key: string, value: string, apply: () => void) {
    if (this.cache.get(key) === value) return;
    this.cache.set(key, value);
    apply();
  }

  setMap(rgba: Uint8Array, size: number) {
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    const ctx = c.getContext('2d')!;
    const img = new ImageData(Uint8ClampedArray.from(rgba.subarray(0, size * size * 4)), size, size);
    ctx.putImageData(img, 0, 0);
    this.mapImg = c;
  }

  setNet(text: string) {
    this.set('net', text, () => (this.net.textContent = text));
  }

  updateTop(alive: number, kills: number) {
    this.set('alive', String(alive), () => (this.alive.textContent = String(alive)));
    this.set('kills', String(kills), () => (this.kills.textContent = String(kills)));
  }

  updateCompass(yaw: number) {
    // heading: 0 = north (-Z), 90 = east (+X)
    let h = 180 - (yaw * 180) / Math.PI;
    h = ((h % 360) + 360) % 360;
    const w = this.compass.clientWidth || 300;
    this.compassStrip.style.transform = `translateX(${-(h + 180) * 4 + w / 2}px)`;
  }

  updateZone(z: ZoneState, me: { x: number; z: number } | null, phase: number) {
    let text = '';
    let warn = false;
    if (phase === 1) {
      if (z.done) text = '最終エリア';
      else if (z.shrinking) text = `⚠ 安全地帯 縮小中 ${fmtTime(z.timer)}`;
      else text = `安全地帯 縮小まで ${fmtTime(z.timer)}`;
      if (me) {
        const d = Math.hypot(me.x - z.cx, me.z - z.cz) - z.r;
        if (d > 0) {
          text += ` ・ 圏外 ${Math.round(d)}m`;
          warn = true;
        }
      }
    }
    this.set('zone', text + warn, () => {
      this.zoneText.textContent = text;
      this.zoneText.classList.toggle('warn', warn);
    });
    this.zoneV.style.opacity = warn ? '1' : '0';
  }

  updateSelf(s: SelfState, wdefMag: number[], displayCur: number, predMag: number, ammoType: number[]) {
    const hp = Math.max(0, Math.round(s.health));
    this.set('hp', String(hp), () => {
      this.hpFill.style.width = `${hp}%`;
      this.hpFill.style.background = hp > 60 ? '#f1f1f1' : hp > 30 ? '#ffb347' : '#ff4d4d';
      this.hpText.textContent = String(hp);
    });
    const boost = Math.round(s.boost);
    this.set('boost', String(boost), () => (this.hpBoost.style.width = `${Math.min(100, boost)}%`));
    const gear = (e: HTMLDivElement, lv: number, dur: number, label: string) => {
      const key = `${label}${lv}:${Math.round(dur / 10)}`;
      this.set(label, key, () => {
        e.textContent = lv > 0 ? `${label}${lv}` : label;
        e.style.borderColor = LEVEL_CSS[lv];
        e.style.color = lv > 0 ? LEVEL_CSS[lv] : '#777';
        e.style.background = lv > 0 ? `linear-gradient(to top, rgba(255,255,255,0.18) ${Math.min(100, dur)}%, transparent ${Math.min(100, dur)}%)` : 'transparent';
      });
    };
    gear(this.vest, s.vest, s.vestDur, 'V');
    gear(this.helm, s.helm, s.helmDur, 'H');
    for (let i = 0; i < 3; i++) {
      const e = this.slotEls[i];
      const kind = i < 2 ? s.slots[i].kind : 0;
      const name = i < 2 ? (kind ? WEAPON_SHORT[kind] : '—') : '素手';
      let ammo = '';
      if (i < 2 && kind) {
        const mag = i === displayCur ? predMag : s.slots[i].mag;
        ammo = `${mag}/${wdefMag[kind]}  ${s.ammo[ammoType[kind]] ?? 0}`;
      }
      const key = `${name}|${ammo}|${displayCur === i}|${kind}`;
      this.set(`slot${i}`, key, () => {
        (e.children[0] as HTMLElement).textContent = name;
        (e.children[1] as HTMLElement).textContent = ammo;
        e.classList.toggle('active', displayCur === i);
        e.style.borderColor = kind ? RARITY_CSS[kind] : '';
      });
    }
    for (let h = 0; h < 3; h++) {
      const n = s.heals[h];
      this.set(`heal${h}`, String(n), () => {
        (this.healEls[h].children[1] as HTMLElement).textContent = String(n);
        this.healEls[h].classList.toggle('empty', n === 0);
      });
    }
    this.set('nades', String(s.nades), () => {
      (this.nadeEl.children[1] as HTMLElement).textContent = String(s.nades);
      this.nadeEl.classList.toggle('empty', s.nades === 0);
    });
  }

  crosshair(spreadPx: number, visible: boolean) {
    const g = Math.round(Math.min(80, Math.max(3, spreadPx)));
    this.set('cross', `${g}|${visible}`, () => {
      this.cross.style.setProperty('--gap', `${g}px`);
      this.cross.style.display = visible ? 'block' : 'none';
    });
  }

  hitmarker(head: boolean, kill: boolean) {
    this.hitm.className = `hitmarker show${head ? ' head' : ''}${kill ? ' kill' : ''}`;
    this.hitT = 0.25;
  }

  damageNumber(x: number, y: number, dmg: number, head: boolean, armor: boolean) {
    const d = el('div', `dmgnum${head ? ' head' : ''}${armor ? ' armor' : ''}`, this.dmgLayer, String(dmg));
    d.style.left = `${x + (Math.random() - 0.5) * 30}px`;
    d.style.top = `${y - 10}px`;
    setTimeout(() => d.remove(), 900);
  }

  damageDir(angle: number) {
    const d = el('div', 'dmgdir', this.dirLayer);
    d.style.transform = `rotate(${angle}rad)`;
    setTimeout(() => d.remove(), 1200);
  }

  hurtFlash(amount: number) {
    this.hurtV.style.transition = 'none';
    this.hurtV.style.opacity = String(Math.min(0.85, 0.25 + amount / 60));
    void this.hurtV.offsetWidth;
    this.hurtV.style.transition = 'opacity 0.7s ease-out';
    this.hurtV.style.opacity = '0';
  }

  killfeed(killer: string, victim: string, weapon: string, mine: 'me' | 'victim' | '') {
    const row = el('div', `kf${mine ? ` kf-${mine}` : ''}`, this.feed);
    el('span', 'kf-a', row, killer);
    el('span', 'kf-w', row, ` [${weapon}] `);
    el('span', 'kf-v', row, victim);
    while (this.feed.children.length > 5) this.feed.firstChild?.remove();
    setTimeout(() => row.classList.add('fade'), 5000);
    setTimeout(() => row.remove(), 6000);
  }

  showToast(text: string) {
    this.toast.textContent = text;
    this.toast.classList.add('show');
    this.toastT = 1.6;
  }

  showBanner(text: string, secs = 2.5, cls = '') {
    this.banner.textContent = text;
    this.banner.className = `banner show ${cls}`;
    this.bannerT = secs;
  }

  setPrompt(text: string | null) {
    this.set('prompt', text ?? '', () => {
      this.prompt.textContent = text ?? '';
      this.prompt.style.display = text ? 'block' : 'none';
    });
  }

  setProgress(label: string | null, frac: number) {
    this.set('prog', label ? `${label}${Math.round(frac * 50)}` : '', () => {
      this.progress.style.display = label ? 'flex' : 'none';
      this.progressLabel.textContent = label ?? '';
      this.progress.style.setProperty('--p', `${Math.round(frac * 100)}%`);
    });
  }

  setScope(on: boolean) {
    this.set('scope', String(on), () => this.scope.classList.toggle('on', on));
  }

  setSpectate(name: string | null) {
    this.set('spect', name ?? '', () => {
      this.spect.style.display = name !== null ? 'flex' : 'none';
      this.spectName.textContent = name ?? '';
    });
  }

  tick(dt: number) {
    if (this.bannerT > 0) {
      this.bannerT -= dt;
      if (this.bannerT <= 0) this.banner.classList.remove('show');
    }
    if (this.toastT > 0) {
      this.toastT -= dt;
      if (this.toastT <= 0) this.toast.classList.remove('show');
    }
    if (this.hitT > 0) {
      this.hitT -= dt;
      if (this.hitT <= 0) this.hitm.classList.remove('show');
    }
  }

  drawMinimap(px: number, pz: number, yaw: number, zone: ZoneState | null, plane: PlaneState | null, markers: MapMarker[], now: number, mapSize: number) {
    const ctx = this.miniCtx;
    const W = this.mini.width;
    const R = 170; // meters shown from center to edge
    const s = W / (2 * R);
    ctx.clearRect(0, 0, W, W);
    ctx.save();
    ctx.beginPath();
    ctx.arc(W / 2, W / 2, W / 2 - 1, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = '#1e4f78';
    ctx.fillRect(0, 0, W, W);
    if (this.mapImg) {
      const k = this.mapImg.width / mapSize;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.mapImg, (px - R) * k, (pz - R) * k, 2 * R * k, 2 * R * k, 0, 0, W, W);
    }
    const tx = (x: number) => W / 2 + (x - px) * s;
    const tz = (z: number) => W / 2 + (z - pz) * s;
    this.drawZone(ctx, zone, tx, tz, s, 2);
    if (plane && plane.active) this.drawPlane(ctx, plane, tx, tz, mapSize);
    for (const m of markers) {
      const age = now - m.t;
      if (age > 2.5) continue;
      ctx.fillStyle = `rgba(255,70,50,${1 - age / 2.5})`;
      const x = tx(m.x), y = tz(m.z);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    this.arrow(ctx, W / 2, W / 2, yaw, 9, '#ffd54a');
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(W / 2, W / 2, W / 2 - 1, 0, Math.PI * 2);
    ctx.stroke();
  }

  private drawZone(ctx: CanvasRenderingContext2D, zone: ZoneState | null, tx: (x: number) => number, tz: (z: number) => number, s: number, lw: number) {
    if (!zone) return;
    if (zone.nr < zone.r - 1 || !zone.done) {
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.arc(tx(zone.nx), tz(zone.nz), Math.max(1, zone.nr * s), 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = 'rgba(70,150,255,0.95)';
    ctx.lineWidth = lw + 1;
    ctx.beginPath();
    ctx.arc(tx(zone.cx), tz(zone.cz), Math.max(1, zone.r * s), 0, Math.PI * 2);
    ctx.stroke();
  }

  private drawPlane(ctx: CanvasRenderingContext2D, p: PlaneState, tx: (x: number) => number, tz: (z: number) => number, mapSize: number) {
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(tx(p.sx), tz(p.sz));
    ctx.lineTo(tx(p.sx + p.dx * mapSize * 1.7), tz(p.sz + p.dz * mapSize * 1.7));
    ctx.stroke();
    ctx.setLineDash([]);
    const x = p.sx + p.dx * p.t, z = p.sz + p.dz * p.t;
    this.arrow(ctx, tx(x), tz(z), Math.atan2(p.dx, p.dz), 8, '#fff');
  }

  private arrow(ctx: CanvasRenderingContext2D, x: number, y: number, yaw: number, r: number, col: string) {
    // map: +x right, +z down. forward (sin yaw, cos yaw)
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    ctx.fillStyle = col;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + fx * r, y + fz * r);
    ctx.lineTo(x - fx * r * 0.6 + fz * r * 0.6, y - fz * r * 0.6 - fx * r * 0.6);
    ctx.lineTo(x - fx * r * 0.25, y - fz * r * 0.25);
    ctx.lineTo(x - fx * r * 0.6 - fz * r * 0.6, y - fz * r * 0.6 + fx * r * 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  openMap(open: boolean) {
    this.mapOpen = open;
    this.fullmap.style.display = open ? 'flex' : 'none';
  }

  drawFullMap(px: number | null, pz: number, yaw: number, zone: ZoneState | null, plane: PlaneState | null, towns: { x: number; z: number; name: number }[], mapSize: number) {
    if (!this.mapOpen) return;
    const c = this.fullCanvas;
    const size = Math.floor(Math.min(window.innerWidth, window.innerHeight) * 0.92 * Math.min(2, window.devicePixelRatio || 1));
    if (c.width !== size) {
      c.width = size;
      c.height = size;
    }
    const ctx = c.getContext('2d')!;
    const s = size / mapSize;
    ctx.fillStyle = '#1e4f78';
    ctx.fillRect(0, 0, size, size);
    if (this.mapImg) ctx.drawImage(this.mapImg, 0, 0, size, size);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 8; g++) {
      const v = (g * mapSize) / 8 * s;
      ctx.beginPath();
      ctx.moveTo(v, 0);
      ctx.lineTo(v, size);
      ctx.moveTo(0, v);
      ctx.lineTo(size, v);
      ctx.stroke();
    }
    const tx = (x: number) => x * s;
    this.drawZone(ctx, zone, tx, tx, s, Math.max(2, size / 300));
    if (plane && plane.active) this.drawPlane(ctx, plane, tx, tx, mapSize);
    ctx.font = `bold ${Math.max(11, Math.round(size / 48))}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    for (const t of towns) {
      const name = TOWN_NAMES[t.name % TOWN_NAMES.length];
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
      ctx.strokeText(name, tx(t.x), tx(t.z));
      ctx.fillStyle = '#fff';
      ctx.fillText(name, tx(t.x), tx(t.z));
    }
    if (px !== null) this.arrow(ctx, tx(px), tx(pz), yaw, Math.max(9, size / 60), '#ffd54a');
  }
}
