// Mobile touch controls: floating joystick with sprint lock, swipe-to-look,
// dual fire buttons (fire while dragging to aim), ADS, jump, crouch, reload, grenade, pickup.

import type { Input } from './input';

const ICONS: Record<string, string> = {
  fire: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2v6M12 16v6M2 12h6M16 12h6" stroke="currentColor" stroke-width="2"/></svg>',
  ads: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="2" fill="currentColor"/><path d="M12 4v4M12 16v4M4 12h4M16 12h4" stroke="currentColor" stroke-width="1.5"/></svg>',
  jump: '<svg viewBox="0 0 24 24"><path d="M12 4l7 8h-4v7H9v-7H5z" fill="currentColor"/></svg>',
  crouch: '<svg viewBox="0 0 24 24"><path d="M12 20l-7-8h4V5h6v7h4z" fill="currentColor"/></svg>',
  reload: '<svg viewBox="0 0 24 24"><path d="M19 12a7 7 0 1 1-2.05-4.95" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M20 3v6h-6z" fill="currentColor"/></svg>',
  nade: '<svg viewBox="0 0 24 24"><circle cx="12" cy="14" r="6" fill="currentColor"/><rect x="10" y="4" width="4" height="4" fill="currentColor"/><path d="M14 5h4" stroke="currentColor" stroke-width="2"/></svg>',
  map: '<svg viewBox="0 0 24 24"><path d="M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 4v14M15 6v14" stroke="currentColor" stroke-width="1.5"/></svg>',
  menu: '<svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2.4"/></svg>',
  heal: '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="12" rx="2" fill="currentColor"/><path d="M12 9.5v7M8.5 13h7" stroke="#c22" stroke-width="2.4"/></svg>',
  lock: '<svg viewBox="0 0 24 24"><rect x="6" y="11" width="12" height="9" rx="2" fill="currentColor"/><path d="M8.5 11V8a3.5 3.5 0 0 1 7 0v3" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
};

export type TouchButton = 'fire' | 'fireL' | 'ads' | 'jump' | 'crouch' | 'reload' | 'nade' | 'pickup' | 'map' | 'menu' | 'heal';

export class TouchControls {
  readonly root: HTMLDivElement;
  private input: Input;
  private btns = new Map<TouchButton, HTMLButtonElement>();
  private joyBase: HTMLDivElement;
  private joyKnob: HTMLDivElement;
  private lockIcon: HTMLDivElement;
  private joyId = -1;
  private joyOrigin = [0, 0];
  private sprintLocked = false;
  private looks = new Map<number, [number, number]>();
  private held = new Map<number, TouchButton>();
  private flying = false;
  lookScale = 1.6;
  onButton: (b: TouchButton) => void = () => {};

  constructor(parent: HTMLElement, input: Input) {
    this.input = input;
    const root = document.createElement('div');
    root.id = 'touch';
    parent.appendChild(root);
    this.root = root;

    this.joyBase = document.createElement('div');
    this.joyBase.className = 'joy-base';
    this.joyKnob = document.createElement('div');
    this.joyKnob.className = 'joy-knob';
    this.joyBase.appendChild(this.joyKnob);
    this.lockIcon = document.createElement('div');
    this.lockIcon.className = 'joy-lock';
    this.lockIcon.innerHTML = ICONS.lock;
    this.joyBase.appendChild(this.lockIcon);
    root.appendChild(this.joyBase);

    const mk = (id: TouchButton, icon: string, label: string) => {
      const b = document.createElement('button');
      b.className = `tbtn tb-${id}`;
      b.dataset.btn = id;
      b.setAttribute('aria-label', label);
      b.innerHTML = icon;
      if (id === 'pickup') {
        const s = document.createElement('span');
        s.className = 'tb-label';
        b.appendChild(s);
      }
      root.appendChild(b);
      this.btns.set(id, b);
    };
    mk('fire', ICONS.fire, '射撃');
    mk('fireL', ICONS.fire, '射撃');
    mk('ads', ICONS.ads, 'スコープ');
    mk('jump', ICONS.jump, 'ジャンプ');
    mk('crouch', ICONS.crouch, 'しゃがむ');
    mk('reload', ICONS.reload, 'リロード');
    mk('nade', ICONS.nade, 'グレネード');
    mk('heal', ICONS.heal, '回復');
    mk('pickup', '', '拾う');
    mk('map', ICONS.map, 'マップ');
    mk('menu', ICONS.menu, 'メニュー');

    root.addEventListener('pointerdown', (e) => this.down(e));
    root.addEventListener('pointermove', (e) => this.move(e));
    root.addEventListener('pointerup', (e) => this.up(e));
    root.addEventListener('pointercancel', (e) => this.up(e));
    root.addEventListener('lostpointercapture', (e) => this.up(e));
    root.addEventListener('contextmenu', (e) => e.preventDefault());
    // iOS: block pinch-zoom / double-tap zoom gestures over the game
    for (const ev of ['gesturestart', 'gesturechange', 'dblclick']) root.addEventListener(ev, (e) => e.preventDefault());
  }

  show(v: boolean) {
    this.root.style.display = v ? 'block' : 'none';
    if (!v) this.releaseAll();
  }

  releaseAll() {
    this.joyId = -1;
    this.joyBase.classList.remove('active');
    this.looks.clear();
    this.held.clear();
    this.input.touchFire = false;
    this.input.touchUp = this.input.touchDown = false;
    this.input.moveX = 0;
    this.input.moveZ = 0;
    this.input.touchSprint = this.sprintLocked = false;
    this.lockIcon.classList.remove('on');
    for (const b of this.btns.values()) b.classList.remove('down');
  }

  /** Context-dependent appearance. */
  setState(s: { plane: boolean; air: boolean; canJumpPlane: boolean; pickup: string | null; ads: boolean; crouch: boolean; nades: number; heals: number; dead: boolean; driving?: boolean; flying?: boolean }) {
    this.flying = !!s.flying;
    const jump = this.btns.get('jump')!;
    jump.classList.toggle('big', s.plane || s.air);
    jump.dataset.label = s.plane ? (s.canJumpPlane ? '降下' : '待機') : s.air ? '開傘' : s.flying ? '上昇' : '';
    // at the controls of a helicopter: hold to climb, hold to descend, fire launches missiles
    jump.classList.toggle('fly', this.flying);
    const crouch = this.btns.get('crouch')!;
    crouch.classList.toggle('fly', this.flying);
    crouch.dataset.label = this.flying ? '下降' : '';
    jump.classList.toggle('disabled', s.plane && !s.canJumpPlane);
    const combat = !s.plane && !s.air && !s.dead && !s.driving && !s.flying;
    for (const id of ['ads', 'reload'] as TouchButton[]) this.btns.get(id)!.style.display = combat ? '' : 'none';
    for (const id of ['fire', 'fireL', 'crouch'] as TouchButton[]) this.btns.get(id)!.style.display = combat || s.flying ? '' : 'none';
    this.btns.get('nade')!.style.display = combat && s.nades > 0 ? '' : 'none';
    this.btns.get('heal')!.style.display = combat && s.heals > 0 ? '' : 'none';
    this.btns.get('jump')!.style.display = s.dead || s.driving ? 'none' : '';
    const pick = this.btns.get('pickup')!;
    pick.style.display = s.pickup && (combat || s.driving || s.flying) ? '' : 'none';
    const label = pick.querySelector('.tb-label');
    if (label && label.textContent !== (s.pickup ?? '')) label.textContent = s.pickup ?? '';
    this.btns.get('ads')!.classList.toggle('on', s.ads);
    this.btns.get('crouch')!.classList.toggle('on', s.crouch && !this.flying);
  }

  private press(b: TouchButton) {
    const i = this.input;
    switch (b) {
      case 'fire':
      case 'fireL':
        i.touchFire = true;
        break;
      case 'ads':
        i.adsToggled = !i.adsToggled;
        break;
      case 'jump':
        i.edges.jump = true;
        i.touchUp = true;
        break;
      case 'crouch':
        if (this.flying) i.touchDown = true;
        else i.crouch = !i.crouch;
        break;
      case 'reload':
        i.edges.reload = true;
        break;
      case 'nade':
        i.edges.nade = true;
        break;
      case 'pickup':
        i.edges.pickup = true;
        break;
      case 'heal':
        i.edges.heal = 4;
        break;
      case 'map':
        i.edges.map = true;
        break;
      case 'menu':
        i.edges.menu = true;
        break;
    }
    this.onButton(b);
  }

  private down(e: PointerEvent) {
    if (e.pointerType === 'mouse') return;
    e.preventDefault();
    const t = e.target as HTMLElement;
    const btnEl = t.closest('[data-btn]') as HTMLElement | null;
    try {
      this.root.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    if (btnEl) {
      const b = btnEl.dataset.btn as TouchButton;
      btnEl.classList.add('down');
      this.held.set(e.pointerId, b);
      this.press(b);
      if (b === 'fire' || b === 'fireL') this.looks.set(e.pointerId, [e.clientX, e.clientY]);
      return;
    }
    const w = window.innerWidth;
    if (e.clientX < w * 0.42 && this.joyId < 0) {
      this.joyId = e.pointerId;
      if (this.sprintLocked) {
        this.sprintLocked = false;
        this.input.touchSprint = false;
        this.lockIcon.classList.remove('on');
      }
      this.joyOrigin = [e.clientX, e.clientY];
      this.joyBase.style.left = `${e.clientX}px`;
      this.joyBase.style.top = `${e.clientY}px`;
      this.joyBase.classList.add('active');
      this.updateJoy(e.clientX, e.clientY);
      return;
    }
    this.looks.set(e.pointerId, [e.clientX, e.clientY]);
  }

  private radius() {
    return Math.min(70, Math.max(46, window.innerHeight * 0.11));
  }

  private updateJoy(x: number, y: number) {
    const R = this.radius();
    let dx = x - this.joyOrigin[0];
    let dy = y - this.joyOrigin[1];
    const d = Math.hypot(dx, dy);
    const lockZone = dy < -R * 2.1 && Math.abs(dx) < R * 1.2;
    if (d > R) {
      dx = (dx / d) * R;
      dy = (dy / d) * R;
    }
    this.joyKnob.style.transform = `translate(${dx}px, ${dy}px)`;
    const mx = dx / R;
    const mz = -dy / R;
    this.input.moveX = Math.abs(mx) < 0.08 ? 0 : mx;
    this.input.moveZ = Math.abs(mz) < 0.08 ? 0 : mz;
    this.input.touchSprint = d > R * 1.25 && mz > 0.7;
    this.lockIcon.classList.toggle('arm', lockZone);
  }

  private move(e: PointerEvent) {
    if (e.pointerType === 'mouse') return;
    if (e.pointerId === this.joyId) {
      this.updateJoy(e.clientX, e.clientY);
      return;
    }
    const prev = this.looks.get(e.pointerId);
    if (prev) {
      this.input.lookDX += (e.clientX - prev[0]) * this.lookScale;
      this.input.lookDY += (e.clientY - prev[1]) * this.lookScale;
      prev[0] = e.clientX;
      prev[1] = e.clientY;
    }
  }

  private up(e: PointerEvent) {
    if (e.pointerType === 'mouse') return;
    if (e.pointerId === this.joyId) {
      const R = this.radius();
      const dy = e.clientY - this.joyOrigin[1];
      const dx = e.clientX - this.joyOrigin[0];
      this.joyId = -1;
      this.joyBase.classList.remove('active');
      this.lockIcon.classList.remove('arm');
      if (e.type === 'pointerup' && dy < -R * 2.1 && Math.abs(dx) < R * 1.2) {
        // auto-run lock
        this.sprintLocked = true;
        this.input.moveX = 0;
        this.input.moveZ = 1;
        this.input.touchSprint = true;
        this.lockIcon.classList.add('on');
      } else {
        this.input.moveX = 0;
        this.input.moveZ = 0;
        this.input.touchSprint = false;
      }
      return;
    }
    this.looks.delete(e.pointerId);
    const b = this.held.get(e.pointerId);
    if (b) {
      this.held.delete(e.pointerId);
      this.btns.get(b)?.classList.remove('down');
      if (b === 'jump') this.input.touchUp = false;
      if (b === 'crouch') this.input.touchDown = false;
      if ((b === 'fire' || b === 'fireL') && ![...this.held.values()].some((v) => v === 'fire' || v === 'fireL')) this.input.touchFire = false;
    }
  }

  get locked() {
    return this.sprintLocked;
  }
}
