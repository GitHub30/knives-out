// Keyboard + mouse input. Touch controls (touch.ts) write into the same state.

export interface Edges {
  jump: boolean;
  reload: boolean;
  pickup: boolean;
  spect: boolean;
  nade: boolean;
  heal: number; // 0 none, 1 bandage, 2 medkit, 3 drink, 4 = best available
  slot: number; // -1 none, 0/1 weapon slot, 2 fists, 9 = cycle next, 8 = cycle prev
  map: boolean;
  menu: boolean;
}

const emptyEdges = (): Edges => ({ jump: false, reload: false, pickup: false, spect: false, nade: false, heal: 0, slot: -1, map: false, menu: false });

export class Input {
  // continuous state
  keys = new Set<string>();
  moveX = 0; // touch joystick (-1..1, +x = right)
  moveZ = 0; // +z = forward
  touchActive = false;
  touchSprint = false;
  lookDX = 0;
  lookDY = 0;
  mouseFire = false;
  touchFire = false;
  mouseAds = false;
  adsToggled = false;
  crouch = false;
  edges = emptyEdges();
  pointerLocked = false;
  enabled = false;
  adsToggleMode = false;
  private expectUnlock = false;

  private canvas: HTMLElement;

  constructor(canvas: HTMLElement) {
    this.canvas = canvas;
    window.addEventListener('keydown', (e) => this.onKey(e, true));
    window.addEventListener('keyup', (e) => this.onKey(e, false));
    window.addEventListener('blur', () => {
      this.keys.clear();
      this.mouseFire = false;
      this.mouseAds = false;
    });
    document.addEventListener('mousedown', (e) => {
      if (!this.enabled || e.button > 2) return;
      const t = e.target as HTMLElement | null;
      // only presses on the play area (canvas or the HUD/touch layers), not on buttons/menus
      if (t && t !== canvas && t.id !== 'touch' && t.id !== 'hud' && t.id !== 'ui') return;
      if (!this.pointerLocked) {
        this.lock();
        return;
      }
      if (e.button === 0) this.mouseFire = true;
      if (e.button === 2) {
        if (this.adsToggleMode) this.adsToggled = !this.adsToggled;
        else this.mouseAds = true;
      }
    });
    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.mouseFire = false;
      if (e.button === 2) this.mouseAds = false;
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('mousemove', (e) => {
      if (!this.enabled || !this.pointerLocked) return;
      // Some browsers report absurd spikes right after locking.
      if (Math.abs(e.movementX) > 400 || Math.abs(e.movementY) > 400) return;
      this.lookDX += e.movementX;
      this.lookDY += e.movementY;
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        if (!this.enabled) return;
        e.preventDefault();
        this.edges.slot = e.deltaY > 0 ? 9 : 8;
      },
      { passive: false },
    );
    document.addEventListener('pointerlockchange', () => {
      const was = this.pointerLocked;
      this.pointerLocked = document.pointerLockElement === this.canvas;
      if (!this.pointerLocked) {
        this.mouseFire = false;
        this.mouseAds = false;
        // Browsers swallow Esc to release pointer lock: treat that as "menu".
        if (was && this.enabled && !this.expectUnlock) this.edges.menu = true;
      }
      this.expectUnlock = false;
    });
  }

  lock() {
    if (this.pointerLocked || matchMedia('(pointer: coarse)').matches) return;
    try {
      const r = this.canvas.requestPointerLock() as unknown;
      if (r instanceof Promise) r.catch(() => {});
    } catch {
      /* ignore */
    }
  }

  unlock() {
    if (document.pointerLockElement) {
      this.expectUnlock = true;
      document.exitPointerLock();
    }
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    const c = e.code;
    if (!this.enabled) {
      this.keys.delete(c);
      return;
    }
    if (down) {
      if (['Space', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(c)) e.preventDefault();
      if (e.repeat) return;
      this.keys.add(c);
      switch (c) {
        case 'Space':
          this.edges.jump = true;
          break;
        case 'KeyR':
          this.edges.reload = true;
          break;
        case 'KeyF':
        case 'KeyE':
          this.edges.pickup = true;
          break;
        case 'KeyG':
          this.edges.nade = true;
          break;
        case 'KeyC':
          this.crouch = !this.crouch;
          break;
        case 'Digit1':
          this.edges.slot = 0;
          break;
        case 'Digit2':
          this.edges.slot = 1;
          break;
        case 'Digit3':
        case 'KeyX':
          this.edges.slot = 2;
          break;
        case 'Digit4':
          this.edges.heal = 1;
          break;
        case 'Digit5':
          this.edges.heal = 2;
          break;
        case 'Digit6':
          this.edges.heal = 3;
          break;
        case 'KeyH':
          this.edges.heal = 4;
          break;
        case 'KeyM':
        case 'Tab':
          this.edges.map = true;
          break;
        case 'KeyN':
          this.edges.spect = true;
          break;
        case 'Escape':
        case 'KeyP':
          this.edges.menu = true;
          break;
      }
    } else {
      this.keys.delete(c);
    }
  }

  /** Movement vector from keys + joystick. */
  move(): [number, number] {
    let x = this.moveX;
    let z = this.moveZ;
    const k = this.keys;
    if (k.has('KeyW') || k.has('ArrowUp')) z += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) z -= 1;
    if (k.has('KeyD') || k.has('ArrowRight')) x += 1;
    if (k.has('KeyA') || k.has('ArrowLeft')) x -= 1;
    const l = Math.hypot(x, z);
    if (l > 1) {
      x /= l;
      z /= l;
    }
    return [x, z];
  }

  get fire() {
    return this.mouseFire || this.touchFire;
  }
  get ads() {
    return this.mouseAds || this.adsToggled;
  }
  get sprint() {
    return this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') || this.touchSprint;
  }

  takeEdges(): Edges {
    const e = this.edges;
    this.edges = emptyEdges();
    return e;
  }

  takeLook(): [number, number] {
    const r: [number, number] = [this.lookDX, this.lookDY];
    this.lookDX = 0;
    this.lookDY = 0;
    return r;
  }

  reset() {
    this.keys.clear();
    this.moveX = this.moveZ = 0;
    this.lookDX = this.lookDY = 0;
    this.mouseFire = this.touchFire = this.mouseAds = this.adsToggled = false;
    this.crouch = false;
    this.touchSprint = false;
    this.edges = emptyEdges();
  }
}
