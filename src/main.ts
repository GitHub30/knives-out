import './style.css';
import { Core, coreModule } from './core';
import { Renderer, QUALITY_PRESETS, type Quality } from './render/renderer';
import carModelUrl from './assets/police_crown_osaka_game.glb?url';
import planeModelUrl from './assets/b777_300er.glb?url';
import planeFanUrl from './assets/b777_300er_fan.glb?url';
import g1Url from './assets/unitree_g1.bin?url';
import { setG1Shapes } from './render/models';
import { add, dirFrom, invert, lookAt, mul, perspectiveRZ, type V3 } from './render/math';
import { Instances } from './render/instances';
import { drawPlane } from './render/models';
import { Input } from './input';
import { TouchControls } from './touch';
import { Hud } from './hud';
import { Sound } from './audio';
import { Game, type MatchResult, type Settings } from './game';
import { serverBase, WorkerTransport, WsTransport, type Transport } from './net';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const DEFAULTS: Settings = {
  sens: 1,
  adsSens: 0.8,
  touchSens: 1.6,
  volume: 0.7,
  quality: 'auto',
  autoFire: true,
  showFps: false,
  invertY: false,
  adsToggle: false,
  bots: 29,
  touchUi: 'auto',
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem('zonefall.settings');
    if (raw) {
      const v = JSON.parse(raw);
      const s = { ...DEFAULTS };
      for (const k of Object.keys(DEFAULTS) as (keyof Settings)[]) {
        if (typeof v[k] === typeof DEFAULTS[k]) (s as Record<string, unknown>)[k] = v[k];
      }
      s.bots = Math.max(5, Math.min(31, Math.round(s.bots)));
      return s;
    }
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS };
}
function saveSettings(s: Settings) {
  try {
    localStorage.setItem('zonefall.settings', JSON.stringify(s));
  } catch {
    /* ignore */
  }
}
function loadName(): string {
  try {
    return localStorage.getItem('zonefall.name') ?? '';
  } catch {
    return '';
  }
}
function saveName(n: string) {
  try {
    localStorage.setItem('zonefall.name', n);
  } catch {
    /* ignore */
  }
}

// Primary pointer decides: touchscreen laptops with a mouse keep desktop controls.
const isTouch = () => matchMedia('(pointer: coarse)').matches;

function pickQuality(s: Settings): Quality {
  if (s.quality !== 'auto') return QUALITY_PRESETS[s.quality];
  if (isTouch()) {
    const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4;
    return mem <= 3 ? QUALITY_PRESETS.low : QUALITY_PRESETS.medium;
  }
  return QUALITY_PRESETS.high;
}

const settings = loadSettings();
let renderer: Renderer;
let core: Core;
let input: Input;
let touch: TouchControls;
let hud: Hud;
const sound = new Sound();
sound.volume = settings.volume;
let game: Game | null = null;
let lastMode: { kind: 'offline' | 'online'; room?: string } = { kind: 'offline' };
let backdropRaf = 0;

function show(id: string, v: boolean) {
  $(id).classList.toggle('hidden', !v);
}

function setScreen(screen: 'menu' | 'game' | 'loading') {
  show('menu', screen === 'menu');
  show('loading', screen === 'loading');
  hud.show(screen === 'game');
  touch.show(screen === 'game' && useTouchUi());
  document.body.classList.toggle('in-game', screen === 'game');
}

function useTouchUi() {
  return settings.touchUi === 'on' || (settings.touchUi === 'auto' && isTouch());
}

function errorBox(msg: string) {
  $('error-msg').textContent = msg;
  show('error', true);
}

// ------------------------------------------------------------------ backdrop
const backdropInst = { boxes: new Instances(64), glow: new Instances(4), smoke: new Instances(4) };
function startBackdrop() {
  cancelAnimationFrame(backdropRaf);
  const t0 = performance.now();
  const loop = (now: number) => {
    backdropRaf = requestAnimationFrame(loop);
    if (game) return;
    const t = (now - t0) / 1000;
    const a = t * 0.05;
    const c: V3 = [512, 0, 512];
    const eye: V3 = [c[0] + Math.cos(a) * 520, 150, c[2] + Math.sin(a) * 520];
    const dir = dirFrom(Math.atan2(c[0] - eye[0], c[2] - eye[2]), -0.3);
    let vp = mul(perspectiveRZ(1.0, renderer.aspect, 0.5), lookAt(eye, add(eye, dir), [0, 1, 0]));
    // debug: a fixed camera [x, y, z, yaw, pitch] for looking at the world up close
    const dbg = (window as unknown as { __zfCam?: number[] }).__zfCam;
    if (dbg) {
      eye[0] = dbg[0];
      eye[1] = dbg[1];
      eye[2] = dbg[2];
      vp = mul(perspectiveRZ(1.0, renderer.aspect, 0.08), lookAt(eye, add(eye, dirFrom(dbg[3], dbg[4])), [0, 1, 0]));
    }
    backdropInst.boxes.reset();
    const pa = t * 0.12;
    // circling left, banked into the turn
    const dbgPlane = (window as unknown as { __zfPlane?: number[] }).__zfPlane;
    if (dbgPlane) drawPlane(renderer, backdropInst.boxes, [dbgPlane[0], dbgPlane[1], dbgPlane[2]], dbgPlane[3], t, dbgPlane[4] ?? 0);
    else drawPlane(renderer, backdropInst.boxes, [512 + Math.cos(pa) * 300, 260, 512 + Math.sin(pa) * 300], pa + Math.PI, t, -0.2);
    renderer.render({
      viewProj: vp,
      invViewProj: invert(vp),
      camPos: eye,
      time: t,
      shadowCenter: dbg ? eye : [512 + Math.cos(pa) * 300, 230, 512 + Math.sin(pa) * 300],
      zone: { cx: 512, cz: 512, r: 1, visible: false },
      hurt: 0,
      focus: null,
      ...backdropInst,
    });
  };
  backdropRaf = requestAnimationFrame(loop);
}

async function loadBackdropWorld() {
  const seed = (Math.random() * 1e9) >>> 0;
  core.init(seed);
  const h = new Float32Array(257 * 257);
  for (let j = 0; j < 257; j++) for (let i = 0; i < 257; i++) h[j * 257 + i] = core.height(i * 4, j * 4);
  renderer.uploadWorld({
    terrainVerts: core.bytes(0),
    terrainIdx: core.bytes(1),
    terrainChunks: core.f32(2),
    propVerts: core.bytes(3),
    propChunks: core.f32(4),
    heights: h,
    mapSize: 1024,
    waterLevel: 0,
    cars: core.f32(8),
  });
  core.freeMeshes();
}

// ------------------------------------------------------------------ game flow
function cleanName(): string {
  const raw = ($('name') as HTMLInputElement).value.replace(/[\u0000-\u001f<>&"']/g, '').trim().slice(0, 14);
  const name = raw || `Player${Math.floor(Math.random() * 900 + 100)}`;
  saveName(name);
  return name;
}

async function enterFullscreen() {
  if (!isTouch()) return;
  try {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
    const o = screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> };
    await o.lock?.('landscape').catch(() => {});
  } catch {
    /* not supported (e.g. iPhone Safari) */
  }
}

async function startGame(kind: 'offline' | 'online', room?: string) {
  lastMode = { kind, room };
  // Tear down any previous match first so a failed connect never leaves it running.
  game?.destroy();
  game = null;
  show('pause', false);
  show('result', false);
  show('lobby', false);
  hud.openMap(false);
  sound.unlock();
  sound.ui('click');
  const name = cleanName();
  void enterFullscreen();
  setScreen('loading');
  $('loading-msg').textContent = kind === 'offline' ? 'ワールドを生成中…' : 'サーバーに接続中…';
  let transport: Transport;
  try {
    const module = await coreModule();
    if (kind === 'offline') {
      transport = new WorkerTransport(module, (Math.random() * 2 ** 32) >>> 0, settings.bots + 1, name);
    } else {
      const q = new URLSearchParams({ name });
      if (room) q.set('room', room);
      const ws = new WsTransport(`${serverBase()}/ws?${q}`);
      await ws.opened();
      transport = ws;
    }
  } catch (e) {
    setScreen('menu');
    errorBox(`接続に失敗しました: ${(e as Error).message}`);
    return;
  }
  input.adsToggleMode = settings.adsToggle;
  touch.lookScale = settings.touchSens;
  const g = new Game({
    core,
    renderer,
    hud,
    input,
    touch,
    sound,
    settings,
    transport,
    onResult: (r) => showResult(r),
    onLobby: (info) => {
      show('lobby', !!info);
      if (info) {
        $('lobby-count').textContent = `${info.count} / ${info.slots}`;
        $('lobby-time').textContent = info.count > 0 ? `${Math.max(0, Math.ceil(info.timeLeft))} 秒後に開始（不足分はボットが参加）` : '';
        $('lobby-room').textContent = room ? `ルーム: ${room}` : '公開マッチ';
        show('lobby-start', info.isPrivate);
      }
    },
    onPause: (p) => {
      show('pause', p);
      $('pause-title').textContent = transport.kind === 'offline' ? '一時停止中' : 'メニュー（試合は続行中）';
    },
    onError: (msg) => {
      if (game !== g) return;
      quitToMenu();
      errorBox(msg);
    },
  });
  game = g;
  try {
    await g.start();
  } catch (e) {
    if (game === g) {
      g.destroy();
      game = null;
      setScreen('menu');
      errorBox((e as Error).message);
    }
    return;
  }
  setScreen('game');
}

function showResult(r: MatchResult) {
  const won = r.won;
  $('result-title').textContent = won ? '🏆 チャンピオン！' : `#${r.place} / ${r.total}`;
  $('result-sub').textContent = won
    ? '最後の生存者になった！'
    : r.final && r.winnerName
      ? `試合終了 — 勝者: ${r.winnerName}`
      : r.killerName
        ? `${r.killerName} に倒された`
        : '安全地帯の外で倒れた';
  $('result-kills').textContent = String(r.kills);
  $('result-dmg').textContent = String(r.damage);
  $('result-time').textContent = `${Math.floor(r.time / 60)}:${String(Math.floor(r.time % 60)).padStart(2, '0')}`;
  $('result').classList.toggle('won', won);
  show('result', true);
  input.unlock();
  touch.releaseAll();
  show('result-spect', !won && !r.final && !!game?.latest && game.latest.phase === 1);
}

function quitToMenu() {
  game?.destroy();
  game = null;
  show('result', false);
  show('pause', false);
  show('lobby', false);
  hud.openMap(false);
  setScreen('menu');
  if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
  void loadBackdropWorld().then(startBackdrop);
}

// ------------------------------------------------------------------ settings UI
function bindSettings() {
  const bindRange = (id: string, key: 'sens' | 'adsSens' | 'touchSens' | 'volume' | 'bots', fmt: (v: number) => string) => {
    const el = $(id) as HTMLInputElement;
    const out = $(`${id}-v`);
    el.value = String(settings[key]);
    out.textContent = fmt(settings[key]);
    el.addEventListener('input', () => {
      settings[key] = Number(el.value);
      out.textContent = fmt(settings[key]);
      if (key === 'volume') sound.setVolume(settings.volume);
      if (key === 'touchSens') touch.lookScale = settings.touchSens;
      saveSettings(settings);
    });
  };
  bindRange('s-sens', 'sens', (v) => v.toFixed(2));
  bindRange('s-ads', 'adsSens', (v) => v.toFixed(2));
  bindRange('s-touch', 'touchSens', (v) => v.toFixed(2));
  bindRange('s-vol', 'volume', (v) => `${Math.round(v * 100)}%`);
  bindRange('s-bots', 'bots', (v) => `${v + 1} 人`);
  const bindCheck = (id: string, key: 'autoFire' | 'showFps' | 'invertY' | 'adsToggle') => {
    const el = $(id) as HTMLInputElement;
    el.checked = settings[key];
    el.addEventListener('change', () => {
      settings[key] = el.checked;
      if (key === 'adsToggle') input.adsToggleMode = el.checked;
      saveSettings(settings);
    });
  };
  bindCheck('s-autofire', 'autoFire');
  bindCheck('s-fps', 'showFps');
  bindCheck('s-invert', 'invertY');
  bindCheck('s-adstoggle', 'adsToggle');
  const q = $('s-quality') as HTMLSelectElement;
  q.value = settings.quality;
  q.addEventListener('change', () => {
    settings.quality = q.value as Settings['quality'];
    saveSettings(settings);
    renderer.setQuality(pickQuality(settings));
  });
  const tu = $('s-touchui') as HTMLSelectElement;
  tu.value = settings.touchUi;
  tu.addEventListener('change', () => {
    settings.touchUi = tu.value as Settings['touchUi'];
    saveSettings(settings);
    if (game) touch.show(useTouchUi());
  });
}

// ------------------------------------------------------------------ boot
async function boot() {
  const canvas = $('view') as HTMLCanvasElement;
  const ui = $('ui');
  ($('name') as HTMLInputElement).value = loadName();
  if (!('gpu' in navigator)) {
    show('nogpu', true);
    show('menu', false);
    return;
  }
  try {
    renderer = await Renderer.create(canvas, pickQuality(settings), new URLSearchParams(location.search).has('offscreen'));
    // characters are drawn as Unitree G1 robots once the parts have loaded
    fetch(new URL(g1Url, document.baseURI).href)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const dv = new DataView(buf);
        if (dv.getUint32(0, true) !== 0x31503147) throw new Error('bad G1 parts file');
        const n = dv.getUint32(4, true);
        let off = 8 + n * 4;
        const meshes: Float32Array[] = [];
        for (let k = 0; k < n; k++) {
          const verts = dv.getUint32(8 + k * 4, true);
          meshes.push(new Float32Array(buf.slice(off, off + verts * 24)));
          off += verts * 24;
        }
        setG1Shapes(renderer.addShapes(meshes));
      })
      .catch((e) => console.warn('G1 model:', e));
    // the parked cars appear once their model has loaded
    renderer.loadCarModel(new URL(carModelUrl, document.baseURI).href).catch((e) => console.warn('car model:', e));
    renderer.loadPlaneModel(new URL(planeModelUrl, document.baseURI).href, new URL(planeFanUrl, document.baseURI).href).catch((e) => console.warn('plane model:', e));
  } catch (e) {
    show('nogpu', true);
    show('menu', false);
    $('nogpu-detail').textContent = String((e as Error).message ?? e);
    return;
  }
  input = new Input(canvas);
  touch = new TouchControls(ui, input);
  touch.lookScale = settings.touchSens;
  hud = new Hud(ui);
  hud.onSlot = (s) => (input.edges.slot = s);
  hud.onHeal = (k) => (input.edges.heal = k);
  hud.onSpectNext = () => (input.edges.spect = true);
  hud.onLeave = () => quitToMenu();
  hud.onMapClose = () => (input.edges.map = true);
  touch.onButton = () => sound.unlock();
  window.addEventListener('resize', () => renderer.resize());
  if (new URLSearchParams(location.search).has('debug')) {
    // Test hook: returns the next rendered frame as a PNG data URL.
    let frames = 0;
    const countFrames = () => {
      frames++;
      requestAnimationFrame(countFrames);
    };
    requestAnimationFrame(countFrames);
    (window as unknown as { __zfFrames: () => number }).__zfFrames = () => frames;
    // Test hooks for the end-to-end prediction / input harnesses.
    Object.defineProperty(window, '__game', { get: () => game });
    Object.defineProperty(window, '__core', { get: () => core });
    Object.defineProperty(window, '__input', { get: () => input });
    Object.defineProperty(window, '__hud', { get: () => hud });
    Object.defineProperty(window, '__settings', { get: () => settings });
    (window as unknown as { __zfCapture: () => Promise<string> }).__zfCapture = async () => {
      const img = await renderer.capture();
      const c = document.createElement('canvas');
      c.width = img.w;
      c.height = img.h;
      c.getContext('2d')!.putImageData(new ImageData(Uint8ClampedArray.from(img.rgba), img.w, img.h), 0, 0);
      return c.toDataURL('image/png');
    };
  }
  bindSettings();
  setScreen('loading');
  $('loading-msg').textContent = 'エンジンを初期化中…';
  try {
    core = await Core.create();
    await loadBackdropWorld();
  } catch (e) {
    errorBox(`初期化に失敗しました: ${(e as Error).message}`);
    return;
  }
  setScreen('menu');
  startBackdrop();
  renderer.device.lost.then((info) => {
    if (info.reason !== 'destroyed') errorBox('GPU デバイスが失われました。ページを再読み込みしてください。');
  });

  $('btn-solo').addEventListener('click', () => void startGame('offline'));
  $('btn-online').addEventListener('click', () => void startGame('online'));
  $('btn-room').addEventListener('click', () => {
    const code = ($('room-code') as HTMLInputElement).value.trim().toUpperCase();
    if (!/^[A-Z0-9]{6,12}$/.test(code)) {
      errorBox('ルームコードは英数字 6〜12 文字で入力してください');
      return;
    }
    void startGame('online', code);
  });
  $('btn-room-new').addEventListener('click', () => {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let c = '';
    const rnd = crypto.getRandomValues(new Uint32Array(6));
    for (const v of rnd) c += chars[v % chars.length];
    ($('room-code') as HTMLInputElement).value = c;
  });
  $('btn-settings').addEventListener('click', () => show('settings', true));
  $('btn-help').addEventListener('click', () => show('help', true));
  for (const b of document.querySelectorAll<HTMLElement>('[data-close]')) b.addEventListener('click', () => show(b.dataset.close!, false));
  $('btn-resume').addEventListener('click', () => {
    if (game?.isPaused) game.setPaused(false);
    show('pause', false);
    input.lock();
  });
  $('btn-pause-settings').addEventListener('click', () => show('settings', true));
  $('btn-quit').addEventListener('click', () => quitToMenu());
  $('btn-again').addEventListener('click', () => {
    show('result', false);
    void startGame(lastMode.kind, lastMode.room);
  });
  $('btn-menu').addEventListener('click', () => quitToMenu());
  $('result-spect').addEventListener('click', () => {
    show('result', false);
    hud.setSpectate('');
  });
  $('lobby-start').addEventListener('click', () => {
    game?.requestStart();
  });
  $('lobby-leave').addEventListener('click', () => quitToMenu());
  $('error-ok').addEventListener('click', () => show('error', false));
  document.addEventListener('pointerdown', () => sound.unlock(), { once: false, passive: true });
}

void boot();
