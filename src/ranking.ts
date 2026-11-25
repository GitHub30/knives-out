// Rankings on the end card: best survival time and number of wins, kept by
// the game server (a Cloudflare Durable Object). Online results are recorded
// by the server itself; solo (offline bot match) results are reported by this
// browser and live on their own, self-reported board.
//
// Everything here is best effort: with no network, no server, or a server
// that does not know the API yet, the end card just says so.

import { serverBase } from './net';

export type Board = 'online' | 'solo';

export interface RankTop {
  survival: { name: string; seconds: number; at: number }[];
  champions: { name: string; wins: number; games: number }[];
  /** Solo submissions: the name as the server stored it. */
  name?: string;
}

export interface OwnResult {
  board: Board;
  /** The player's in-match name. */
  name: string;
  /** Seconds survived (simulation time). */
  seconds: number;
  place: number;
  won: boolean;
  /** The whole match is over (as opposed to: just eliminated). */
  final: boolean;
}

const ROWS = 10;
const TIMEOUT_MS = 7000;

/** HTTP(S) base of the game server: the WebSocket base with the scheme swapped. */
export function rankBase(): string {
  return serverBase().replace(/^ws/i, 'http');
}

/**
 * Whether this page reports its solo results: not a debug page, and not a
 * local build that was not pointed at a server on purpose (development
 * and automated test runs must not write to the public board; they still
 * read it).
 */
export function reportsSolo(): boolean {
  const q = new URLSearchParams(location.search);
  if (q.has('debug')) return false;
  const local = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  return !local || q.has('server') || q.has('dev');
}

export function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function parseTop(v: unknown): RankTop {
  const o = (v && typeof v === 'object' ? v : {}) as Record<string, unknown>;
  const str = (x: unknown) => (typeof x === 'string' ? x.slice(0, 64) : '');
  const num = (x: unknown) => (typeof x === 'number' && Number.isFinite(x) ? x : 0);
  const list = (x: unknown) => (Array.isArray(x) ? (x.slice(0, 50) as Record<string, unknown>[]).filter((r) => r && typeof r === 'object') : []);
  if (!Array.isArray(o.survival) || !Array.isArray(o.champions)) throw new Error('bad ranking response');
  return {
    survival: list(o.survival).map((r) => ({ name: str(r.name), seconds: num(r.seconds), at: num(r.at) })),
    champions: list(o.champions).map((r) => ({ name: str(r.name), wins: num(r.wins), games: num(r.games) })),
    name: typeof o.name === 'string' ? o.name.slice(0, 64) : undefined,
  };
}

async function call(path: string, init?: RequestInit): Promise<RankTop> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(`${rankBase()}${path}`, { ...init, signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return parseTop(await r.json());
  } finally {
    clearTimeout(timer);
  }
}

export function fetchBoard(board: Board): Promise<RankTop> {
  return call(`/api/rank?board=${board}`);
}

/** Report a finished solo match; answers with the updated solo board. */
export function submitSolo(name: string, survived: number, place: number, won: boolean): Promise<RankTop> {
  return call('/api/rank/solo', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, survived: Math.round(survived * 10) / 10, place, won }),
  });
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** The ranking block of the end card. Builds its own DOM inside `card`. */
export class RankPanel {
  private root: HTMLElement;
  private label: HTMLElement;
  private status: HTMLElement;
  private lists: HTMLElement;
  private survival: HTMLOListElement;
  private champions: HTMLOListElement;
  /** Bumped whenever the card changes: answers to older requests are dropped. */
  private gen = 0;
  /** One solo submission per match. */
  private submitted = false;
  private delay: ReturnType<typeof setTimeout> | null = null;

  constructor(card: HTMLElement, before: Element | null) {
    this.root = el('div', 'rank');
    this.label = el('p', 'rank-label');
    this.status = el('p', 'rank-status');
    this.lists = el('div', 'rank-lists');
    const col = (title: string) => {
      const box = el('div', 'rank-col');
      box.append(el('h3', '', title));
      const ol = el('ol');
      box.append(ol);
      this.lists.append(box);
      return ol;
    };
    this.survival = col('生存時間ランキング');
    this.champions = col('チャンピオンランキング');
    this.root.append(this.label, this.status, this.lists);
    card.insertBefore(this.root, before);
  }

  /** A new match begins: forget the previous one. */
  reset() {
    this.gen++;
    this.submitted = false;
    if (this.delay) clearTimeout(this.delay);
    this.delay = null;
  }

  private fill(ol: HTMLOListElement, rows: { name: string; value: string; note?: string }[], own: string) {
    ol.replaceChildren();
    if (!rows.length) {
      ol.append(el('li', 'rank-empty', 'まだ記録がありません'));
      return;
    }
    rows.slice(0, ROWS).forEach((r, i) => {
      const li = el('li', r.name === own ? 'me' : '');
      li.append(el('span', 'rank-no', String(i + 1)), el('span', 'rank-name', r.name), el('span', 'rank-val', r.value));
      if (r.note) li.title = r.note;
      ol.append(li);
    });
  }

  private render(top: RankTop, own: string) {
    this.fill(
      this.survival,
      top.survival.map((r) => ({ name: r.name, value: fmtTime(r.seconds) })),
      own,
    );
    this.fill(
      this.champions,
      top.champions.map((r) => ({ name: r.name, value: `${r.wins}勝`, note: `${r.games}試合` })),
      own,
    );
    this.status.hidden = true;
    this.lists.hidden = false;
  }

  /** Show the board for the match that just ended (or the player's elimination). */
  show(r: OwnResult) {
    const gen = ++this.gen;
    if (this.delay) clearTimeout(this.delay);
    this.delay = null;
    this.label.textContent = r.board === 'online' ? 'オンライン対戦のランキング' : 'ソロ（自己申告）のランキング';
    // keep lists already on screen (eliminated -> match over) while refreshing
    if (this.lists.hidden || !this.survival.childElementCount) {
      this.lists.hidden = true;
      this.status.hidden = false;
      this.status.textContent = '読み込み中…';
    }
    let own = r.name;
    const run = async () => {
      let top: RankTop;
      if (r.board === 'solo' && !this.submitted && reportsSolo()) {
        this.submitted = true;
        try {
          top = await submitSolo(r.name, r.seconds, r.place, r.won);
          if (top.name) own = top.name;
        } catch {
          // not recorded (offline, rate limit, old server): still try to show the board
          top = await fetchBoard('solo');
        }
      } else {
        top = await fetchBoard(r.board);
      }
      if (gen === this.gen) this.render(top, own);
    };
    const start = () => {
      this.delay = null;
      run().catch(() => {
        if (gen !== this.gen) return;
        this.lists.hidden = true;
        this.status.hidden = false;
        this.status.textContent = 'ランキングを取得できません';
      });
    };
    // the room records an online result in the same tick it announces it:
    // give that a moment to land before reading the board
    if (r.board === 'online') this.delay = setTimeout(start, 1500);
    else start();
  }
}
