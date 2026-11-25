// ZONEFALL rankings: one global Durable Object (SQLite) that keeps, per board
// and player name, the best survival time and the number of wins.
//
// Two boards that never mix:
// * "online": written only by Room objects (the authoritative simulation).
// * "solo":   self-reported by browsers after an offline bot match, so it is
//             forgeable by design and is shown as such.
//
// Only a name, numbers and timestamps are stored: no addresses, no ids.

export type Board = 'online' | 'solo';

export interface RankEntry {
  name: string;
  /** Seconds alive in the play phase. */
  seconds: number;
  /** Final placement (0 = unknown: the player left while alive, which
   * counts for the survival time but not as a game played). */
  place: number;
  won: boolean;
}

export interface RankTop {
  survival: { name: string; seconds: number; at: number }[];
  champions: { name: string; wins: number; games: number }[];
}

/** Longest anyone can stay alive, in seconds: the zone schedule (ZONES in
 * the core) closes to nothing after 485 s and then takes 14 health a
 * second, so even with every heal nobody gets far past that. */
export const MAX_SURVIVAL_S = 560;
/** A win in less than this is not a real one: bots are still parachuting. */
export const MIN_WIN_S = 60;
/** Most players a match can hold (MAX_PLAYERS in the core). */
export const MAX_PLACE = 32;
const TOP = 20;
/** Names kept per board; pruned back to this once PRUNE_SLACK more arrive. */
const MAX_NAMES = 2000;
const PRUNE_SLACK = 100;
const MAX_NAME_UNITS = 64;
const MAX_BATCH = 40;

export function isBoard(v: unknown): v is Board {
  return v === 'online' || v === 'solo';
}

export class Ranking {
  private sql: SqlStorage;
  private cache = new Map<Board, string>();
  private writes = new Map<Board, number>();

  constructor(state: DurableObjectState, _env: unknown) {
    this.sql = state.storage.sql;
    this.sql.exec(
      `CREATE TABLE IF NOT EXISTS rank (
         board   TEXT NOT NULL,
         name    TEXT NOT NULL,
         best    REAL NOT NULL DEFAULT 0,
         best_at INTEGER NOT NULL DEFAULT 0,
         wins    INTEGER NOT NULL DEFAULT 0,
         games   INTEGER NOT NULL DEFAULT 0,
         seen    INTEGER NOT NULL DEFAULT 0,
         PRIMARY KEY (board, name)
       ) WITHOUT ROWID`,
    );
    this.sql.exec('CREATE INDEX IF NOT EXISTS rank_best ON rank (board, best DESC, best_at ASC)');
    this.sql.exec('CREATE INDEX IF NOT EXISTS rank_wins ON rank (board, wins DESC, best DESC)');
  }

  private top(board: Board): string {
    const hit = this.cache.get(board);
    if (hit) return hit;
    const survival = this.sql
      .exec<{ name: string; best: number; best_at: number }>('SELECT name, best, best_at FROM rank WHERE board = ? AND best > 0 ORDER BY best DESC, best_at ASC, name ASC LIMIT ?', board, TOP)
      .toArray()
      .map((r) => ({ name: r.name, seconds: r.best, at: r.best_at }));
    const champions = this.sql
      .exec<{ name: string; wins: number; games: number }>('SELECT name, wins, games FROM rank WHERE board = ? AND wins > 0 ORDER BY wins DESC, best DESC, name ASC LIMIT ?', board, TOP)
      .toArray()
      .map((r) => ({ name: r.name, wins: r.wins, games: r.games }));
    const out = JSON.stringify({ survival, champions } satisfies RankTop);
    this.cache.set(board, out);
    return out;
  }

  private record(board: Board, e: RankEntry, now: number) {
    const name = typeof e.name === 'string' ? e.name : '';
    if (!name || name.length > MAX_NAME_UNITS) return;
    const s = Number(e.seconds);
    if (!Number.isFinite(s) || s < 0) return;
    const seconds = Math.round(Math.min(s, MAX_SURVIVAL_S) * 10) / 10;
    const place = Number(e.place);
    // a result without a placement (left while alive) is no game played:
    // joining and leaving under somebody's name must not count against them
    const game = Number.isInteger(place) && place >= 1 ? 1 : 0;
    const won = e.won === true && place === 1 && seconds >= MIN_WIN_S ? 1 : 0;
    this.sql.exec(
      `INSERT INTO rank (board, name, best, best_at, wins, games, seen) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?4)
       ON CONFLICT (board, name) DO UPDATE SET
         best_at = CASE WHEN excluded.best > best THEN excluded.best_at ELSE best_at END,
         best    = MAX(best, excluded.best),
         wins    = wins + excluded.wins,
         games   = games + excluded.games,
         seen    = excluded.seen`,
      board,
      name,
      seconds,
      now,
      won,
      game,
    );
  }

  /** Keep the board bounded: rows that are in neither top half go, stalest first. */
  private prune(board: Board) {
    const n = (this.writes.get(board) ?? 0) + 1;
    this.writes.set(board, n);
    // counting is a scan: only look now and then (and on the first write after a restart)
    if (n !== 1 && n % 20 !== 0) return;
    const count = this.sql.exec<{ c: number }>('SELECT COUNT(*) AS c FROM rank WHERE board = ?', board).one().c;
    if (count <= MAX_NAMES + PRUNE_SLACK) return;
    this.sql.exec(
      `DELETE FROM rank WHERE board = ?1 AND name IN (
         SELECT name FROM rank WHERE board = ?1
           AND name NOT IN (SELECT name FROM rank WHERE board = ?1 ORDER BY best DESC, best_at ASC LIMIT ?2)
           AND name NOT IN (SELECT name FROM rank WHERE board = ?1 AND wins > 0 ORDER BY wins DESC, best DESC LIMIT ?2)
         ORDER BY seen ASC LIMIT ?3)`,
      board,
      MAX_NAMES / 2,
      count - MAX_NAMES,
    );
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const json = (body: string, status = 200) => new Response(body, { status, headers: { 'content-type': 'application/json' } });
    const board = url.searchParams.get('board');
    if (!isBoard(board)) return json('{"error":"board"}', 400);
    if (url.pathname === '/top' && req.method === 'GET') return json(this.top(board));
    if (url.pathname === '/report' && req.method === 'POST') {
      let body: { entries?: RankEntry[] };
      try {
        body = (await req.json()) as typeof body;
      } catch {
        return json('{"error":"body"}', 400);
      }
      const list = Array.isArray(body.entries) ? body.entries.slice(0, MAX_BATCH) : [];
      const now = Date.now();
      for (const e of list) if (e && typeof e === 'object') this.record(board, e, now);
      if (list.length) {
        this.cache.delete(board);
        this.prune(board);
      }
      return json(this.top(board));
    }
    return json('{"error":"not found"}', 404);
  }
}
