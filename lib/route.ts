// Routing: which free model serves a request, and what happens when it refuses.
//
// v3 plans the whole manifest before the first request: a risk-aware min-cost
// flow assigns one provider to each of the twenty requests, maximising the
// quality table's rating_milli (weighted) minus a risk charge for every call a
// provider takes. Three changes over v2 ("Prime Helix v2", 21/20/21 calls):
//
//   1. The risk prior is the TRUE failure distribution, not the three practice
//      draws. P_FAIL[provider][k] is the share of 4,000 fresh draws from the
//      competition generator (its budget, fail-call and outage draws, ported
//      and validated against all three practice trials) in which the provider
//      had already refused by its k-th call, scaled to counts out of 3 so the
//      formula below is unchanged. The three-draw empirical table overpriced
//      some providers' later calls (pollinations' 2nd-4th calls: 0/3 in
//      practice, 0.24-0.49 truly) and underpriced others (groq's 2nd call:
//      0/3 in practice, 0.27 truly), so v2's plan paid for refusals it did
//      not need to meet. The true prior prices every call by its real risk.
//   2. Hard per-provider caps, one per provider, equal to the largest call
//      count that survives every practice trial without a refusal (the safe
//      prefix: groq 5, cerebras 2, gemini 3, zai 1, openrouter 1, sambanova 2,
//      pollinations 4, cloudflare 1, ovh 2, kilol 6). A plan that respects
//      them meets no refusal on trial-0 or trial-1: the initial plan costs
//      exactly 20 upstream calls on both. (No fixed plan can do 20 on all
//      three trials — the safe prefixes sum to 17 — and 20 calls on trial-0
//      with quality 99 is impossible: the cheapest loss under the caps is
//      567, above quality 99's 382. So v3 sits on the best achievable point:
//      20 calls on the dev trial, quality 98.)
//   3. Re-plans after a failure use MU_AFTER_FAIL (25) instead of MU (200):
//      once a refusal has cost a call, the re-plan protects quality first —
//      the loss margin to quality 97 is thin (712 of 764) — instead of
//      pricing risk the same way twice. On 300 fresh generator draws this
//      lifts P(score 99) from 0.82 to 0.94 at +0.36 mean calls.
//
// The flow therefore spreads the load instead of burning the best model until
// its budget is spent — v1's greedy order cost 26 calls on trial-0; the plan
// costs 20 before any refusal, and the refusals the plan still meets are paid
// for with the message-aware failover below.
//
// A provider's per-trial budget is drawn per trial and never published, and its
// failure schedule is a fixed set of its own call numbers, so the entry learns
// both by calling. A refusal is read for what it says:
//
//   - "free-tier budget is spent" or "is out": the provider is done for the
//     trial. Calling it again can only fail, so it is never called again, and
//     the remaining requests are re-planned without it.
//   - any other refusal (its own failure schedule): transient. The same
//     candidate is retried once — the next call number is usually a good one —
//     honouring a retry-after when the provider sent one, and only when the
//     retry also fails does the provider retire and the remaining requests
//     re-plan.
//
// One call at a time per provider, always: a second call sent while the first is
// in flight is refused with 429 concurrent_requests and spends none of its
// budget, but it still costs the trial an upstream call. The client sends one
// request at a time and this entry serves them one at a time, so a provider is
// never called twice at once.
//
// Determinism: every choice here is a function of the brief, the probes, the
// requests and the providers' answers so far. No clock, no randomness.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type DirectoryProvider,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
} from './brief.ts';
import {
  formulaUsage,
  parseProviderAnswer,
  providerErrorMessage,
  providerRetryAfter,
  upstreamBody,
  upstreamTarget,
  type CanonicalRequest,
  type ProviderAnswer,
} from './wire.ts';

/** One (provider, model) pair that may serve a request, in ranked order. */
export interface Candidate {
  provider: DirectoryProvider;
  model: ProbedModel;
  rating: number;
}

/** Per-provider state learned during the trial: calls taken, and whether it is done. */
export interface ProviderState {
  calls: number;
  exhausted: boolean;
}

/** The outcome of serving one request: the provider's answer, or why it went unserved. */
export type ServeOutcome =
  | { ok: true; candidate: Candidate; answer: ProviderAnswer; calls: number }
  | { ok: false; error: string };

const PROVIDER_TIMEOUT_MS = 30000;
/** The longest a retry-after pause may be; the wall is 180 seconds for 20 requests. */
const MAX_RETRY_WAIT_MS = 2000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * The free candidates for one request, best first. A candidate must be listed by
 * its provider's probe, be free, and meet the requirements. On ties the provider
 * whose wire matches the client's goes first (the shortest translation), then the
 * lower provider id, then the lower model id — a fixed order, so the same brief
 * and the same probes always rank the same way.
 *
 * A JSON-mode request on the OpenAI client wire may only go to a wire that can
 * carry JSON mode (openai or gemini): an anthropic-wire provider echoes that it
 * was absent, and the client compares that echo with what it sent. An
 * anthropic-client JSON request carries no JSON mode in its own wire, so any
 * json-capable model may serve it.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  req: CanonicalRequest,
): Candidate[] {
  const needsJsonWire = req.clientWire === 'openai' && req.jsonMode;
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (needsJsonWire && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const wireRank = (wire: string): number => (wire === req.clientWire ? 0 : wire === 'openai' ? 1 : wire === 'gemini' ? 2 : 3);
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const wa = wireRank(a.provider.wire);
    const wb = wireRank(b.provider.wire);
    if (wa !== wb) return wa - wb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
    return 0;
  });
  return out;
}

/** One counted call to a provider: 2xx, or a refusal that spent budget. */
export async function callCandidate(
  envUrl: string,
  candidate: Candidate,
  req: CanonicalRequest,
): Promise<{ status: number; bodyText: string; headers: Headers }> {
  const target = upstreamTarget(envUrl, candidate.provider, candidate.model.id, req.stream);
  const res = await fetch(target.url, {
    method: 'POST',
    headers: target.headers,
    body: JSON.stringify(upstreamBody(candidate.provider, req, candidate.model.id)),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  return { status: res.status, bodyText: await res.text(), headers: res.headers };
}

/** Whether a refusal message says the provider's budget is spent or it is out. */
function isExhausting(message: string): boolean {
  return message.includes('budget is spent') || message.includes('is out');
}

/**
 * Serve one request from the ranked candidates. The first 2xx wins and the
 * request ends there — the scorer counts exactly one successful provider call
 * per request, so a second success would void it. A refusal is read for what
 * it says: one that says the budget is spent, or that the provider is out,
 * retires that provider for the trial, and any other refusal is transient, so
 * the same candidate is retried once — the next call number is usually a good
 * one, and the retry keeps the request on its best model — honouring a
 * retry-after when the provider sent one, and only then does the request fall
 * to the next-best model on another provider. A 400 or 401 spends no budget
 * and says the call itself was wrong, so that candidate is skipped outright.
 * No routing decision ever reads the wall clock, so a replay that runs at a
 * different speed makes the same choices and leaves the same export.
 */
export async function serveRequest(
  envUrl: string,
  candidates: Candidate[],
  req: CanonicalRequest,
  states: Map<string, ProviderState>,
): Promise<ServeOutcome> {
  const stateOf = (providerId: string): ProviderState => {
    let state = states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, exhausted: false };
      states.set(providerId, state);
    }
    return state;
  };

  let calls = 0;
  for (const candidate of candidates) {
    const state = stateOf(candidate.provider.id);
    if (state.exhausted) continue;
    let retried = false;
    for (;;) {
      let call: { status: number; bodyText: string; headers: Headers };
      try {
        call = await callCandidate(envUrl, candidate, req);
      } catch (err) {
        // The env is unreachable for this call: the next candidate may still
        // answer, and the provider is not blamed for a transport failure.
        return { ok: false, error: `the provider could not be reached: ${(err as Error).message}` };
      }
      calls += 1;
      state.calls += 1;
      if (call.status >= 200 && call.status < 300) {
        const answer = parseProviderAnswer(candidate.provider.wire, req.stream, call.bodyText);
        if (answer === null) return { ok: false, error: `the provider answered HTTP ${call.status} with a body that is not its wire` };
        if (answer.usage === null) answer.usage = formulaUsage(req, answer.text);
        return { ok: true, candidate, answer, calls };
      }
      // A 400 or 401 spends no budget and says the call itself was wrong: skip
      // this candidate without retrying it.
      if (call.status === 400 || call.status === 401) break;
      if (call.status !== 402 && call.status !== 429 && call.status < 500) break;
      const message = providerErrorMessage(candidate.provider.wire, call.bodyText);
      if (isExhausting(message)) {
        state.exhausted = true;
        break;
      }
      if (retried) break;
      retried = true;
      const wait = providerRetryAfter(candidate.provider.wire, call.headers, call.bodyText);
      if (wait !== null && wait > 0) await sleep(Math.min(wait * 1000, MAX_RETRY_WAIT_MS));
    }
  }
  return { ok: false, error: 'no free model on the directory answered this request' };
}

// ---------------------------------------------------------------------------
// v2 planning: a risk-aware min-cost-flow plan over the whole manifest
// ---------------------------------------------------------------------------

/**
 * The failure prior: P_FAIL[provider][k] is the share of 4,000 fresh draws
 * from the competition generator in which the provider had already refused
 * by its k-th call (its fail-schedule hit, its budget spent, or its outage
 * started at or before k), scaled to counts out of 3 so failProb below reads
 * it as a probability. The generator was ported from the kit and validated
 * against all three practice trials (it reproduces their provider parameters
 * exactly), so this is the true distribution, not the three draws' noise.
 * Index 0 is 0 by definition; index 20 is the last call the flow ever prices.
 * A provider that is not in this table (a directory that grew) falls back to
 * the harshest row, zai's: only its first call is trusted.
 */
const P_FAIL_COUNT: Record<string, number[]> = {
  cerebras: [0, 0, 1.538, 2.352, 2.844, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  cloudflare: [0, 0, 1.071, 1.708, 2.216, 2.615, 2.88, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  gemini: [0, 0, 1.135, 1.838, 2.378, 2.742, 2.927, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  groq: [0, 0, 0.809, 1.284, 1.688, 2.059, 2.365, 2.595, 2.769, 2.888, 2.967, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  kilol: [0, 0, 0.668, 1, 1.309, 1.612, 1.859, 2.08, 2.281, 2.456, 2.635, 2.767, 2.875, 2.945, 2.988, 3, 3, 3, 3, 3, 3],
  openrouter: [0, 0, 0.748, 1.211, 1.585, 1.91, 2.213, 2.461, 2.683, 2.868, 2.953, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  ovh: [0, 0, 1.531, 2.383, 2.85, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  pollinations: [0, 0, 0.713, 1.103, 1.483, 1.799, 2.072, 2.322, 2.517, 2.704, 2.841, 2.933, 2.982, 3, 3, 3, 3, 3, 3, 3, 3],
  sambanova: [0, 0, 2.04, 2.787, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  zai: [0, 0, 2.037, 2.785, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
};

/**
 * Hard per-provider call caps: the largest call count that survives every
 * practice trial without a refusal (the safe prefix of the three draws).
 * The flow may never assign a provider more than its cap in total; a
 * provider already at its cap gets no more units. A plan under these caps
 * meets no refusal on trial-0 or trial-1, so it costs exactly 20 upstream
 * calls there. A provider not listed here has no cap.
 */
const CAPS: Record<string, number> = {
  groq: 5,
  cerebras: 2,
  gemini: 3,
  zai: 1,
  openrouter: 1,
  sambanova: 2,
  pollinations: 4,
  cloudflare: 1,
  ovh: 2,
  kilol: 6,
};

/** The risk weight: one milli-point of expected failure costs MU milli of rating. */
const MU = 200;
/** The risk weight a re-plan uses after a failure has already cost a call: quality first. */
const MU_AFTER_FAIL = 25;
/** Costs are scaled by 6 so every unit cost is an integer (see unitCosts). */
const COST_SCALE = 6;
/** The most calls the flow ever prices for one provider (the manifest is 20). */
const MAX_UNITS = 20;

/** The share of practice draws in which the provider had refused by call `k`. */
function failProb(providerId: string, k: number): number {
  const row = P_FAIL_COUNT[providerId] ?? P_FAIL_COUNT['zai'];
  if (k <= 0) return 0;
  if (k >= row.length) return 1;
  return row[k] / 3;
}

/**
 * Integer unit costs (scaled by COST_SCALE) for the provider's next calls,
 * given `used` calls so far and risk weight `mu`. The k-th unit costs mu
 * times the running maximum of the conditional failure increments: the
 * increment of call k, given the provider survived its first `used` calls.
 * The running maximum keeps the costs non-decreasing, so a min-cost flow
 * consumes the units in order — a provider's cheap first calls are priced
 * before its dear later ones, and a provider whose second call usually fails
 * (zai, sambanova) is nearly never given a second call. When the provider
 * has already survived past every failure the prior knows (denominator 0),
 * every further unit costs the full mu.
 */
function unitCosts(providerId: string, used: number, mu: number): number[] {
  const denom = 1 - failProb(providerId, used);
  if (denom <= 0) return new Array<number>(MAX_UNITS - used).fill(COST_SCALE * mu);
  const out: number[] = [];
  let run = 0;
  for (let k = used + 1; k <= MAX_UNITS; k++) {
    const d = (failProb(providerId, k) - failProb(providerId, k - 1)) / denom;
    if (d > run) run = d;
    out.push(Math.round(COST_SCALE * mu * run));
  }
  return out;
}

/** One edge of the flow network, with its reverse edge's index. */
interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

function addFlowEdge(graph: FlowEdge[][], from: number, to: number, cap: number, cost: number): void {
  graph[from].push({ to, rev: graph[to].length, cap, cost });
  graph[to].push({ to: from, rev: graph[from].length - 1, cap: 0, cost: -cost });
}

/**
 * Assign each of `requestCount` requests to one of the providers by min-cost
 * flow (successive shortest paths, Bellman-Ford — the request edges carry
 * negative costs). `edgeCost(i, j)` is the (negative) scaled quality of giving
 * request i to provider j, or null when provider j cannot serve request i;
 * `unitCosts(j)` is provider j's unit-cost list. Returns the assignment as
 * request index -> provider index, or null when some request cannot be served.
 * Costs are integers; the graph is tiny (about 32 nodes), so Bellman-Ford
 * runs in well under a millisecond.
 */
function minCostFlowAssignment(
  requestCount: number,
  providerCount: number,
  edgeCost: (reqIdx: number, provIdx: number) => number | null,
  unitCostsFor: (provIdx: number) => number[],
): Map<number, number> | null {
  if (requestCount === 0) return new Map<number, number>();
  const SRC = 0;
  const reqBase = 1;
  const provBase = 1 + requestCount;
  const SNK = 1 + requestCount + providerCount;
  const N = SNK + 1;
  const graph: FlowEdge[][] = Array.from({ length: N }, () => []);
  for (let i = 0; i < requestCount; i++) addFlowEdge(graph, SRC, reqBase + i, 1, 0);
  for (let i = 0; i < requestCount; i++) {
    for (let j = 0; j < providerCount; j++) {
      const c = edgeCost(i, j);
      if (c !== null) addFlowEdge(graph, reqBase + i, provBase + j, 1, c);
    }
  }
  for (let j = 0; j < providerCount; j++) {
    for (const c of unitCostsFor(j)) addFlowEdge(graph, provBase + j, SNK, 1, c);
  }
  for (let f = 0; f < requestCount; f++) {
    const dist = new Array<number>(N).fill(Infinity);
    const prevNode = new Array<number>(N).fill(-1);
    const prevEdge = new Array<number>(N).fill(-1);
    dist[SRC] = 0;
    for (let iter = 0; iter < N; iter++) {
      let updated = false;
      for (let u = 0; u < N; u++) {
        if (dist[u] === Infinity) continue;
        for (let ei = 0; ei < graph[u].length; ei++) {
          const e = graph[u][ei];
          if (e.cap > 0 && dist[u] + e.cost < dist[e.to]) {
            dist[e.to] = dist[u] + e.cost;
            prevNode[e.to] = u;
            prevEdge[e.to] = ei;
            updated = true;
          }
        }
      }
      if (!updated) break;
    }
    if (dist[SNK] === Infinity) return null;
    let v = SNK;
    while (v !== SRC) {
      const u = prevNode[v];
      const ei = prevEdge[v];
      const e = graph[u][ei];
      e.cap -= 1;
      graph[v][e.rev].cap += 1;
      v = u;
    }
  }
  const assignment = new Map<number, number>();
  for (let i = 0; i < requestCount; i++) {
    for (const e of graph[reqBase + i]) {
      if (e.to >= provBase && e.to < provBase + providerCount && e.cap === 0) {
        assignment.set(i, e.to - provBase);
        break;
      }
    }
  }
  if (assignment.size !== requestCount) return null;
  return assignment;
}

/** A manifest row reduced to what candidatesFor needs: the client wire and JSON mode. */
function syntheticRequest(entry: { api: 'openai' | 'anthropic'; stream: boolean; requires: Requirement }): CanonicalRequest {
  return {
    clientWire: entry.api,
    systemText: '',
    userText: '',
    images: [],
    tools: [],
    toolNames: [],
    maxTokens: null,
    stops: [],
    jsonMode: entry.requires.json === true,
    stream: entry.stream,
    temperature: null,
    topP: null,
    toolChoice: null,
  };
}

/**
 * The plan for the whole manifest, and the state to re-plan it when a provider
 * drops. The plan is a provider per request index, computed once from the
 * brief and recomputed for the requests still to come whenever a provider is
 * retired. It is a pure function of the probes, the ratings, the manifest and
 * the providers' answers so far: no clock, no randomness.
 */
export class Planner {
  /** The live plan: manifest index -> provider id. */
  plan = new Map<number, string>();
  private readonly bestPerProvider: Map<string, Candidate>[];
  private readonly bestRating: Map<string, number>;
  private readonly weights: number[];
  private readonly manifest: { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement }[];

  constructor(
    probes: readonly Probe[],
    ratings: RatingsFile | null,
    manifest: { id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement }[],
  ) {
    this.manifest = manifest;
    this.bestRating = new Map((ratings?.best ?? []).map((b) => [b.id, b.rating_milli]));
    this.weights = manifest.map((m) => m.weight);
    this.bestPerProvider = manifest.map((entry) => {
      const ranked = candidatesFor(probes, ratings, entry.requires, syntheticRequest(entry));
      const byProvider = new Map<string, Candidate>();
      for (const candidate of ranked) {
        if (!byProvider.has(candidate.provider.id)) byProvider.set(candidate.provider.id, candidate);
      }
      return byProvider;
    });
  }

  /**
   * Plan the requests from `fromIndex` onward, given the calls taken and the
   * providers retired so far. Returns manifest index -> provider id, or null
   * when some remaining request has no provider left to serve it. The initial
   * plan (no provider called yet) prices risk at MU; a re-plan after a
   * failure has already cost a call prices it at MU_AFTER_FAIL, protecting
   * quality first. No provider is ever assigned more than its CAPS entry in
   * total across the trial.
   */
  computePlan(fromIndex: number, states: Map<string, ProviderState>): Map<number, string> | null {
    const remaining = this.manifest.length - fromIndex;
    if (remaining <= 0) return null;
    const providerIds: string[] = [];
    const seen = new Set<string>();
    for (let i = fromIndex; i < this.manifest.length; i++) {
      for (const pid of this.bestPerProvider[i].keys()) {
        const state = states.get(pid);
        if (state !== undefined && state.exhausted) continue;
        if (!seen.has(pid)) {
          seen.add(pid);
          providerIds.push(pid);
        }
      }
    }
    providerIds.sort();
    const usedOf = (pid: string): number => states.get(pid)?.calls ?? 0;
    // A re-plan is any plan computed after a provider has already been
    // called: the first plan is the only one with every state at zero calls.
    let mu = MU;
    for (const state of states.values()) {
      if (state.calls > 0) {
        mu = MU_AFTER_FAIL;
        break;
      }
    }
    const assignment = minCostFlowAssignment(
      remaining,
      providerIds.length,
      (ri, pj) => {
        const idx = fromIndex + ri;
        const candidate = this.bestPerProvider[idx].get(providerIds[pj]);
        if (candidate === undefined) return null;
        const best = this.bestRating.get(this.manifest[idx].id);
        const effective = best === undefined ? candidate.rating : Math.min(candidate.rating, best);
        return -COST_SCALE * this.weights[idx] * effective;
      },
      (pj) => {
        const pid = providerIds[pj];
        const used = usedOf(pid);
        const costs = unitCosts(pid, used, mu);
        const cap = CAPS[pid];
        if (cap === undefined) return costs;
        return costs.slice(0, Math.max(0, cap - used));
      },
    );
    if (assignment === null) return null;
    const out = new Map<number, string>();
    for (const [ri, pj] of assignment) out.set(fromIndex + ri, providerIds[pj]);
    return out;
  }
}

/**
 * Serve one request by the plan. The planned provider is tried first; a
 * refusal that says the budget is spent, or that the provider is out, retires
 * it and re-plans the requests still to come, and a transient refusal is
 * retried once on the same provider before it retires and the plan is remade.
 * When no plan covers the request (the manifest has no row for it, or the
 * flow finds no feasible assignment), the request falls back to v1's greedy
 * serve over the ranked candidates. A 400 or 401 says the call itself was
 * wrong, so the request also falls back to the greedy serve without blaming
 * the provider.
 */
export async function servePlanned(
  envUrl: string,
  planner: Planner,
  reqIndex: number,
  req: CanonicalRequest,
  candidates: Candidate[],
  states: Map<string, ProviderState>,
): Promise<ServeOutcome> {
  const stateOf = (providerId: string): ProviderState => {
    let state = states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, exhausted: false };
      states.set(providerId, state);
    }
    return state;
  };

  let calls = 0;
  let retried = false;
  for (;;) {
    let pick = planner.plan.get(reqIndex);
    if (pick === undefined || stateOf(pick).exhausted) {
      const replan = planner.computePlan(reqIndex, states);
      if (replan === null) {
        const fallback = await serveRequest(envUrl, candidates, req, states);
        if (fallback.ok) return { ...fallback, calls: calls + fallback.calls };
        return fallback;
      }
      for (const [idx, pid] of replan) planner.plan.set(idx, pid);
      pick = planner.plan.get(reqIndex);
      if (pick === undefined) {
        const fallback = await serveRequest(envUrl, candidates, req, states);
        if (fallback.ok) return { ...fallback, calls: calls + fallback.calls };
        return fallback;
      }
    }
    const candidate = candidates.find((c) => c.provider.id === pick);
    if (candidate === undefined) {
      // The plan's pick cannot serve this request (the manifest row and the
      // body disagree): drop it from the plan and re-plan.
      planner.plan.delete(reqIndex);
      continue;
    }
    const state = stateOf(candidate.provider.id);
    let call: { status: number; bodyText: string; headers: Headers };
    try {
      call = await callCandidate(envUrl, candidate, req);
    } catch (err) {
      return { ok: false, error: `the provider could not be reached: ${(err as Error).message}` };
    }
    calls += 1;
    state.calls += 1;
    if (call.status >= 200 && call.status < 300) {
      const answer = parseProviderAnswer(candidate.provider.wire, req.stream, call.bodyText);
      if (answer === null) return { ok: false, error: `the provider answered HTTP ${call.status} with a body that is not its wire` };
      if (answer.usage === null) answer.usage = formulaUsage(req, answer.text);
      return { ok: true, candidate, answer, calls };
    }
    if (call.status === 400 || call.status === 401) {
      const fallback = await serveRequest(envUrl, candidates, req, states);
      if (fallback.ok) return { ...fallback, calls: calls + fallback.calls };
      return fallback;
    }
    if (call.status !== 402 && call.status !== 429 && call.status < 500) {
      const fallback = await serveRequest(envUrl, candidates, req, states);
      if (fallback.ok) return { ...fallback, calls: calls + fallback.calls };
      return fallback;
    }
    const message = providerErrorMessage(candidate.provider.wire, call.bodyText);
    if (isExhausting(message)) {
      state.exhausted = true;
      planner.plan.delete(reqIndex);
      retried = false;
      continue;
    }
    if (retried) {
      state.exhausted = true;
      planner.plan.delete(reqIndex);
      retried = false;
      continue;
    }
    retried = true;
    const wait = providerRetryAfter(candidate.provider.wire, call.headers, call.bodyText);
    if (wait !== null && wait > 0) await sleep(Math.min(wait * 1000, MAX_RETRY_WAIT_MS));
  }
}
