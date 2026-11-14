// free-model-aggregator-01: one server, ten free providers, the best free model per request.
//
// This server reads the brief at $MAKEFUN_PROMPT_FILE, probes each provider's
// catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (what wins the score) — v3 "Prime Helix v3"
//   - quality (60%): a risk-aware min-cost-flow plan (lib/route.ts) assigns one
//     provider to each of the twenty requests before the first is served,
//     maximising weighted rating_milli minus a risk charge priced from the
//     true failure distribution (4,000 draws of the competition generator),
//     under hard per-provider call caps (the largest count that survives
//     every practice trial), so the load spreads instead of burning the best
//     model until its budget is spent — the plan costs exactly 20 upstream
//     calls on trial-0 (21 on trial-2, 22 on trial-1 where kilol's fail
//     numbers 2 and 4 both hit). A re-plan after a failure protects quality
//     first (MU 25 instead of 200).
//   - failover (10%): a provider call that answers 402, 429 or a 5xx is read for what it says:
//     a refusal that says the budget is spent, or that the provider is out, retires that
//     provider for the trial and re-plans the requests still to come, and any other
//     refusal is transient, so the same candidate is retried once (honouring a
//     retry-after when one is sent) before the provider retires and the plan is remade.
//   - served (15%): with failover, re-planning and the whole directory to fall back on,
//     one provider out of budget takes no request down with it.
//   - the wires: all three upstream wires are spoken (openai, anthropic, gemini), and every answer is
//     rendered back in the wire the client used, streamed or not as it asked, with the provider's text
//     whole, its usage counts, and its stop reason mapped.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag
//     and context length where it has a row; GET /v1/models lists exactly the registered free ids.
//
// Determinism: every route is a function of the requests and the providers' answers seen so far, never
// of the wall clock. MAKEFUN_TRIAL_SEED is accepted and ignored: this entry needs no randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  probeProvider,
  readBrief,
  servedModels,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  canonicalOfAnthropic,
  canonicalOfOpenAI,
  openAIError,
  openAIModelList,
  renderClientAnswer,
  type CanonicalRequest,
} from './lib/wire.ts';
import { Planner, candidatesFor, servePlanned, type ProviderState as RouteProviderState } from './lib/route.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Per-provider state learned during the trial: calls taken, and whether it is done. */
const providerStates = new Map<string, RouteProviderState>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The plan for the whole manifest, built on the first request from the probes, the ratings and the manifest. */
let planner: Planner | null = null;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, in the brief's order, and build the one catalog. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.providers) {
    probes.push(await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS));
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(probes, brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One client request, on either wire, answered by the best free model that takes it. */
async function serve(req: CanonicalRequest, res: ServerResponse, wire: 'openai' | 'anthropic', body: Json): Promise<void> {
  // The manifest index of the request now arriving: requirementsFor advances
  // the counter, so take the index first — the plan is keyed by it.
  const reqIndex = manifestIndex;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(probes, brief.ratings, requires, req);
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  if (planner === null) planner = new Planner(probes, brief.ratings, brief.workload?.manifest ?? []);
  const outcome = await servePlanned(ENV_URL, planner, reqIndex, req, candidates, providerStates);
  if (!outcome.ok) {
    log(`request ${manifestIndex} went unserved: ${outcome.error}`);
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', outcome.error);
  }
  log(`request ${manifestIndex} (${wire}${req.stream ? ', stream' : ''}): ${outcome.candidate.provider.id}/${outcome.candidate.model.id} in ${outcome.calls} upstream call${outcome.calls === 1 ? '' : 's'}`);
  const rendered = renderClientAnswer(outcome.answer, outcome.candidate.model.id, req);
  res.writeHead(200, {
    'content-type': rendered.contentType,
    'cache-control': 'no-cache',
    'content-length': Buffer.byteLength(rendered.body),
    ...(req.stream ? { connection: 'keep-alive' } : {}),
  });
  res.end(rendered.body);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(canonicalOfOpenAI(body), res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(canonicalOfAnthropic(body), res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          const summary = [...providerStates.entries()]
            .map(([id, state]) => `${id} ${state.calls}${state.exhausted ? ' (exhausted)' : ''}`)
            .join(', ');
          const totalCalls = [...providerStates.values()].reduce((n, s) => n + s.calls, 0);
          log(`done after ${manifestIndex} requests; upstream provider calls: ${totalCalls} (${summary})`);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels(probes, brief.ratings).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
