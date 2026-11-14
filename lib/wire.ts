// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. The entry answers every request from one of the ten providers, speaking
// that provider's own wire:
//
//   openai     POST <base>/v1/chat/completions   Authorization: Bearer <key>
//   anthropic  POST <base>/v1/messages           x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     POST <base>/v1beta/models/<model>:generateContent | :streamGenerateContent
//                                                 x-goog-api-key: <key>
//
// Every request is first reduced to one canonical form — the fields the provider
// echoes back and the client compares — then rebuilt in the provider's wire without
// rewriting or dropping a field: the system prompt, the user text (images included),
// the tool names in order, the stop sequences, the max_tokens, the JSON mode. The
// provider's answer is buffered whole, parsed to (text, usage, stop), and rendered
// back in the wire the client used, streamed or not as it asked, with the
// provider's text whole, its usage counts, and its stop reason mapped end_turn to
// stop, max_tokens to length, tool_use to tool_calls and refusal to content_filter.
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// The two model listings use fixed timestamps for the same reason.

import { authHeaders, isRecord, joinBase, type DirectoryProvider, type Json, type Wire } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters of answer text one SSE content piece carries. */
const SSE_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into SSE-sized pieces; an empty text is one empty piece. */
export function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE_CHARS) pieces.push(text.slice(at, at + SSE_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

// ---------------------------------------------------------------------------
// Canonical request
// ---------------------------------------------------------------------------

/** One image as the client sent it: inline base64, or a remote URL. */
export type CanonicalImage = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; url: string };

/** One tool as the client defined it, in the client's order. */
export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

/**
 * One client request reduced to the fields the provider echoes and the client
 * compares. Text blocks join with "\n" exactly as the providers parse them; the
 * user text and images are those of the last user message (a provider overwrites
 * them per message); the system text joins every system part with "\n".
 */
export interface CanonicalRequest {
  clientWire: 'openai' | 'anthropic';
  systemText: string;
  userText: string;
  images: CanonicalImage[];
  tools: CanonicalTool[];
  toolNames: string[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | null;
  topP: number | null;
  toolChoice: Json | string | null;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function imagesOf(content: unknown): CanonicalImage[] {
  if (!Array.isArray(content)) return [];
  const out: CanonicalImage[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) out.push({ kind: 'base64', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2] });
      else out.push({ kind: 'url', url });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        out.push({ kind: 'base64', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
      } else if (source.type === 'url' && typeof source.url === 'string') out.push({ kind: 'url', url: source.url });
    }
  }
  return out;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.flatMap((x) => (typeof x === 'string' ? [x] : []));
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** An OpenAI Chat Completions body as the canonical request. */
export function canonicalOfOpenAI(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let images: CanonicalImage[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = textOf(message.content);
      images = imagesOf(message.content);
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    clientWire: 'openai',
    systemText: systemParts.join('\n'),
    userText,
    images,
    tools,
    toolNames: tools.map((t) => t.name),
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    toolChoice: isRecord(body.tool_choice) || typeof body.tool_choice === 'string' ? (body.tool_choice as Json | string) : null,
  };
}

/** An Anthropic Messages body as the canonical request. */
export function canonicalOfAnthropic(body: Json): CanonicalRequest {
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let images: CanonicalImage[] = [];
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = textOf(message.content);
    images = imagesOf(message.content);
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    clientWire: 'anthropic',
    systemText,
    userText,
    images,
    tools,
    toolNames: tools.map((t) => t.name),
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    toolChoice: isRecord(body.tool_choice) ? (body.tool_choice as Json) : null,
  };
}

// ---------------------------------------------------------------------------
// Upstream requests: canonical form in each provider wire
// ---------------------------------------------------------------------------

function openAIImagePart(image: CanonicalImage): Json {
  if (image.kind === 'base64') return { type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } };
  return { type: 'image_url', image_url: { url: image.url } };
}

function anthropicImageBlock(image: CanonicalImage): Json {
  if (image.kind === 'base64') return { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } };
  return { type: 'image', source: { type: 'url', url: image.url } };
}

function geminiImagePart(image: CanonicalImage): Json {
  if (image.kind === 'base64') return { inlineData: { mimeType: image.mime, data: image.data } };
  return { fileData: { fileUri: image.url } };
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  const userParts: Json[] = [{ type: 'text', text: req.userText }];
  for (const image of req.images) userParts.push(openAIImagePart(image));
  messages.push({ role: 'user', content: userParts });
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (req.toolChoice !== null) out.tool_choice = req.toolChoice;
  }
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    // Ask the provider to close its stream with a usage chunk: the client never
    // sets stream_options, and without it the provider's SSE carries no usage.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function toAnthropic(req: CanonicalRequest, model: string): Json {
  const content: Json[] = [{ type: 'text', text: req.userText }];
  for (const image of req.images) content.push(anthropicImageBlock(image));
  const out: Json = { model, messages: [{ role: 'user', content }] };
  if (req.systemText !== '') out.system = req.systemText;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    if (req.toolChoice !== null) out.tool_choice = req.toolChoice;
  }
  if (req.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body. */
export function toGemini(req: CanonicalRequest): Json {
  const parts: Json[] = [{ text: req.userText }];
  for (const image of req.images) parts.push(geminiImagePart(image));
  const out: Json = { contents: [{ role: 'user', parts }] };
  if (req.systemText !== '') out.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Where one provider's inference route lives, and the headers its key wants. */
export function upstreamTarget(envUrl: string, provider: DirectoryProvider, model: string, stream: boolean): { url: string; headers: Record<string, string> } {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const encoded = encodeURIComponent(model);
  return {
    url: `${base}/v1beta/models/${encoded}:${stream ? 'streamGenerateContent' : 'generateContent'}`,
    headers,
  };
}

/** The upstream body for one provider wire. */
export function upstreamBody(provider: DirectoryProvider, req: CanonicalRequest, model: string): Json {
  if (provider.wire === 'openai') return toOpenAI(req, model);
  if (provider.wire === 'anthropic') return toAnthropic(req, model);
  return toGemini(req);
}

// ---------------------------------------------------------------------------
// Upstream answers: (text, usage, stop) from any wire, JSON or SSE
// ---------------------------------------------------------------------------

export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stop: string | null;
}

/** The provider's deterministic usage formula, for when its answer carries none. */
export function formulaUsage(req: CanonicalRequest, text: string): { input: number; output: number } {
  const prompt = req.systemText.length + req.userText.length + req.toolNames.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage)
    ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) }
    : null;
  return { text, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
}

function parseOpenAIStream(text: string): ProviderAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, usage, stop };
}

function parseAnthropicJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = textOf(body.content);
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) } : null;
  return { text, usage, stop: typeof body.stop_reason === 'string' ? body.stop_reason : null };
}

function parseAnthropicStream(text: string): ProviderAnswer {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

function geminiTextOf(body: unknown): string {
  if (!isRecord(body)) return '';
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return '';
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  // Gemini parts carry a bare `text` field and no `type: 'text'`: take any part
  // with a string text, in order, joined with "\n" as the provider parses them.
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

function parseGeminiJson(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const text = geminiTextOf(body);
  const usage = isRecord(body.usageMetadata)
    ? { input: numberOrZero(body.usageMetadata.promptTokenCount), output: numberOrZero(body.usageMetadata.candidatesTokenCount) }
    : null;
  return { text, usage, stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null };
}

function parseGeminiStream(text: string): ProviderAnswer {
  let out = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    out += geminiTextOf(chunk);
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(chunk.usageMetadata)) {
      usage = { input: numberOrZero(chunk.usageMetadata.promptTokenCount), output: numberOrZero(chunk.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, usage, stop };
}

/** One provider's whole answer — JSON body or SSE stream — as (text, usage, stop). */
export function parseProviderAnswer(wire: Wire, stream: boolean, bodyText: string): ProviderAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (wire === 'openai') return parseOpenAIJson(body);
    if (wire === 'anthropic') return parseAnthropicJson(body);
    return parseGeminiJson(body);
  }
  if (wire === 'openai') return parseOpenAIStream(bodyText);
  if (wire === 'anthropic') return parseAnthropicStream(bodyText);
  return parseGeminiStream(bodyText);
}

/** A provider error body's retry_after, from its header or its envelope. */
export function providerRetryAfter(wire: Wire, headers: Headers, bodyText: string): number | null {
  const header = headers.get('retry-after');
  if (header !== null) {
    const n = Number(header);
    if (Number.isFinite(n) && n > 0) return n;
  }
  try {
    const body: unknown = JSON.parse(bodyText);
    if (!isRecord(body)) return null;
    if (isRecord(body.error) && typeof body.error.retry_after === 'number') return body.error.retry_after;
  } catch {
    // Not JSON: no retry_after inside.
  }
  return null;
}

/** A provider error body's message, if it carries one in its wire's envelope. */
export function providerErrorMessage(wire: Wire, bodyText: string): string {
  let body: unknown;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return bodyText.slice(0, 300);
  }
  if (!isRecord(body)) return bodyText.slice(0, 300);
  if (isRecord(body.error) && typeof body.error.message === 'string') return body.error.message;
  if (typeof body.message === 'string') return body.message;
  return bodyText.slice(0, 300);
}

// ---------------------------------------------------------------------------
// Client answers: the provider's (text, usage, stop) in the client's wire
// ---------------------------------------------------------------------------

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: string | null): string {
  if (finish === 'length' || finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call' || finish === 'tool_use') return 'tool_use';
  if (finish === 'content_filter' || finish === 'refusal' || finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: string | null): string {
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_use' || stop === 'function_call' || stop === 'tool_calls') return 'tool_calls';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Map a provider's stop reason into the client's wire: OpenAI or Anthropic spelling. */
export function clientStop(providerStop: string | null, clientWire: 'openai' | 'anthropic'): string {
  if (clientWire === 'openai') return finishReasonOf(providerStop);
  return stopReasonOf(providerStop);
}

function openAIUsage(usage: { input: number; output: number }): Json {
  return { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
}

/** The provider's answer as an OpenAI Chat Completions object naming `model`. */
export function renderOpenAIJson(answer: ProviderAnswer, model: string, req: CanonicalRequest): Json {
  const usage = answer.usage ?? formulaUsage(req, answer.text);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: clientStop(answer.stop, 'openai'), logprobs: null }],
    usage: openAIUsage(usage),
  };
}

/** The provider's answer as an OpenAI event stream naming `model`. */
export function renderOpenAIStream(answer: ProviderAnswer, model: string, req: CanonicalRequest): string {
  const usage = answer.usage ?? formulaUsage(req, answer.text);
  const finish = clientStop(answer.stop, 'openai');
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, finish)));
  // The usage chunk: the client never asks for it, but without one its usage
  // check has nothing to compare, so the stream must carry it.
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: openAIUsage(usage) }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The provider's answer as an Anthropic Messages object naming `model`. */
export function renderAnthropicJson(answer: ProviderAnswer, model: string, req: CanonicalRequest): Json {
  const usage = answer.usage ?? formulaUsage(req, answer.text);
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The provider's answer as an Anthropic event stream naming `model`. */
export function renderAnthropicStream(answer: ProviderAnswer, model: string, req: CanonicalRequest): string {
  const usage = answer.usage ?? formulaUsage(req, answer.text);
  const stop = clientStop(answer.stop, 'anthropic');
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** The provider's answer in the client's wire, streamed or not as it asked. */
export function renderClientAnswer(answer: ProviderAnswer, model: string, req: CanonicalRequest): { contentType: string; body: string } {
  if (req.clientWire === 'openai') {
    if (req.stream) return { contentType: 'text/event-stream', body: renderOpenAIStream(answer, model, req) };
    return { contentType: 'application/json', body: JSON.stringify(renderOpenAIJson(answer, model, req)) };
  }
  if (req.stream) return { contentType: 'text/event-stream', body: renderAnthropicStream(answer, model, req) };
  return { contentType: 'application/json', body: JSON.stringify(renderAnthropicJson(answer, model, req)) };
}
