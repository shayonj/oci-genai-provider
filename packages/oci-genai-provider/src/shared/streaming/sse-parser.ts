import { createParser, type EventSourceMessage } from 'eventsource-parser';
import type { LanguageModelV3FinishReason } from '@ai-sdk/provider';
import type { StreamPart, UnifiedFinishReason, FinishPart } from './types';

const FINISH_REASON_MAP: Record<string, UnifiedFinishReason> = {
  STOP: 'stop',
  LENGTH: 'length',
  CONTENT_FILTER: 'content-filter',
  TOOL_CALLS: 'tool-calls',
  ERROR: 'error',
};

/**
 * Maps OCI GenAI finish reasons to AI SDK v3 LanguageModelV3FinishReason structure.
 */
export function mapFinishReason(reason: string): LanguageModelV3FinishReason {
  return {
    unified: FINISH_REASON_MAP[reason] ?? 'other',
    raw: reason,
  };
}

/**
 * OCI streaming format (2025+):
 * Text chunk:  {"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":"..."}]},"pad":"..."}
 * Tool call:   {"message":{"role":"ASSISTANT","toolCalls":[{"id":"...","type":"FUNCTION","function":{"name":"...","arguments":"..."}}]},...}
 * Finish:      {"message":{"role":"ASSISTANT"},"finishReason":"stop","pad":"..."}
 */
interface OCIStreamToolCall {
  id?: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
  // Cohere format
  name?: string;
  parameters?: Record<string, unknown>;
}

interface OCIStreamChunk {
  index?: number;
  message?: {
    role?: string;
    content?: Array<{ type?: string; text?: string; thinking?: string }>;
    toolCalls?: OCIStreamToolCall[];
    reasoningContent?: string;
  };
  // Cohere format: tool calls at top level
  toolCalls?: OCIStreamToolCall[];
  finishReason?: string;
  pad?: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    completionTokensDetails?: {
      reasoningTokens?: number;
      acceptedPredictionTokens?: number;
      rejectedPredictionTokens?: number;
    };
  };
}

/**
 * Parses an OCI GenAI streaming response.
 * Accepts either a ReadableStream directly (new OCI SDK behavior) or a Response object.
 */
export async function* parseSSEStream(
  input: ReadableStream<Uint8Array> | Response,
  options?: { includeRawChunks?: boolean; apiFormat?: 'GENERIC' | 'COHERE' | 'COHEREV2' }
): AsyncGenerator<StreamPart> {
  // Handle both ReadableStream and Response objects
  const stream = input instanceof ReadableStream ? input : input.body;
  if (!stream) {
    throw new Error('Response body is not readable');
  }

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const parts: StreamPart[] = [];
  let yieldedIndex = 0;

  // Accumulate tool calls across streaming chunks (the OCI native API may
  // deliver tool-call arguments incrementally, like the OpenAI streaming
  // format). Keyed by index within the current response.
  const pendingToolCalls = new Map<number, { id: string; name: string; arguments: string }>();

  // Buffer for finish event to combine reason and usage
  let lastFinishReason: UnifiedFinishReason = 'stop';
  let lastRawFinishReason = 'STOP';
  let lastUsage: FinishPart['usage'] = { promptTokens: 0, completionTokens: 0 };
  let hasFinishOrUsage = false;

  const includeRawChunks = options?.includeRawChunks ?? false;

  const parser = createParser({
    onEvent: (event: EventSourceMessage) => {
      const data = event.data;
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data) as OCIStreamChunk;

        if (includeRawChunks) {
          parts.push({
            type: 'raw',
            rawValue: parsed,
          });
        }

        // Check for reasoning delta (Generic or CohereV2 format)
        if (parsed.message?.reasoningContent) {
          parts.push({
            type: 'reasoning-delta',
            reasoningDelta: parsed.message.reasoningContent,
          });
        }

        // Support for top-level text or content[0].text (Older Cohere format)
        const topLevelText = (parsed as any).text || (parsed as any).textDelta;
        if (topLevelText) {
          parts.push({
            type: 'text-delta',
            textDelta: topLevelText,
          });
        }

        if (parsed.message?.content) {
          for (const part of parsed.message.content) {
            if (part.type === 'THINKING' && part.thinking) {
              parts.push({
                type: 'reasoning-delta',
                reasoningDelta: part.thinking,
              });
            }
          }
        }

        // Check for text delta in message content
        const contentParts = parsed.message?.content;
        if (contentParts) {
          for (const part of contentParts) {
            if ((part.type === 'TEXT' || !part.type) && part.text) {
              parts.push({
                type: 'text-delta',
                textDelta: part.text,
              });
            }
          }
        }

        // Accumulate tool calls across streaming chunks. The OCI native
        // API may split tool-call arguments across multiple SSE events.
        const toolCalls = parsed.message?.toolCalls ?? parsed.toolCalls;
        if (toolCalls && toolCalls.length > 0) {
          for (let idx = 0; idx < toolCalls.length; idx++) {
            const toolCall = toolCalls[idx];
            const current = pendingToolCalls.get(idx) ?? {
              id: toolCall.id ?? `tool-call-${Date.now()}-${idx}`,
              name: '',
              arguments: '',
            };
            current.id = toolCall.id ?? current.id;
            if (toolCall.function?.name) {
              current.name = toolCall.function.name;
              current.arguments += toolCall.function.arguments ?? '';
            } else if (toolCall.name) {
              current.name = toolCall.name;
              current.arguments = JSON.stringify(toolCall.parameters ?? {});
            }
            pendingToolCalls.set(idx, current);
          }
        }

        // Buffer finish and usage
        if (parsed.finishReason || parsed.usage) {
          hasFinishOrUsage = true;
          if (parsed.finishReason) {
            lastRawFinishReason = parsed.finishReason.toUpperCase();
            lastFinishReason = FINISH_REASON_MAP[lastRawFinishReason] ?? 'other';
          }
          if (parsed.usage) {
            const usage = parsed.usage;
            const tokenDetails = usage.completionTokensDetails;
            lastUsage = {
              promptTokens:
                usage.promptTokens ?? (usage as any).promptTokenCount ?? lastUsage.promptTokens,
              completionTokens:
                usage.completionTokens ??
                (usage as any).completionTokenCount ??
                lastUsage.completionTokens,
              reasoningTokens: tokenDetails?.reasoningTokens ?? lastUsage.reasoningTokens,
              acceptedPredictionTokens:
                tokenDetails?.acceptedPredictionTokens ?? lastUsage.acceptedPredictionTokens,
              rejectedPredictionTokens:
                tokenDetails?.rejectedPredictionTokens ?? lastUsage.rejectedPredictionTokens,
            };
          }
        }
      } catch (error) {
        // Log parsing errors - silent failures here cause major debugging issues
        console.error('[OCI SSE Parser] Failed to parse chunk:', {
          data: data.length > 200 ? data.substring(0, 200) + '...' : data,
          error: error instanceof Error ? error.message : String(error),
        });

        if (includeRawChunks) {
          parts.push({
            type: 'raw',
            rawValue: data,
          });
        }
      }
    },
  });

  while (true) {
    const result = await reader.read();

    if (result.done) break;

    parser.feed(decoder.decode(result.value, { stream: true }));

    // Yield any parts that were parsed
    while (yieldedIndex < parts.length) {
      yield parts[yieldedIndex++];
    }
  }

  // Emit accumulated tool calls before the finish event
  for (const toolCall of pendingToolCalls.values()) {
    if (toolCall.name) {
      yield {
        type: 'tool-call' as const,
        toolCallId: toolCall.id,
        toolName: toolCall.name,
        input: toolCall.arguments || '{}',
      };
    }
  }

  // Always yield a finish event - use defaults if none received from API
  yield {
    type: 'finish',
    finishReason: hasFinishOrUsage
      ? { unified: lastFinishReason, raw: lastRawFinishReason }
      : { unified: 'other', raw: 'INCOMPLETE' },
    usage: lastUsage,
  };

  // Yield any remaining parts
  while (yieldedIndex < parts.length) {
    yield parts[yieldedIndex++];
  }
}
