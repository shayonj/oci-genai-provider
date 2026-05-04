import type {
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from '@ai-sdk/provider';
import { createParser, type EventSourceMessage } from 'eventsource-parser';
import { convertToOCIMessages, type OCIMessage } from './converters/messages';
import { convertToOCIToolChoice, convertToOCITools } from './converters/tools';
import { getAPIKey, getCompartmentId, getOpenAICompatibleEndpoint } from '../auth/index.js';
import { composeRequest } from 'oci-common';
import type { GenerativeAiInferenceClient } from 'oci-generativeaiinference';
import type { OCIConfig, OCIProviderOptions } from '../types';

interface OpenAICompatibleMessageContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

type OpenAICompatibleMessage =
  | { role: 'system' | 'user'; content: string | OpenAICompatibleMessageContentPart[] }
  | {
      role: 'assistant';
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    }
  | { role: 'tool'; content: string; tool_call_id: string };

interface OpenAICompatibleChatChunk {
  choices?: Array<{
    delta?: {
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

type Uint8ArrayReadResult = { done: true; value?: undefined } | { done: false; value: Uint8Array };

const FINISH_REASON_MAP: Record<string, LanguageModelV3FinishReason['unified']> = {
  stop: 'stop',
  length: 'length',
  tool_calls: 'tool-calls',
  content_filter: 'content-filter',
};

function isUint8ArrayReadableResult(value: unknown): value is Uint8ArrayReadResult {
  if (!value || typeof value !== 'object' || !('done' in value)) {
    return false;
  }

  const candidate = value as { done?: unknown; value?: unknown };
  return (
    typeof candidate.done === 'boolean' &&
    (candidate.done === true || candidate.value instanceof Uint8Array)
  );
}

export async function doOpenAICompatibleStream(
  modelId: string,
  config: OCIConfig,
  options: LanguageModelV3CallOptions,
  ociOptions: OCIProviderOptions | undefined,
  warnings: SharedV3Warning[]
): Promise<LanguageModelV3StreamResult> {
  const promptMessages = convertToOCIMessages(options.prompt);
  const endpoint = `${getOpenAICompatibleEndpoint(config)}/chat/completions`;
  const apiKey = getAPIKey(config);
  const compartmentId = ociOptions?.compartmentId ?? getCompartmentId(config);
  const body = createOpenAICompatibleRequestBody(modelId, promptMessages, options, ociOptions);

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'x-oci-compartment-id': compartmentId,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `OCI OpenAI-compatible request failed (${response.status} ${response.statusText})${errorBody ? `: ${errorBody}` : ''}`
    );
  }

  const requestId = response.headers.get('opc-request-id') ?? undefined;
  const requestBody = JSON.stringify(body);

  return {
    request: {
      body: requestBody,
    },
    response: {
      headers: requestId ? { 'opc-request-id': requestId } : undefined,
    },
    stream: createOpenAICompatibleResponseStream(response, requestId, warnings),
  };
}

function createOpenAICompatibleRequestBody(
  modelId: string,
  messages: OCIMessage[],
  options: LanguageModelV3CallOptions,
  ociOptions: OCIProviderOptions | undefined
): Record<string, unknown> {
  const functionTools = options.tools?.filter((tool) => tool.type === 'function') ?? [];
  const tokenLimit = options.maxOutputTokens;
  const useMaxCompletionTokens = modelId.includes('gpt-5') || modelId.includes('gpt-oss');

  return {
    model: modelId,
    messages: messages.map(convertMessageToOpenAICompatibleFormat),
    stream: true,
    ...(tokenLimit !== undefined
      ? useMaxCompletionTokens
        ? { max_completion_tokens: tokenLimit }
        : { max_tokens: tokenLimit }
      : {}),
    temperature: options.temperature,
    top_p: options.topP,
    stop: options.stopSequences,
    seed: options.seed,
    ...(functionTools.length > 0
      ? {
          tools: convertToOCITools(functionTools, 'GENERIC').map((tool) => ({
            type: 'function',
            function: {
              name: tool.name,
              description: tool.description,
              parameters: 'parameters' in tool ? tool.parameters : {},
            },
          })),
        }
      : {}),
    ...(options.toolChoice
      ? {
          tool_choice: toOpenAICompatibleToolChoice(convertToOCIToolChoice(options.toolChoice)),
        }
      : {}),
    ...(ociOptions?.reasoningEffort && modelId.startsWith('openai.gpt-oss')
      ? { reasoning_effort: ociOptions.reasoningEffort }
      : {}),
  };
}

function convertMessageToOpenAICompatibleFormat(message: OCIMessage): OpenAICompatibleMessage {
  switch (message.role) {
    case 'SYSTEM':
      return {
        role: 'system',
        content: serializeOpenAICompatibleContent(message.content),
      };
    case 'USER':
      return {
        role: 'user',
        content: serializeOpenAICompatibleContent(message.content),
      };
    case 'ASSISTANT':
      return {
        role: 'assistant',
        content: serializeAssistantContent(message.content),
        ...(message.toolCalls
          ? {
              tool_calls: message.toolCalls.map((toolCall) => ({
                id: toolCall.id,
                type: 'function' as const,
                function: {
                  name: toolCall.name,
                  arguments: toolCall.arguments,
                },
              })),
            }
          : {}),
      };
    case 'TOOL':
      return {
        role: 'tool',
        tool_call_id: message.toolCallId ?? 'tool-call-missing-id',
        content: message.content
          .filter((content) => content.type === 'TEXT')
          .map((content) => content.text)
          .join('\n'),
      };
  }
}

function serializeOpenAICompatibleContent(
  content: OCIMessage['content']
): string | OpenAICompatibleMessageContentPart[] {
  if (content.length === 1 && content[0]?.type === 'TEXT') {
    return content[0].text;
  }

  return content.map((part) =>
    part.type === 'TEXT'
      ? { type: 'text', text: part.text }
      : { type: 'image_url', image_url: { url: part.imageUrl.url } }
  );
}

function serializeAssistantContent(content: OCIMessage['content']): string | null {
  const text = content
    .filter((part) => part.type === 'TEXT')
    .map((part) => part.text)
    .join('\n');

  return text.length > 0 ? text : null;
}

function toOpenAICompatibleToolChoice(choice: ReturnType<typeof convertToOCIToolChoice>): unknown {
  switch (choice.type) {
    case 'AUTO':
      return 'auto';
    case 'REQUIRED':
      return 'required';
    case 'NONE':
      return 'none';
    case 'FUNCTION':
      return {
        type: 'function',
        function: { name: choice.function.name },
      };
  }
}

function createOpenAICompatibleResponseStream(
  response: Response,
  requestId: string | undefined,
  warnings: SharedV3Warning[]
): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller): Promise<void> {
      controller.enqueue({ type: 'stream-start', warnings });

      const body = response.body as ReadableStream<Uint8Array> | null;
      if (!body) {
        controller.enqueue({
          type: 'error',
          error: new Error('OCI OpenAI-compatible response body is missing'),
        });
        controller.close();
        return;
      }

      const reader = body.getReader();
      const decoder = new TextDecoder();
      const toolCalls = new Map<
        number,
        {
          id: string;
          name: string;
          arguments: string;
        }
      >();
      let promptTokens = 0;
      let completionTokens = 0;
      let finishReason: LanguageModelV3FinishReason = { unified: 'other', raw: 'INCOMPLETE' };
      let textStarted = false;
      let reasoningStarted = false;
      const textId = `text-${Date.now()}`;
      const reasoningId = `reasoning-${Date.now()}`;

      const parser = createParser({
        onEvent(event: EventSourceMessage): void {
          if (event.data === '[DONE]') {
            return;
          }

          const parsed = JSON.parse(event.data) as OpenAICompatibleChatChunk;
          const choice = parsed.choices?.[0];

          if (choice?.delta?.content) {
            if (!textStarted) {
              controller.enqueue({ type: 'text-start', id: textId });
              textStarted = true;
            }
            controller.enqueue({ type: 'text-delta', id: textId, delta: choice.delta.content });
          }

          if (choice?.delta?.reasoning_content) {
            if (!reasoningStarted) {
              controller.enqueue({ type: 'reasoning-start', id: reasoningId });
              reasoningStarted = true;
            }
            controller.enqueue({
              type: 'reasoning-delta',
              id: reasoningId,
              delta: choice.delta.reasoning_content,
            });
          }

          for (const toolCall of choice?.delta?.tool_calls ?? []) {
            const index = toolCall.index ?? 0;
            const current = toolCalls.get(index) ?? {
              id: toolCall.id ?? `tool-call-${Date.now()}-${index}`,
              name: '',
              arguments: '',
            };
            current.id = toolCall.id ?? current.id;
            current.name = toolCall.function?.name ?? current.name;
            current.arguments += toolCall.function?.arguments ?? '';
            toolCalls.set(index, current);
          }

          if (choice?.finish_reason) {
            finishReason = {
              unified: FINISH_REASON_MAP[choice.finish_reason] ?? 'other',
              raw: choice.finish_reason,
            };
          }

          if (parsed.usage) {
            promptTokens = parsed.usage.prompt_tokens ?? promptTokens;
            completionTokens = parsed.usage.completion_tokens ?? completionTokens;
          }
        },
      });

      try {
        while (true) {
          const readResultUnknown: unknown = await reader.read();
          if (!isUint8ArrayReadableResult(readResultUnknown)) {
            throw new Error('OCI OpenAI-compatible stream returned an invalid chunk');
          }

          if (readResultUnknown.done) {
            break;
          }

          const chunk = readResultUnknown.value;
          parser.feed(decoder.decode(chunk, { stream: true }));
        }

        for (const toolCall of toolCalls.values()) {
          controller.enqueue({
            type: 'tool-call',
            toolCallId: toolCall.id,
            toolName: toolCall.name,
            input: toolCall.arguments || '{}',
          });
        }

        if (textStarted) {
          controller.enqueue({ type: 'text-end', id: textId });
        }

        if (reasoningStarted) {
          controller.enqueue({ type: 'reasoning-end', id: reasoningId });
        }

        controller.enqueue({
          type: 'finish',
          finishReason,
          usage: {
            inputTokens: {
              total: promptTokens,
              noCache: undefined,
              cacheRead: undefined,
              cacheWrite: undefined,
            },
            outputTokens: {
              total: completionTokens,
              text: completionTokens,
              reasoning: undefined,
            },
          },
          providerMetadata: {
            oci: {
              requestId,
              transport: 'openai-compatible',
            },
          },
        });
      } catch (error) {
        controller.enqueue({
          type: 'error',
          error: error instanceof Error ? error : new Error(String(error)),
        });
      } finally {
        controller.close();
      }
    },
  });
}

/**
 * Like doOpenAICompatibleStream but uses OCI request signing instead of a
 * Bearer API key. This lets config_file auth users hit the OpenAI-compatible
 * endpoint, which correctly returns tool-call arguments (the native GENERIC
 * endpoint does not).
 */
export async function doSignedOpenAICompatibleStream(
  modelId: string,
  client: GenerativeAiInferenceClient,
  compartmentId: string,
  options: LanguageModelV3CallOptions,
  ociOptions: OCIProviderOptions | undefined,
  warnings: SharedV3Warning[]
): Promise<LanguageModelV3StreamResult> {
  const promptMessages = convertToOCIMessages(options.prompt);
  const body = createOpenAICompatibleRequestBody(
    modelId, promptMessages, options, ociOptions
  );
  const bodyString = JSON.stringify(body);

  const request = await composeRequest({
    baseEndpoint: (client as unknown as { endpoint: string }).endpoint,
    defaultHeaders: {},
    path: '/actions/v1/chat/completions',
    method: 'POST',
    bodyContent: bodyString,
    pathParams: {},
    headerParams: {},
    queryParams: {},
  });
  request.headers.set('CompartmentId', compartmentId);

  const httpClient = (client as unknown as { _httpClient: {
    signer: { signHttpRequest: (r: unknown) => Promise<void> };
  } })._httpClient;
  await httpClient.signer.signHttpRequest(request);

  const headers: Record<string, string> = {};
  for (const [k, v] of request.headers) {
    headers[k] = v;
  }

  const response = await fetch(request.uri, {
    method: 'POST',
    headers,
    body: bodyString,
  });

  if (!response.ok || !response.body) {
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `OCI OpenAI-compatible request failed (${response.status} ${response.statusText})${errorBody ? `: ${errorBody}` : ''}`
    );
  }

  const requestId = response.headers.get('opc-request-id') ?? undefined;

  return {
    request: { body: bodyString },
    response: {
      headers: requestId ? { 'opc-request-id': requestId } : undefined,
    },
    stream: createOpenAICompatibleResponseStream(response, requestId, warnings),
  };
}
