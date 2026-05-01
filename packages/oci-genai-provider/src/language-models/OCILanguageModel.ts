import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
  LanguageModelV3FinishReason,
  LanguageModelV3Usage,
  LanguageModelV3Content,
} from '@ai-sdk/provider';
import { NoSuchModelError } from '@ai-sdk/provider';
import { GenerativeAiInferenceClient, models as OCIModel } from 'oci-generativeaiinference';
import { Region } from 'oci-common';
import type { OCIConfig, RequestOptions } from '../types';
import { isValidModelId, getModelMetadata, supportsReasoning, supportsVision } from './registry';
import { convertToOCIMessages } from './converters/messages';
import type { OCIMessage } from './converters/messages';
import { convertToCohereFormat } from './converters/cohere-messages';
import { convertToOCITools, convertToOCIToolChoice, supportsToolCalling } from './converters/tools';
import { parseSSEStream } from '../shared/streaming/sse-parser';
import { createAuthProvider, getRegion, getCompartmentId, isAPIKeyAuth } from '../auth/index.js';
import { handleOCIError } from '../shared/errors/index.js';
import { withRetry, withTimeout, isRetryableError } from '../shared/utils/index.js';
import {
  getOCIProviderOptions,
  resolveCompartmentId,
  resolveEndpoint,
  resolveServingMode,
} from '../shared/provider-options';
import { resolveRequestOptions } from '../shared/request-options';
import {
  type OCIApiFormat,
  toOCIReasoningEffort,
  createThinkingConfig,
} from '../shared/oci-sdk-types';
import {
  applyInputGuardrails,
  getEffectiveGuardrailsWarning,
  getInputGuardrailText,
  toJSONGuardrailsMetadata,
} from '../shared/guardrails';
import { doOpenAICompatibleStream, doSignedOpenAICompatibleStream } from './openai-compatible';

type ChatRequest =
  | OCIModel.GenericChatRequest
  | OCIModel.CohereChatRequest
  | OCIModel.CohereChatRequestV2;

const COHERE_REQUEST_DEBUG_ENV = 'OCI_GENAI_DEBUG_COHERE_REQUESTS';

function serializeChatDetails(chatDetails: OCIModel.ChatDetails): string {
  const serializer = (
    OCIModel as unknown as
      | {
          ChatDetails?: { getJsonObj?: (value: OCIModel.ChatDetails) => object };
        }
      | undefined
  )?.ChatDetails;

  if (serializer?.getJsonObj) {
    return JSON.stringify(serializer.getJsonObj(chatDetails));
  }

  return JSON.stringify(chatDetails);
}

function shouldLogCohereToolRequest(apiFormat: OCIApiFormat, toolCount: number): boolean {
  return (
    apiFormat === 'COHERE' &&
    toolCount > 0 &&
    typeof process !== 'undefined' &&
    process.env[COHERE_REQUEST_DEBUG_ENV] === '1'
  );
}

export class OCILanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3';
  readonly provider = 'oci-genai';
  readonly defaultObjectGenerationMode = 'tool';
  readonly supportedUrls: Record<string, RegExp[]> = {};
  private _client?: GenerativeAiInferenceClient;

  constructor(
    public readonly modelId: string,
    private readonly config: OCIConfig
  ) {
    if (!isValidModelId(modelId)) {
      throw new NoSuchModelError({
        modelId,
        modelType: 'languageModel',
      });
    }
  }

  private async getClient(endpointOverride?: string): Promise<GenerativeAiInferenceClient> {
    const resolvedEndpoint = resolveEndpoint(this.config.endpoint, endpointOverride);

    if (!this._client || (endpointOverride && endpointOverride !== this.config.endpoint)) {
      try {
        const authProvider = await createAuthProvider(this.config);
        const regionId = getRegion(this.config);

        const client = new GenerativeAiInferenceClient({
          authenticationDetailsProvider: authProvider,
        });

        client.region = Region.fromRegionId(regionId);

        if (resolvedEndpoint) {
          client.endpoint = resolvedEndpoint;
        }

        if (!endpointOverride || endpointOverride === this.config.endpoint) {
          this._client = client;
        }

        return client;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Unknown error during client initialization';
        throw new Error(
          `Failed to initialize OCI client: ${message}. ` +
            `Check your OCI configuration (config file, credentials, region).`
        );
      }
    }
    return this._client;
  }

  private getRequestOptions(perRequestOptions?: RequestOptions): Required<RequestOptions> {
    return resolveRequestOptions(this.config.requestOptions, perRequestOptions);
  }

  private getApiFormat(hasImages: boolean = false): OCIApiFormat {
    const metadata = getModelMetadata(this.modelId);
    if (metadata?.family === 'cohere') {
      if (
        this.modelId === 'cohere.command-a-reasoning-08-2025' ||
        this.modelId === 'cohere.command-a-vision-07-2025' ||
        this.modelId === 'cohere.command-a-reasoning' ||
        this.modelId === 'cohere.command-a-vision'
      ) {
        return 'COHEREV2';
      }
      // Upgrade to COHEREV2 for vision-capable models when images are present
      // COHERE V1 format silently drops images, V2 supports them properly
      if (hasImages && metadata.capabilities.vision) {
        return 'COHEREV2';
      }
      return 'COHERE';
    }
    return 'GENERIC';
  }

  private async executeWithResilience<T>(
    operation: () => Promise<T>,
    operationName: string,
    requestOptions?: RequestOptions
  ): Promise<T> {
    const options = this.getRequestOptions(requestOptions);
    const withTimeoutOperation = (): Promise<T> =>
      withTimeout(operation(), options.timeoutMs, operationName);

    if (options.retry.enabled) {
      return withRetry(withTimeoutOperation, {
        maxRetries: options.retry.maxRetries,
        baseDelayMs: options.retry.baseDelayMs,
        maxDelayMs: options.retry.maxDelayMs,
        isRetryable: isRetryableError,
      });
    }

    return withTimeoutOperation();
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const { stream, response, request } = await this.doStream(options);
    const reader = stream.getReader();
    const warnings: SharedV3Warning[] = [];
    const content: LanguageModelV3Content[] = [];
    let usage: LanguageModelV3Usage = {
      inputTokens: {
        total: 0,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: 0,
        text: undefined,
        reasoning: undefined,
      },
    };
    let finishReason: LanguageModelV3FinishReason = {
      unified: 'other',
      raw: 'initializing',
    };
    let providerMetadata: Record<string, any> | undefined;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        switch (value.type) {
          case 'stream-start':
            if (value.warnings) {
              warnings.push(...value.warnings);
            }
            break;
          case 'text-delta': {
            const last = content[content.length - 1];
            if (last?.type === 'text') {
              last.text += value.delta;
            } else {
              content.push({ type: 'text', text: value.delta });
            }
            break;
          }
          case 'reasoning-delta': {
            const last = content[content.length - 1];
            if (last?.type === 'reasoning') {
              last.text += value.delta;
            } else {
              content.push({ type: 'reasoning', text: value.delta });
            }
            break;
          }
          case 'tool-call':
            content.push({
              type: 'tool-call',
              toolCallId: value.toolCallId,
              toolName: value.toolName,
              input: value.input,
            });
            break;
          case 'finish':
            finishReason = value.finishReason;
            usage = value.usage;
            providerMetadata = value.providerMetadata;
            break;
          case 'error':
            throw value.error;
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {
      content,
      usage,
      finishReason,
      request,
      response,
      warnings,
      providerMetadata,
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const messages: OCIMessage[] = convertToOCIMessages(options.prompt);
    const ociOptions = getOCIProviderOptions(options.providerOptions);

    // Check for images early - needed for API format selection
    const hasImages = messages.some((m) => m.content.some((c) => c.type === 'IMAGE'));
    const apiFormat = this.getApiFormat(hasImages);
    const warnings: SharedV3Warning[] = [];

    if (options.responseFormat?.type === 'json') {
      warnings.push({
        type: 'unsupported',
        feature: 'responseFormat.json',
        details: 'OCI response format JSON is not supported in this provider.',
      });
    }

    const modelSupportsTools = supportsToolCalling(this.modelId);
    const modelSupportsReasoning = supportsReasoning(this.modelId);
    const hasTools = options.tools && options.tools.length > 0;
    const functionTools = hasTools ? options.tools!.filter((t) => t.type === 'function') : [];

    // Add tools warning for unsupported models (matching doGenerate)
    if (hasTools && !modelSupportsTools) {
      warnings.push({
        type: 'unsupported',
        feature: 'tools',
        details: `Model ${this.modelId} does not support tool calling. Supported: Llama 3.1+, Cohere Command R/R+, Grok, Gemini.`,
      });
    }

    // Add reasoning model validation (matching doGenerate)
    if (ociOptions?.reasoningEffort && !modelSupportsReasoning) {
      warnings.push({
        type: 'unsupported',
        feature: 'reasoningEffort',
        details: `Model ${this.modelId} does not support reasoning. Use a reasoning model like xai.grok-4-1-fast-reasoning or cohere.command-a-reasoning-08-2025.`,
      });
    }

    if (ociOptions?.thinking && !modelSupportsReasoning) {
      warnings.push({
        type: 'unsupported',
        feature: 'thinking',
        details: `Model ${this.modelId} does not support thinking/reasoning. Use a reasoning model like cohere.command-a-reasoning-08-2025.`,
      });
    }

    // Vision support validation
    const modelSupportsVision = supportsVision(this.modelId);

    if (hasImages && !modelSupportsVision) {
      warnings.push({
        type: 'unsupported',
        feature: 'vision',
        details: `Model ${this.modelId} does not support image input. Use a vision model like meta.llama-3.2-90b-vision-instruct, google.gemini-2.5-flash, or cohere.command-a-vision.`,
      });
    }

    // Cohere V1 format silently drops images - warn the user
    // Note: vision-capable Cohere models are automatically upgraded to COHEREV2 format
    if (hasImages && apiFormat === 'COHERE') {
      warnings.push({
        type: 'unsupported',
        feature: 'vision',
        details: `Cohere V1 API format does not support images. Images in your prompt will be ignored. Use a vision-capable model like cohere.command-a-vision which uses the Cohere V2 format.`,
      });
    }

    const guardrailsWarning = getEffectiveGuardrailsWarning(this.config, ociOptions?.guardrails);
    if (guardrailsWarning) {
      warnings.push({
        type: 'unsupported',
        feature: 'guardrails',
        details: guardrailsWarning,
      });
    }

    if (isAPIKeyAuth(this.config)) {
      return doOpenAICompatibleStream(this.modelId, this.config, options, ociOptions, warnings);
    }

    // The native GENERIC endpoint does not populate tool-call arguments
    // in streaming responses. Route GENERIC models through the OpenAI-
    // compatible endpoint using OCI request signing so tool calling works
    // with config_file auth.
    if (apiFormat === 'GENERIC') {
      const client = await this.getClient(ociOptions?.endpoint);
      const compartmentId = resolveCompartmentId(
        getCompartmentId(this.config),
        ociOptions?.compartmentId
      );
      return doSignedOpenAICompatibleStream(
        this.modelId, client, compartmentId, options, ociOptions, warnings
      );
    }

    const client = await this.getClient(ociOptions?.endpoint);
    const compartmentId = resolveCompartmentId(
      getCompartmentId(this.config),
      ociOptions?.compartmentId
    );
    const guardrailsMetadata =
      ociOptions?.guardrails?.input && !guardrailsWarning
        ? await applyInputGuardrails(
            client,
            compartmentId,
            getInputGuardrailText(
              messages.flatMap((message) =>
                message.content
                  .filter((content) => content.type === 'TEXT')
                  .map((content) => content.text)
              )
            ),
            ociOptions.guardrails
          )
        : undefined;

    try {
      const commonParams = {
        maxTokens: options.maxOutputTokens ? Math.min(options.maxOutputTokens, 4000) : undefined,
        temperature: options.temperature,
        topP: options.topP,
        topK: options.topK ? Math.min(options.topK, 40) : undefined,
        isStream: true,
      };

      const toolParams =
        modelSupportsTools && functionTools.length > 0
          ? {
              tools: convertToOCITools(functionTools, apiFormat),
              ...(options.toolChoice
                ? { toolChoice: convertToOCIToolChoice(options.toolChoice) }
                : {}),
            }
          : {};

      let chatRequest: ChatRequest;
      if (apiFormat === 'COHEREV2') {
        chatRequest = {
          apiFormat,
          messages: messages.map((m) => {
            const content: OCIModel.CohereContentV2[] = m.content.map((c) => {
              if (c.type === 'IMAGE') {
                return {
                  type: 'IMAGE_URL',
                  imageUrl: c.imageUrl as OCIModel.CohereImageUrlV2,
                } as OCIModel.CohereImageContentV2;
              }
              return { type: 'TEXT', text: c.text ?? '' } as OCIModel.CohereTextContentV2;
            });
            return {
              role: m.role,
              content,
            } as OCIModel.CohereMessageV2;
          }),
          ...commonParams,
          ...toolParams,
        } as OCIModel.CohereChatRequestV2;
      } else if (apiFormat === 'COHERE') {
        const cohereFormat = convertToCohereFormat(messages);
        const hasToolResults = (cohereFormat.toolResults?.length ?? 0) > 0;
        chatRequest = {
          apiFormat,
          ...cohereFormat,
          ...commonParams,
          ...toolParams,
          // Cohere requires isForceSingleStep=true when tool results are present
          // This ensures multi-step tool use works correctly
          ...(hasToolResults ? { isForceSingleStep: true } : {}),
        } as OCIModel.CohereChatRequest;
      } else {
        chatRequest = {
          apiFormat,
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content as OCIModel.ChatContent[],
            toolCalls: m.toolCalls,
            toolCallId: m.toolCallId,
          })) as OCIModel.Message[],
          ...commonParams,
          ...toolParams,
          stop: options.stopSequences,
        } as OCIModel.GenericChatRequest;
      }

      if (ociOptions?.reasoningEffort && apiFormat === 'GENERIC') {
        const genericReq = chatRequest as OCIModel.GenericChatRequest;
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- OCI SDK type mismatch
        genericReq.reasoningEffort = toOCIReasoningEffort(ociOptions.reasoningEffort) as any;
      }

      if (ociOptions?.thinking && (apiFormat === 'COHEREV2' || apiFormat === 'COHERE')) {
        const cohereReq = chatRequest as OCIModel.CohereChatRequestV2;
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any -- OCI SDK type mismatch
        cohereReq.thinking = createThinkingConfig(true, ociOptions.tokenBudget) as any;
      }

      if (options.seed !== undefined) chatRequest.seed = options.seed;

      const chatDetails = {
        compartmentId,
        servingMode: resolveServingMode(
          this.modelId,
          this.config.servingMode,
          ociOptions?.servingMode
        ),
        chatRequest,
      } as OCIModel.ChatDetails;
      const serializedRequestBody = serializeChatDetails(chatDetails);

      if (shouldLogCohereToolRequest(apiFormat, functionTools.length)) {
        console.error(
          `[oci-genai-provider] Cohere tool request payload (${COHERE_REQUEST_DEBUG_ENV}=1): ${serializedRequestBody}`
        );
      }

      const response = (await this.executeWithResilience<unknown>(
        () =>
          client.chat({
            chatDetails,
          }),
        'OCI chat stream',
        ociOptions?.requestOptions
      )) as {
        body?: ReadableStream<Uint8Array>;
        headers?: { entries(): IterableIterator<[string, string]> };
      };

      // response can be ReadableStream (Browser/new SDK) or Response object (Node/Fetch polyfill)
      // Some mocks might just return the stream directly.
      const streamInput = response.body ?? (response as unknown as ReadableStream<Uint8Array>);

      if (!streamInput) {
        throw new Error('No stream received from OCI.');
      }

      const stream = parseSSEStream(streamInput, {
        includeRawChunks: options.includeRawChunks,
      });

      const headers = response.headers ? Object.fromEntries(response.headers.entries()) : undefined;

      return {
        stream: new ReadableStream<LanguageModelV3StreamPart>({
          async start(controller): Promise<void> {
            controller.enqueue({ type: 'stream-start', warnings });

            let hasText = false;
            let hasReasoning = false;
            const textId = `text-${Date.now()}`;
            const reasoningId = `reasoning-${Date.now()}`;

            try {
              for await (const part of stream) {
                switch (part.type) {
                  case 'text-delta':
                    if (!hasText) {
                      controller.enqueue({ type: 'text-start', id: textId });
                      hasText = true;
                    }
                    controller.enqueue({
                      type: 'text-delta',
                      delta: part.textDelta,
                      id: textId,
                    });
                    break;
                  case 'reasoning-delta':
                    if (!hasReasoning) {
                      controller.enqueue({ type: 'reasoning-start', id: reasoningId });
                      hasReasoning = true;
                    }
                    controller.enqueue({
                      type: 'reasoning-delta',
                      delta: part.reasoningDelta,
                      id: reasoningId,
                    });
                    break;
                  case 'tool-call':
                    controller.enqueue({
                      type: 'tool-call',
                      toolCallId: part.toolCallId,
                      toolName: part.toolName,
                      input: part.input,
                    });
                    break;
                  case 'finish':
                    if (hasText) {
                      controller.enqueue({ type: 'text-end', id: textId });
                    }
                    if (hasReasoning) {
                      controller.enqueue({ type: 'reasoning-end', id: reasoningId });
                    }
                    controller.enqueue({
                      type: 'finish',
                      finishReason: part.finishReason,
                      usage: {
                        inputTokens: {
                          total: part.usage.promptTokens,
                          noCache: undefined,
                          cacheRead: undefined,
                          cacheWrite: undefined,
                        },
                        outputTokens: {
                          total: part.usage.completionTokens,
                          text:
                            part.usage.reasoningTokens !== undefined
                              ? part.usage.completionTokens - part.usage.reasoningTokens
                              : part.usage.completionTokens,
                          reasoning: part.usage.reasoningTokens,
                        },
                      },
                      providerMetadata: {
                        oci: {
                          requestId: headers?.['opc-request-id'],
                          ...(guardrailsMetadata
                            ? { guardrails: toJSONGuardrailsMetadata(guardrailsMetadata) }
                            : {}),
                        },
                      },
                    });
                    break;
                  case 'raw':
                    controller.enqueue({ type: 'raw', rawValue: part.rawValue });
                    break;
                }
              }
            } catch (error) {
              // Log stream errors for debugging and monitoring
              console.error('[OCI Stream] Error during stream processing:', error);
              controller.enqueue({ type: 'error', error });
            } finally {
              controller.close();
            }
          },
        }),
        request: { body: serializedRequestBody },
        response: {
          headers,
        },
      };
    } catch (error) {
      throw handleOCIError(error);
    }
  }
}
