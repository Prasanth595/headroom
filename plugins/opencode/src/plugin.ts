import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import { createHash } from "node:crypto";
import path from "node:path";
import type { Plugin as PluginV2 } from "@opencode/plugin";
import { z } from "zod";

import {
  createHeadroomRetrieveTool,
  getDefaultProxyUrl,
  trimTrailingSlashes,
} from "./retrieve.js";
import type { HeadroomToolPolicyConfig } from "./transport.js";
import {
  acknowledgeNativeToolExecution,
  acknowledgeUnknownNativeToolExecution,
  assertNativeToolExecutionBinding,
  enforceNativeToolExecution,
  installHeadroomTransport,
  refreshHeadroomToolPolicy,
  EXCLUDE_HOSTS_ENV,
  TOOL_POLICY_ENV,
  TOOL_POLICY_PATH_ENV,
  TOOL_POLICY_REFRESH_SECONDS_ENV,
  TOOL_POLICY_TOKEN_ENV,
  TOOL_POLICY_URL_ENV,
  TOOL_POLICY_VALID_UNTIL_ENV,
  BASE_URL_HEADER,
  ORIGINAL_PATH_HEADER,
  PROJECT_HEADER,
  SESSION_TOKEN_HEADER,
  modelBaseRoutesThroughProxy,
  normalizeExcludeHosts,
} from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  excludeHosts?: string[];
  backend?: string;
  debug?: boolean;
  toolPolicy?: HeadroomToolPolicyConfig | string;
  pendingPreflightTtlMs?: number;
  maxPendingPreflights?: number;
  maxRetiredCallIdentities?: number;
  sessionToken?: string;
}

const DEFAULT_PENDING_PREFLIGHT_TTL_MS = 5 * 60 * 1_000;
const DEFAULT_MAX_PENDING_PREFLIGHTS = 1_024;
const DEFAULT_MAX_RETIRED_CALL_IDENTITIES = 65_536;

interface PendingPreflight {
  preflight: NonNullable<Awaited<ReturnType<typeof enforceNativeToolExecution>>>;
  ambiguous: boolean;
  timer: ReturnType<typeof setTimeout>;
}

interface NativeExecutionIdentity {
  sessionID: string;
  callID: string;
  taskID?: string;
}

function positiveInteger(value: number | undefined, fallback: number): number {
  return Number.isSafeInteger(value) && value! > 0 ? value! : fallback;
}

export const HEADROOM_PLUGIN_ID = "headroom";

function normalizeProxyUrl(url: string): string {
  return trimTrailingSlashes(url);
}

function retiredCallFingerprint(key: string): string {
  return createHash("sha256").update(key).digest("hex").slice(0, 32);
}

function nativeCallKey(execution: NativeExecutionIdentity): string {
  return `${execution.sessionID}\0${execution.callID}`;
}

function freezeArguments(value: unknown, seen = new Set<object>()): void {
  if (!value || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  for (const child of Object.values(value as Record<string, unknown>)) {
    freezeArguments(child, seen);
  }
  Object.freeze(value);
}

function toolArguments(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return { value };
}

function createNativeToolLifecycle(
  projectPath: string,
  options: HeadroomOpenCodePluginOptions,
) {
  const pendingPreflights = new Map<string, PendingPreflight>();
  const retiredCallFingerprints = new Set<string>();
  let retiredCallHistoryExhausted = false;
  const pendingPreflightTtlMs = positiveInteger(
    options.pendingPreflightTtlMs,
    DEFAULT_PENDING_PREFLIGHT_TTL_MS,
  );
  const maxPendingPreflights = positiveInteger(
    options.maxPendingPreflights,
    DEFAULT_MAX_PENDING_PREFLIGHTS,
  );
  const maxRetiredCallIdentities = positiveInteger(
    options.maxRetiredCallIdentities,
    DEFAULT_MAX_RETIRED_CALL_IDENTITIES,
  );

  const effectiveCwd = (args: Record<string, unknown>): string => {
    const configured = typeof args.workdir === "string" ? args.workdir : projectPath;
    return path.resolve(projectPath, configured);
  };

  const hasRetiredCallKey = (key: string): boolean =>
    retiredCallHistoryExhausted ||
    retiredCallFingerprints.has(retiredCallFingerprint(key));

  const retireCallKey = (key: string): void => {
    if (retiredCallHistoryExhausted) return;
    const fingerprint = retiredCallFingerprint(key);
    if (retiredCallFingerprints.has(fingerprint)) return;
    if (retiredCallFingerprints.size >= maxRetiredCallIdentities) {
      retiredCallHistoryExhausted = true;
      return;
    }
    retiredCallFingerprints.add(fingerprint);
  };

  const finishUnknown = (
    key: string,
    reason: NonNullable<Parameters<typeof acknowledgeUnknownNativeToolExecution>[1]>,
  ): void => {
    const pending = pendingPreflights.get(key);
    if (!pending) return;
    pendingPreflights.delete(key);
    clearTimeout(pending.timer);
    retireCallKey(key);
    acknowledgeUnknownNativeToolExecution(pending.preflight, reason);
  };

  const rememberPreflight = (
    key: string,
    preflight: NonNullable<Awaited<ReturnType<typeof enforceNativeToolExecution>>>,
  ): void => {
    finishUnknown(key, "call_replaced");
    const ambiguous = hasRetiredCallKey(key);
    while (pendingPreflights.size >= maxPendingPreflights) {
      const oldestKey = pendingPreflights.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      finishUnknown(oldestKey, "capacity_evicted");
    }
    const timer = setTimeout(
      () => finishUnknown(key, "postflight_timeout"),
      pendingPreflightTtlMs,
    );
    timer.unref?.();
    pendingPreflights.set(key, { preflight, ambiguous, timer });
  };

  return {
    async before(
      toolName: string,
      args: Record<string, unknown>,
      execution: NativeExecutionIdentity,
    ): Promise<boolean> {
      const preflight = await enforceNativeToolExecution(
        toolName,
        args,
        effectiveCwd(args),
        execution,
      );
      if (!preflight) return false;
      freezeArguments(args);
      rememberPreflight(nativeCallKey(execution), preflight);
      return true;
    },
    after(
      toolName: string,
      args: Record<string, unknown>,
      execution: NativeExecutionIdentity,
    ): void {
      const key = nativeCallKey(execution);
      const pending = pendingPreflights.get(key);
      if (!pending) return;
      if (pending.ambiguous) {
        finishUnknown(key, "ambiguous_reused_call");
        return;
      }
      try {
        acknowledgeNativeToolExecution(
          pending.preflight,
          toolName,
          args,
          effectiveCwd(args),
          execution,
        );
        pendingPreflights.delete(key);
        clearTimeout(pending.timer);
        retireCallKey(key);
      } catch (error) {
        finishUnknown(key, "postflight_mismatch");
        throw error;
      }
    },
    afterError(
      toolName: string,
      args: Record<string, unknown>,
      execution: NativeExecutionIdentity,
    ): void {
      const key = nativeCallKey(execution);
      const pending = pendingPreflights.get(key);
      if (!pending) return;
      try {
        assertNativeToolExecutionBinding(
          pending.preflight,
          toolName,
          args,
          effectiveCwd(args),
          execution,
        );
        finishUnknown(key, "postflight_error");
      } catch (error) {
        finishUnknown(key, "postflight_mismatch");
        throw error;
      }
    },
    dispose(): void {
      for (const key of [...pendingPreflights.keys()]) {
        finishUnknown(key, "plugin_disposed");
      }
    },
  };
}

function resolveProxyUrl(options?: HeadroomOpenCodePluginOptions): string {
  return normalizeProxyUrl(
    options?.proxyUrl ??
      process.env.HEADROOM_PROXY_URL ??
      process.env.HEADROOM_BASE_URL ??
      getDefaultProxyUrl(),
  );
}

// Read by workspace_registry.resolve_registered_cwd() on the proxy side --
// mirrors wrap.py's _apply_session_token_header_env for claude.
function resolveSessionToken(options?: HeadroomOpenCodePluginOptions): string | undefined {
  return options?.sessionToken ?? process.env.HEADROOM_OPENCODE_SESSION_TOKEN;
}


// The exclusion list governs both the transport patch and the 2.x model
// rewrite, so resolve it once: option over environment variable, normalized
// the same way the transport does.
function resolveExcludeHosts(options?: HeadroomOpenCodePluginOptions): string[] {
  return normalizeExcludeHosts(options?.excludeHosts ?? process.env[EXCLUDE_HOSTS_ENV] ?? "");
}

function headroomShellEnv(
  proxyUrl: string,
  project: string,
  options: HeadroomOpenCodePluginOptions,
): Record<string, string> {
  return {
    HEADROOM_ACTIVE: "1",
    HEADROOM_PROXY_URL: proxyUrl,
    HEADROOM_PROJECT: project,
    ...(options.backend ? { HEADROOM_BACKEND: options.backend } : {}),
  };
}

function applyHeadroomShellEnv(
  env: Record<string, string | undefined>,
  proxyUrl: string,
  project: string,
  options: HeadroomOpenCodePluginOptions,
): void {
  delete env[TOOL_POLICY_TOKEN_ENV];
  delete env[TOOL_POLICY_URL_ENV];
  Object.assign(env, headroomShellEnv(proxyUrl, project, options));
  for (const name of [
    EXCLUDE_HOSTS_ENV,
    TOOL_POLICY_ENV,
    TOOL_POLICY_PATH_ENV,
    TOOL_POLICY_REFRESH_SECONDS_ENV,
    TOOL_POLICY_VALID_UNTIL_ENV,
  ]) {
    if (process.env[name]) {
      env[name] = process.env[name];
    } else {
      delete env[name];
    }
  }
}

// The endpoint an OpenCode 2.x package calls under its baseURL, for packages
// on the OpenAI chat-completions or responses wire format; undefined otherwise.
// `.../messages` (Anthropic wire) and other protocols are left alone.
function openAiWireSuffix(pkg: string | undefined): string | undefined {
  if (!pkg?.startsWith("@opencode/ai/providers/")) return undefined;
  if (pkg === "@opencode/ai/providers/openai-compatible" || pkg.endsWith("/chat")) {
    return "/chat/completions";
  }
  if (pkg.endsWith("-responses") || pkg.endsWith("/responses")) return "/responses";
  return undefined;
}

// OpenCode 2.x sends model traffic through Effect's FetchHttpClient, which
// captures `globalThis.fetch` on the process's first HTTP request, before any
// plugin runs, so the transport's fetch patch never sees it. Point each
// remote OpenAI-wire model at the proxy instead, with the same routing headers
// the transport sends: the upstream origin, and the real path the proxy
// appends to it.
type ModelEditor = Parameters<Parameters<PluginV2.Context["model"]["transform"]>[0]>[0];

function routeModelsThroughProxy(
  models: ModelEditor,
  proxyUrl: string,
  project: string,
  excludeHosts: string[],
  sessionToken?: string,
): void {
  for (const model of models.list()) {
    // DeepMutable turns the branded ID strings into object types.
    const providerID = String(model.providerID);
    const modelID = String(model.id);
    const provider = models.provider.get(providerID)?.provider;
    const suffix = openAiWireSuffix(model.package ?? provider?.package);
    const baseURL = model.settings?.baseURL ?? provider?.settings?.baseURL;
    if (!suffix || typeof baseURL !== "string") continue;
    if (!modelBaseRoutesThroughProxy(baseURL, proxyUrl, excludeHosts)) continue;
    const upstream = new URL(baseURL);
    models.update(providerID, modelID, (draft) => {
      draft.settings = { ...draft.settings, baseURL: `${proxyUrl}/v1` };
      draft.headers = {
        ...draft.headers,
        [BASE_URL_HEADER]: upstream.origin,
        [ORIGINAL_PATH_HEADER]: `${trimTrailingSlashes(upstream.pathname)}${suffix}`,
        [PROJECT_HEADER]: project,
        ...(sessionToken ? { [SESSION_TOKEN_HEADER]: sessionToken } : {}),
      };
    });
  }
}

// OpenCode 1.x: a factory returning a hooks object.
export const HeadroomPlugin: Plugin = async (input, options = {}) => {
  const pluginOptions = options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const projectPath = input.worktree || input.directory;
  const project =
    pluginOptions.project ??
    (input.project as { id?: string } | undefined)?.id ??
    input.directory;
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    policyProject: projectPath,
    excludeHosts: pluginOptions.excludeHosts,
    debug: pluginOptions.debug,
    toolPolicy: pluginOptions.toolPolicy,
    sessionToken: resolveSessionToken(pluginOptions),
  });
  await refreshHeadroomToolPolicy();
  const nativeLifecycle = createNativeToolLifecycle(projectPath, pluginOptions);

  return {
    dispose: async () => {
      nativeLifecycle.dispose();
      uninstallTransport();
    },
    tool: {
      headroom_retrieve: tool({
        description: retrieveTool.description,
        args: {
          hash: z
            .string()
            .regex(/^[a-f0-9]{24}$/i, "Expected 24-character hex hash"),
        },
        async execute(args) {
          return retrieveTool.execute(args);
        },
      }),
    },
    "shell.env": async (_input, output) => {
      applyHeadroomShellEnv(output.env, proxyUrl, project, pluginOptions);
    },
    "tool.execute.before": async (hookInput, output) => {
      const bound = await nativeLifecycle.before(
        hookInput.tool,
        output.args,
        {
          sessionID: hookInput.sessionID,
          callID: hookInput.callID,
        },
      );
      if (bound) {
        Object.freeze(output);
      }
    },
    "tool.execute.after": async (hookInput) => {
      nativeLifecycle.after(hookInput.tool, hookInput.args, {
        sessionID: hookInput.sessionID,
        callID: hookInput.callID,
      });
    },
  };
};

// OpenCode 2.x: setup registers hooks imperatively and returns its cleanup.
// OpenCode reruns setup (after cleanup) whenever it hot-reloads the plugin;
// the transport is refcounted, so reinstalling is safe.
export const headroomSetup: PluginV2.Plugin["setup"] = async (ctx) => {
  const pluginOptions = ctx.options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const project =
    pluginOptions.project ?? ctx.location.project.id ?? ctx.location.directory;
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const excludeHosts = resolveExcludeHosts(pluginOptions);
  const sessionToken = resolveSessionToken(pluginOptions);
  const nativeLifecycle = createNativeToolLifecycle(ctx.location.directory, pluginOptions);
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    policyProject: ctx.location.directory,
    excludeHosts,
    debug: pluginOptions.debug,
    toolPolicy: pluginOptions.toolPolicy,
    sessionToken,
  });

  try {
    await refreshHeadroomToolPolicy();
    await ctx.model.transform((models) => {
      routeModelsThroughProxy(models, proxyUrl, project, excludeHosts, sessionToken);
    });
    await ctx.tool.transform((editor) => {
      editor.add({
        name: retrieveTool.name,
        description: retrieveTool.description,
        input: retrieveTool.parameters,
        // Offer the tool directly, like OpenCode's built-ins, rather than only
        // through the code-mode `execute` tool.
        options: { codemode: false },
        async execute(args) {
          return { content: await retrieveTool.execute(args as { hash: string }) };
        },
      });
    });
    await ctx.shell.hook("create.before", (shell) => {
      applyHeadroomShellEnv(shell.env, proxyUrl, project, pluginOptions);
    });
    await ctx.tool.hook("execute.before", async (input) => {
      const execution = {
        sessionID: String(input.sessionID),
        callID: String(input.id),
        taskID: String(input.messageID),
      };
      const bound = await nativeLifecycle.before(
        input.tool,
        toolArguments(input.input),
        execution,
      );
      if (bound) Object.freeze(input);
    });
    await ctx.tool.hook("execute.after", (input) => {
      const execution = {
        sessionID: String(input.sessionID),
        callID: String(input.id),
        taskID: String(input.messageID),
      };
      if (input.status === "error") {
        nativeLifecycle.afterError(
          input.tool,
          toolArguments(input.input),
          execution,
        );
        return;
      }
      nativeLifecycle.after(input.tool, toolArguments(input.input), execution);
    });
  } catch (error) {
    nativeLifecycle.dispose();
    uninstallTransport();
    throw error;
  }

  return () => {
    nativeLifecycle.dispose();
    uninstallTransport();
  };
};

// One default export for both loaders: OpenCode 1.x reads `server`, 2.x reads
// `setup`, and each ignores the other's key.
export const HeadroomOpenCodePlugin = {
  id: HEADROOM_PLUGIN_ID,
  server: HeadroomPlugin,
  setup: headroomSetup,
};

export default HeadroomOpenCodePlugin;
