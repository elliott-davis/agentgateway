### LLM Switchyard virtual-model routing

This example shows how to route one public virtual model across an equivalence class of models using [NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) (`switchyard-libsy` 0.3.0), while keeping the agent's prompt and tool contract intact.

Agent harnesses are often tuned (prompts, tool schemas) for a specific family of models, and long agent sessions build up a provider-side prefix/KV cache. Swapping in an arbitrary cheaper model can break that contract, and flipping models every turn throws the cache away. Switchyard routing addresses both:

- the client keeps sending the same public model name (`coding-agent`)
- every candidate model belongs to one **equivalence class** the operator vouches for
- only the `model` of the request is rewritten; the client's prompt, tools, and messages are forwarded untouched
- a session stays on its warm model unless there is a decisive reason to switch

The important distinction is:

- `coding-agent` is the public virtual model name the caller sends
- `claude-3-7-sonnet` and `claude-3-5-sonnet` are internal concrete models
- each target in the class has a **tier**: `capable` or `efficient`
- the **first target is the anchor**: it is used for strict/bypass requests and as the fallback

```yaml
virtualModels:
- name: coding-agent
  routing:
    switchyard:
      strategy: stageRouter
      equivalenceClass: claude
      targets:
      - model: claude-3-7-sonnet   # anchor
        tier: capable
        contextWindow: 200000
      - model: claude-3-5-sonnet
        tier: efficient
        contextWindow: 200000
      sessionAffinity:
        header: x-agent-session-id
        idleTimeout: 1h
        maxSessions: 10000
      fallbackBehavior: fallbackToRequested
      stageRouter:
        mode: efficientFirst
        confidenceThreshold: 0.6
        capableHoldTurns: 2
```

## Request flow

For each request to a Switchyard virtual model, in order:

1. **Bypass.** `x-routing-mode: strict` or `x-allow-dynamic-routing: false` routes to the anchor with decision `bypass`. No evaluation runs and no session pin is recorded.
2. **Capability floor.** `x-routing-allowed-models` and `x-routing-min-context-window` narrow the candidate targets. If nothing is left the request fails with `400 routing_constraints_unsatisfiable`.
3. **StageRouter.** The in-process Switchyard StageRouter heuristic is evaluated on every turn (so its per-session state sees every turn) with a 500ms timeout. It inspects the conversation, for example tool results that show consecutive failures or error text. Only its model choice is used; any prompt rewrites it produces are discarded.
4. **Session affinity.** If the request carries a session id, the session stays pinned to its previous model (preserving the prefix/KV cache). It only switches when StageRouter reports a decisive signal (`override`, `dimensions`, or `capable_hold`): the decision is then `escalate` (to a capable model) or `de-escalate` (to an efficient one). Otherwise the decision is `session_affinity`.
5. **Equivalence-class guard.** A recommendation outside the (narrowed) class is handled per `fallbackBehavior`. An evaluation error or timeout routes to the anchor with decision `evaluation_error`.

If only one tier remains after the capability floor, StageRouter is skipped and the remaining target is used with decision `constrained`.

## Client request headers

| Header | Meaning |
| --- | --- |
| `x-routing-mode: strict` | Bypass dynamic routing and use the anchor. |
| `x-allow-dynamic-routing: false` | Same as `strict`. |
| `x-routing-allowed-models` | Comma-separated list of target model names the client allows. Brackets and quotes are tolerated, for example `["claude-3-7-sonnet"]`. |
| `x-routing-min-context-window` | Minimum context window in tokens. Targets without a configured `contextWindow` never satisfy it. A non-integer value fails with `400 invalid_routing_header`. |
| `x-agent-session-id` | Session identifier for session affinity. The header name is configurable via `sessionAffinity.header`. Empty values and values longer than 256 bytes are ignored. |

## Response headers

| Header | Meaning |
| --- | --- |
| `x-agentgateway-routed-model` | The concrete model selected (omitted when the request was rejected). |
| `x-agentgateway-original-model` | The virtual model the client requested. |
| `x-agentgateway-routing-decision` | Why that model was selected. |

Decision values:

| Value | Meaning |
| --- | --- |
| `bypass` | A strict/bypass header selected the anchor. |
| `session_affinity` | The session stayed on its pinned model. |
| `escalate` | A pinned session moved to a capable model on a decisive signal. |
| `de-escalate` | A pinned session moved to an efficient model on a decisive signal. |
| `constrained` | The capability floor left a single tier, so StageRouter was skipped. |
| `fallback_to_requested` | The recommendation was out of class and the anchor was used. |
| `evaluation_error` | StageRouter failed or timed out and the anchor was used. |
| `rejected_out_of_class` | Recommendation was out of class with `rejectIfOutOfClass` (the request fails with `503 model_routing_out_of_class`; the headers are still emitted). |
| StageRouter source, e.g. `override`, `dimensions`, `capable_hold`, `ambiguous`, `fall_open` | For a session's first request (or an unpinned request), the decision is the StageRouter source that produced the pick. `override` is a hard signal such as repeated tool failures, `dimensions` is a confident score, `capable_hold` keeps the capable tier after an escalation, and `fall_open` means nothing resolved the turn and the picker's default tier (`mode`) was used. |

## Telemetry

Traces and logs include these attributes:

- `llm.routed_model`
- `llm.original_model`
- `llm.routing_strategy` (`stage_router`)
- `llm.session_affinity_hit`
- `llm.routing_decision`

## Session affinity

- The session id comes from `sessionAffinity.header` (default `x-agent-session-id`). Without a header, or with `sessionAffinity` unset, there is no pinning.
- Sessions idle longer than `idleTimeout` (default `1h`) are treated as new. At most `maxSessions` (default `10000`) sessions are tracked.
- Pins are kept in gateway memory. They reset when that virtual model's configuration changes, and they are not shared between gateway replicas.
- A pinned model that is no longer in the narrowed class (for example excluded by `x-routing-allowed-models`) is ignored for that request.

## Fallback behavior

- `fallbackToRequested` (default): an out-of-class recommendation routes to the anchor (decision `fallback_to_requested`).
- `rejectIfOutOfClass`: the request fails with `503 model_routing_out_of_class`.
- Evaluation errors and timeouts always route to the anchor with `evaluation_error`.

## Limitations

- Only the StageRouter strategy is supported.
- Model-as-judge routing (Switchyard's `CallModel` step) has an internal async `JudgeClient` seam but no gateway-backed judge yet, so it is not configurable. Such calls fail and are treated as evaluation errors.
- Only OpenAI chat completions and Anthropic messages bodies feed routing signals. Other APIs fall open to the default tier.

## Run standalone

```shell
cargo run -- -f examples/llm-switchyard-routing/standalone/config.yaml
```

Replace the `ANTHROPIC_API_KEY` environment variable (or the `apiKey` values in `config.yaml`) before sending requests to a real provider. In standalone config the enum values are camelCase (`capable`, `stageRouter`, `rejectIfOutOfClass`).

## Run on Kubernetes

Install agentgateway and create the provider secret:

```shell
kubectl create secret generic anthropic-secret \
  -n agentgateway-system \
  --from-literal=Authorization="$ANTHROPIC_API_KEY"

kubectl apply -f examples/llm-switchyard-routing/k8s/agentgateway-routing.yaml
kubectl port-forward -n agentgateway-system deploy/agentgateway-proxy 8080:80
```

The `AgentgatewayModel` resources use the Kubernetes convention of PascalCase enum values (`Capable`, `Efficient`, `StageRouter`, `RejectIfOutOfClass`), while standalone config uses camelCase. The CRD requires at least two targets, at least one `Capable` and one `Efficient`, and expresses the confidence threshold as an integer percentage (`confidenceThresholdPercent: 60` is `0.6`). Targets reference concrete models with `modelRef`, and internal models are never selectable directly by clients. Check `kubectl get agentgatewaymodel coding-agent -o yaml` for the `ResolvedRefs` condition if a target cannot be resolved.

The examples below use `http://localhost:4000` (standalone). For Kubernetes use `http://localhost:8080`.

## Examples

First turn of a session. With `efficientFirst` and no strong signal, StageRouter has nothing decisive and uses the default tier:

```shell
curl -si http://localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-agent-session-id: demo-1' \
  -d '{"model":"coding-agent","max_tokens":256,"messages":[{"role":"user","content":"say hi"}]}'
```

```text
x-agentgateway-routed-model: claude-3-5-sonnet
x-agentgateway-original-model: coding-agent
x-agentgateway-routing-decision: fall_open
```

Second turn with the same session id. Without a decisive signal the session stays on its warm model, even if the prompt looks very different:

```shell
curl -si http://localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-agent-session-id: demo-1' \
  -d '{"model":"coding-agent","max_tokens":256,"messages":[{"role":"user","content":"say hi"},{"role":"assistant","content":"hi"},{"role":"user","content":"now summarize this long document ..."}]}'
```

```text
x-agentgateway-routed-model: claude-3-5-sonnet
x-agentgateway-routing-decision: session_affinity
```

Strict bypass uses the anchor and skips evaluation:

```shell
curl -si http://localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-routing-mode: strict' \
  -d '{"model":"coding-agent","max_tokens":256,"messages":[{"role":"user","content":"say hi"}]}'
```

```text
x-agentgateway-routed-model: claude-3-7-sonnet
x-agentgateway-routing-decision: bypass
```

Restrict the candidates. Only one tier remains, so StageRouter is skipped:

```shell
curl -si http://localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-routing-allowed-models: claude-3-5-sonnet' \
  -d '{"model":"coding-agent","max_tokens":256,"messages":[{"role":"user","content":"say hi"}]}'
```

```text
x-agentgateway-routed-model: claude-3-5-sonnet
x-agentgateway-routing-decision: constrained
```

A constraint no target can satisfy fails with `400 routing_constraints_unsatisfiable`, for example `-H 'x-routing-min-context-window: 1000000'`.

Escalation after repeated tool failures. This turn carries three failed tool results, which StageRouter treats as a hard signal. For a session pinned to the efficient model the decision is `escalate`; for a session with no pin it is `override`. Both route to the capable model:

```shell
curl -si http://localhost:4000/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'x-agent-session-id: demo-1' \
  -d '{
    "model": "coding-agent",
    "max_tokens": 256,
    "system": "You are a coding agent",
    "tools": [{"name":"Bash","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],
    "messages": [
      {"role":"user","content":"fix the build"},
      {"role":"assistant","content":[{"type":"tool_use","id":"tu_0","name":"Bash","input":{"command":"cargo test"}}]},
      {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_0","is_error":true,"content":"fatal runtime error: out of memory"}]},
      {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"Bash","input":{"command":"cargo test"}}]},
      {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","is_error":true,"content":"fatal runtime error: out of memory"}]},
      {"role":"assistant","content":[{"type":"tool_use","id":"tu_2","name":"Bash","input":{"command":"cargo test"}}]},
      {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_2","is_error":true,"content":"fatal runtime error: out of memory"}]}
    ]
  }'
```

```text
x-agentgateway-routed-model: claude-3-7-sonnet
x-agentgateway-routing-decision: escalate
```
