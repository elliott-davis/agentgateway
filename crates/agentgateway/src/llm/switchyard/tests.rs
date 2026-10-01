use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ::http::{HeaderMap, HeaderName, HeaderValue};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::*;

const CAPABLE: &str = "claude-3-7-sonnet";
const EFFICIENT: &str = "claude-3-5-sonnet";

fn target(model: &str, tier: Tier, context_window: Option<u64>) -> SwitchyardTarget {
	SwitchyardTarget {
		model: model.to_string(),
		tier,
		context_window,
		invalid: false,
	}
}

fn routing_with(
	fallback: FallbackBehavior,
	efficient_first: bool,
	idle: Option<Duration>,
) -> SwitchyardRouting {
	SwitchyardRouting::new(
		"claude-class".to_string(),
		vec![
			target(CAPABLE, Tier::Capable, Some(200_000)),
			target(EFFICIENT, Tier::Efficient, Some(8_000)),
		],
		Some(SessionAffinity::new(
			"x-agent-session-id".to_string(),
			idle,
			None,
		)),
		fallback,
		StageRouterSettings {
			efficient_first,
			..Default::default()
		},
	)
	.expect("valid routing")
}

fn real_router() -> SwitchyardRouting {
	routing_with(FallbackBehavior::FallbackToRequested, true, None)
}

fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
	let mut h = HeaderMap::new();
	for (k, v) in pairs {
		h.insert(
			HeaderName::from_bytes(k.as_bytes()).unwrap(),
			HeaderValue::from_str(v).unwrap(),
		);
	}
	h
}

fn plain_body(prompt: &str) -> Value {
	json!({"model": "smart", "messages": [{"role": "user", "content": prompt}]})
}

fn failing_chat_body(failures: usize) -> Value {
	let mut messages = vec![json!({"role": "user", "content": "fix the build"})];
	for i in 0..failures {
		messages.push(json!({"role": "assistant", "tool_calls": [{
			"id": format!("call_{i}"), "type": "function",
			"function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}
		}]}));
		messages.push(json!({"role": "tool", "tool_call_id": format!("call_{i}"),
			"content": "fatal runtime error: out of memory"}));
	}
	json!({"model": "smart", "messages": messages})
}

fn failing_anthropic_body(failures: usize) -> Value {
	let mut messages = vec![json!({"role": "user", "content": "fix the build"})];
	for i in 0..failures {
		messages.push(json!({"role": "assistant", "content": [
			{"type": "tool_use", "id": format!("tu_{i}"), "name": "Bash", "input": {"command": "cargo test"}}
		]}));
		messages.push(json!({"role": "user", "content": [
			{"type": "tool_result", "tool_use_id": format!("tu_{i}"), "is_error": true, "content": "command failed"}
		]}));
	}
	json!({"model": "smart", "system": "You are a coding agent", "messages": messages})
}

/// Returns `(model, source)` pairs in order, repeating the last one.
struct Scripted {
	script: Mutex<VecDeque<(String, String)>>,
	calls: AtomicUsize,
}

impl Scripted {
	fn new(script: &[(&str, &str)]) -> Arc<Self> {
		Arc::new(Self {
			script: Mutex::new(
				script
					.iter()
					.map(|(m, s)| (m.to_string(), s.to_string()))
					.collect(),
			),
			calls: AtomicUsize::new(0),
		})
	}
}

#[async_trait]
impl sy::Algorithm for Scripted {
	fn name(&self) -> &str {
		"scripted"
	}
	async fn route(
		self: Arc<Self>,
		_driver: sy::Driver,
		request: sp::Request,
	) -> sy::Result<sy::RoutingOutcome> {
		self.calls.fetch_add(1, Ordering::SeqCst);
		let (model, source) = {
			let mut script = self.script.lock().unwrap();
			if script.len() > 1 {
				script.pop_front().unwrap()
			} else {
				script.front().cloned().unwrap()
			}
		};
		let mut outcome = sy::RoutingOutcome::route_to(sp::ModelId::new(model), vec![], request);
		outcome.metadata = Some(sy::OutcomeMetadata::new(
			"scripted".to_string(),
			Some(json!({"source": source})),
		));
		Ok(outcome)
	}
}

fn with_script(
	r: SwitchyardRouting,
	script: &[(&str, &str)],
) -> (SwitchyardRouting, Arc<Scripted>) {
	let algo = Scripted::new(script);
	(r.with_algorithm(algo.clone()), algo)
}

async fn route(r: &SwitchyardRouting, h: &HeaderMap, body: &Value) -> ModelRoutingDecision {
	r.resolve("smart", h, Some(body)).await.expect("routes")
}

#[tokio::test]
async fn repeated_tool_failures_route_to_capable() {
	let r = real_router();
	let d = route(&r, &HeaderMap::new(), &failing_chat_body(3)).await;
	assert_eq!(d.routed_model, CAPABLE, "{d:?}");
	assert_eq!(d.original_model, "smart");
	assert_eq!(d.strategy, "stage_router");
	assert_eq!(d.equivalence_class, "claude-class");
	assert_ne!(d.decision, "evaluation_error");
}

#[tokio::test]
async fn anthropic_tool_result_errors_route_to_capable() {
	let r = real_router();
	let d = route(&r, &HeaderMap::new(), &failing_anthropic_body(3)).await;
	assert_eq!(d.routed_model, CAPABLE, "{d:?}");
}

#[tokio::test]
async fn low_complexity_prompt_efficient_first_routes_to_efficient() {
	let r = real_router();
	let d = route(&r, &HeaderMap::new(), &plain_body("say hi")).await;
	assert_eq!(d.routed_model, EFFICIENT, "{d:?}");
	assert_eq!(d.decision, "fall_open");
}

#[tokio::test]
async fn real_router_never_leaves_the_class() {
	let r = real_router();
	for body in [
		plain_body("say hi"),
		failing_chat_body(1),
		failing_anthropic_body(2),
		json!({"model": "smart"}),
		json!({"model": "smart", "input": "responses api shape"}),
	] {
		let d = route(&r, &HeaderMap::new(), &body).await;
		assert!(
			[CAPABLE, EFFICIENT].contains(&d.routed_model.as_str()),
			"{d:?}"
		);
	}
}

#[tokio::test]
async fn out_of_class_recommendation_falls_back_to_anchor() {
	let (r, _) = with_script(
		routing_with(FallbackBehavior::FallbackToRequested, false, None),
		&[("llama-3.1-8b-instruct", "dimensions")],
	);
	let d = route(&r, &HeaderMap::new(), &plain_body("hi")).await;
	assert_eq!(d.routed_model, CAPABLE);
	assert_eq!(d.decision, "fallback_to_requested");
}

#[tokio::test]
async fn out_of_class_recommendation_rejected_when_configured() {
	let (r, _) = with_script(
		routing_with(FallbackBehavior::RejectIfOutOfClass, false, None),
		&[("llama-3.1-8b-instruct", "dimensions")],
	);
	let err = r
		.resolve("smart", &HeaderMap::new(), Some(&plain_body("hi")))
		.await
		.unwrap_err();
	assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
	assert_eq!(err.code, "model_routing_out_of_class");
	let d = err.decision.expect("headers still emitted");
	assert_eq!(d.original_model, "smart");
	assert!(d.routed_model.is_empty());
}

#[tokio::test]
async fn evaluation_error_fails_safe_even_when_rejecting() {
	struct Failing;
	#[async_trait]
	impl sy::Algorithm for Failing {
		fn name(&self) -> &str {
			"failing"
		}
		async fn route(
			self: Arc<Self>,
			_: sy::Driver,
			_: sp::Request,
		) -> sy::Result<sy::RoutingOutcome> {
			Err(sy::LibsyError::NoTargets)
		}
	}
	let r = routing_with(FallbackBehavior::RejectIfOutOfClass, false, None)
		.with_algorithm(Arc::new(Failing));
	let d = route(&r, &HeaderMap::new(), &plain_body("hi")).await;
	assert_eq!(d.routed_model, CAPABLE);
	assert_eq!(d.decision, "evaluation_error");
}

#[tokio::test(start_paused = true)]
async fn evaluation_timeout_is_an_evaluation_error() {
	struct Hang;
	#[async_trait]
	impl sy::Algorithm for Hang {
		fn name(&self) -> &str {
			"hang"
		}
		async fn route(
			self: Arc<Self>,
			_: sy::Driver,
			_: sp::Request,
		) -> sy::Result<sy::RoutingOutcome> {
			std::future::pending().await
		}
	}
	let r = real_router().with_algorithm(Arc::new(Hang));
	let d = route(&r, &HeaderMap::new(), &plain_body("hi")).await;
	assert_eq!(d.decision, "evaluation_error");
	assert_eq!(d.routed_model, CAPABLE);
}

#[tokio::test]
async fn allowed_models_header_narrows_candidates() {
	let (r, algo) = with_script(real_router(), &[(CAPABLE, "dimensions")]);
	// Only the efficient model is allowed; the capable recommendation is out of class.
	let h = headers(&[(HEADER_ALLOWED_MODELS, r#"[ "claude-3-5-sonnet" ]"#)]);
	let d = route(&r, &h, &plain_body("hi")).await;
	// One tier remains, so libsy is skipped.
	assert_eq!(algo.calls.load(Ordering::SeqCst), 0);
	assert_eq!(d.routed_model, EFFICIENT);
	assert_eq!(d.decision, "constrained");

	let h = headers(&[(HEADER_ALLOWED_MODELS, "'claude-3-7-sonnet', other")]);
	let d = route(&r, &h, &plain_body("hi")).await;
	assert_eq!(d.routed_model, CAPABLE);
}

#[tokio::test]
async fn min_context_window_narrows_candidates() {
	let r = real_router();
	let h = headers(&[(HEADER_MIN_CONTEXT, "100000")]);
	let d = route(&r, &h, &plain_body("hi")).await;
	assert_eq!(d.routed_model, CAPABLE);
	assert_eq!(d.decision, "constrained");
}

#[tokio::test]
async fn unknown_context_window_fails_the_floor() {
	let r = SwitchyardRouting::new(
		"c".into(),
		vec![
			target("a", Tier::Capable, None),
			target("b", Tier::Efficient, Some(10)),
		],
		None,
		FallbackBehavior::FallbackToRequested,
		StageRouterSettings::default(),
	)
	.unwrap();
	let h = headers(&[(HEADER_MIN_CONTEXT, "5")]);
	let d = route(&r, &h, &plain_body("hi")).await;
	assert_eq!(d.routed_model, "b");
}

#[tokio::test]
async fn unsatisfiable_and_invalid_constraints_are_client_errors() {
	let r = real_router();
	for (h, code) in [
		(
			headers(&[(HEADER_MIN_CONTEXT, "999999999")]),
			"routing_constraints_unsatisfiable",
		),
		(
			headers(&[(HEADER_ALLOWED_MODELS, "nope")]),
			"routing_constraints_unsatisfiable",
		),
		(
			headers(&[(HEADER_MIN_CONTEXT, "lots")]),
			"invalid_routing_header",
		),
	] {
		let err = r
			.resolve("smart", &h, Some(&plain_body("hi")))
			.await
			.unwrap_err();
		assert_eq!(err.status, StatusCode::BAD_REQUEST);
		assert_eq!(err.code, code);
	}
}

#[tokio::test]
async fn no_valid_target_is_not_found() {
	let mut r = real_router();
	for t in &mut r.targets {
		t.invalid = true;
	}
	let err = r
		.resolve("smart", &HeaderMap::new(), None)
		.await
		.unwrap_err();
	assert_eq!(err.status, StatusCode::NOT_FOUND);
	assert_eq!(err.code, "virtual_model_target_not_found");
}

#[tokio::test]
async fn bypass_headers_skip_evaluation_and_cache() {
	for h in [
		headers(&[(HEADER_ROUTING_MODE, "Strict"), ("x-agent-session-id", "s")]),
		headers(&[(HEADER_ALLOW_DYNAMIC, "false"), ("x-agent-session-id", "s")]),
	] {
		let (r, algo) = with_script(real_router(), &[(EFFICIENT, "dimensions")]);
		let d = route(&r, &h, &plain_body("hi")).await;
		assert_eq!(d.routed_model, CAPABLE);
		assert_eq!(d.decision, "bypass");
		assert_eq!(algo.calls.load(Ordering::SeqCst), 0);
		assert!(r.live_pin("s").is_none());
	}
}

#[tokio::test]
async fn session_affinity_holds_until_a_decisive_signal() {
	let (r, algo) = with_script(
		real_router(),
		&[
			(EFFICIENT, "fall_open"),
			(CAPABLE, "fall_open"),
			(CAPABLE, "dimensions"),
			(EFFICIENT, "dimensions"),
		],
	);
	let h = headers(&[("x-agent-session-id", "sess-1")]);

	let d1 = route(&r, &h, &plain_body("short")).await;
	assert_eq!(
		(d1.routed_model.as_str(), d1.decision.as_str()),
		(EFFICIENT, "fall_open")
	);
	assert!(!d1.session_affinity_hit);

	// Very different, much longer body; non-decisive recommendation stays pinned.
	let long = plain_body(&"lorem ipsum ".repeat(5_000));
	let d2 = route(&r, &h, &long).await;
	assert_eq!(d2.routed_model, EFFICIENT);
	assert_eq!(d2.decision, "session_affinity");
	assert!(d2.session_affinity_hit);

	let d3 = route(&r, &h, &long).await;
	assert_eq!(d3.routed_model, CAPABLE);
	assert_eq!(d3.decision, "escalate");
	assert!(!d3.session_affinity_hit);

	let d4 = route(&r, &h, &long).await;
	assert_eq!(d4.routed_model, EFFICIENT);
	assert_eq!(d4.decision, "de-escalate");
	assert_eq!(
		algo.calls.load(Ordering::SeqCst),
		4,
		"heuristic runs every turn"
	);
	assert_eq!(r.live_pin("sess-1").unwrap().turns, 4);

	// Another session is independent.
	let other = headers(&[("x-agent-session-id", "sess-2")]);
	assert!(!route(&r, &other, &long).await.session_affinity_hit);
}

#[tokio::test]
async fn real_signals_escalate_a_pinned_session() {
	let r = real_router();
	let h = headers(&[("x-agent-session-id", "real")]);
	let d1 = route(&r, &h, &plain_body("say hi")).await;
	assert_eq!(d1.routed_model, EFFICIENT);
	let d2 = route(&r, &h, &failing_chat_body(3)).await;
	assert_eq!(d2.routed_model, CAPABLE, "{d2:?}");
	assert_eq!(d2.decision, "escalate");
}

#[tokio::test]
async fn expired_session_entries_are_treated_as_new() {
	let (r, _) = with_script(
		routing_with(
			FallbackBehavior::FallbackToRequested,
			true,
			Some(Duration::from_millis(30)),
		),
		&[(EFFICIENT, "fall_open"), (CAPABLE, "fall_open")],
	);
	let h = headers(&[("x-agent-session-id", "idle")]);
	assert_eq!(
		route(&r, &h, &plain_body("a")).await.routed_model,
		EFFICIENT
	);
	tokio::time::sleep(Duration::from_millis(80)).await;
	let d = route(&r, &h, &plain_body("a")).await;
	assert_eq!(d.routed_model, CAPABLE);
	assert!(!d.session_affinity_hit);
	assert_eq!(r.live_pin("idle").unwrap().turns, 1);
}

#[tokio::test]
async fn invalid_session_ids_are_ignored() {
	let r = real_router();
	assert!(
		r.session_id(&headers(&[("x-agent-session-id", "")]))
			.is_none()
	);
	let long = "x".repeat(MAX_SESSION_ID_LEN + 1);
	assert!(
		r.session_id(&headers(&[("x-agent-session-id", &long)]))
			.is_none()
	);
	assert!(
		r.session_id(&headers(&[("x-agent-session-id", "ok")]))
			.is_some()
	);
}

#[tokio::test]
async fn clones_share_session_state() {
	let (r, _) = with_script(
		real_router(),
		&[(EFFICIENT, "fall_open"), (CAPABLE, "fall_open")],
	);
	let h = headers(&[("x-agent-session-id", "shared")]);
	route(&r, &h, &plain_body("a")).await;
	let clone = r.clone();
	assert!(
		route(&clone, &h, &plain_body("a"))
			.await
			.session_affinity_hit
	);
}

struct MockJudge(AtomicUsize);

#[async_trait]
impl JudgeClient for MockJudge {
	async fn call(&self, _: sp::Request, models: Vec<sp::ModelId>) -> sy::Result<sp::Response> {
		assert_eq!(models[0].as_str(), "judge");
		self.0.fetch_add(1, Ordering::SeqCst);
		tokio::task::yield_now().await;
		Ok(sp::Response {
			llm_response: sp::LlmResponse::Agg(sp::text_response(None, "verdict")),
			metadata: None,
			upstream_headers: HeaderMap::new(),
		})
	}
}

struct AsksJudge;

#[async_trait]
impl sy::Algorithm for AsksJudge {
	fn name(&self) -> &str {
		"asks_judge"
	}
	async fn route(
		self: Arc<Self>,
		driver: sy::Driver,
		request: sp::Request,
	) -> sy::Result<sy::RoutingOutcome> {
		driver
			.call_model(request.clone(), vec![sp::ModelId::new("judge")])
			.await?;
		let mut outcome = sy::RoutingOutcome::route_to(sp::ModelId::new(EFFICIENT), vec![], request);
		outcome.metadata = Some(sy::OutcomeMetadata::new(
			"asks_judge".to_string(),
			Some(json!({"source": "llm-classifier"})),
		));
		Ok(outcome)
	}
}

#[tokio::test(flavor = "current_thread")]
async fn judge_calls_are_served_asynchronously() {
	// libsy's `call_model` span records fields after creation, which the gateway's log formatter
	// rejects (debug_assert) once any test has installed the global subscriber. On a current-thread
	// runtime the spawned algorithm task shares this thread, so a thread-local default covers it.
	let _no_tracing = tracing::subscriber::set_default(tracing::subscriber::NoSubscriber::default());
	let judge = Arc::new(MockJudge(AtomicUsize::new(0)));
	let r = real_router()
		.with_algorithm(Arc::new(AsksJudge))
		.with_judge(judge.clone());
	let d = route(&r, &HeaderMap::new(), &plain_body("hi")).await;
	assert_eq!(judge.0.load(Ordering::SeqCst), 1);
	assert_eq!(d.routed_model, EFFICIENT);
	assert_eq!(d.decision, "llm-classifier");
}

#[tokio::test(flavor = "current_thread")]
async fn unconfigured_judge_is_an_evaluation_error() {
	// libsy's `call_model` span records fields after creation, which the gateway's log formatter
	// rejects (debug_assert) once any test has installed the global subscriber. On a current-thread
	// runtime the spawned algorithm task shares this thread, so a thread-local default covers it.
	let _no_tracing = tracing::subscriber::set_default(tracing::subscriber::NoSubscriber::default());
	let r = real_router().with_algorithm(Arc::new(AsksJudge));
	let d = route(&r, &HeaderMap::new(), &plain_body("hi")).await;
	assert_eq!(d.decision, "evaluation_error");
	assert_eq!(d.routed_model, CAPABLE);
}

#[test]
fn invalid_confidence_threshold_is_rejected() {
	let err = SwitchyardRouting::new(
		"c".into(),
		vec![target("a", Tier::Capable, None)],
		None,
		FallbackBehavior::FallbackToRequested,
		StageRouterSettings {
			confidence_threshold: 2.0,
			..Default::default()
		},
	)
	.unwrap_err();
	assert!(err.to_string().contains("confidence_threshold"), "{err}");
}

#[test]
fn adapter_converts_openai_and_anthropic_shapes() {
	let h = headers(&[("x-switchyard-session-id", "from-header")]);
	let req = adapter::build_request(&h, Some(&failing_chat_body(1)), Some("ours"));
	assert_eq!(
		req.metadata.as_ref().unwrap().session_id.as_deref(),
		Some("ours")
	);
	let msgs = &req.llm_request.messages;
	assert_eq!(msgs.len(), 3);
	assert!(
		matches!(msgs[1].content[0], sp::ContentBlock::ToolCall(ref c) if c.name == "Bash" && c.arguments["command"] == "cargo test")
	);
	assert!(
		matches!(msgs[2].content[0], sp::ContentBlock::ToolResult(ref r) if r.tool_call_id == "call_0")
	);
	assert_eq!(msgs[2].role, sp::Role::Tool);

	let req = adapter::build_request(&h, Some(&failing_anthropic_body(1)), None);
	assert_eq!(
		req.metadata.as_ref().unwrap().session_id.as_deref(),
		Some("from-header")
	);
	assert_eq!(req.llm_request.instructions.len(), 1);
	assert!(
		matches!(req.llm_request.messages[2].content[0], sp::ContentBlock::ToolResult(ref r) if r.is_error == Some(true))
	);

	let req = adapter::build_request(&h, None, None);
	assert!(req.llm_request.messages.is_empty());
}
