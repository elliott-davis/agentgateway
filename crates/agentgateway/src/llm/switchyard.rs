//! Contract-aware dynamic model routing backed by NeMo Switchyard (`switchyard-libsy`).
//!
//! libsy only produces a *recommendation*. The gateway re-checks it against the configured
//! equivalence class (narrowed by client capability headers), applies session affinity, and
//! then rewrites only the `model` of the client request. libsy's own rewritten request
//! (tier prompts, handoff notes) is never forwarded.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::http::{HeaderMap, StatusCode};
use async_trait::async_trait;
use quick_cache::sync::Cache;
use serde_json::Value;
use switchyard_libsy as sy;
use switchyard_protocol as sp;
use tracing::instrument::WithSubscriber;

use crate::{apply, schema_enum, schema_ser_schema};

mod adapter;
#[cfg(test)]
mod tests;

/// Upper bound for one libsy evaluation, including any judge calls.
const EVALUATION_TIMEOUT: Duration = Duration::from_millis(500);
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(3600);
const DEFAULT_MAX_SESSIONS: usize = 10_000;
const MAX_SESSION_ID_LEN: usize = 256;
pub const STRATEGY_STAGE_ROUTER: &str = "stage_router";

pub const HEADER_ROUTING_MODE: &str = "x-routing-mode";
pub const HEADER_ALLOW_DYNAMIC: &str = "x-allow-dynamic-routing";
pub const HEADER_ALLOWED_MODELS: &str = "x-routing-allowed-models";
pub const HEADER_MIN_CONTEXT: &str = "x-routing-min-context-window";

#[apply(schema_enum!)]
#[derive(Default)]
pub enum SwitchyardStrategy {
	#[default]
	StageRouter,
}

#[apply(schema_enum!)]
pub enum Tier {
	Capable,
	Efficient,
}

#[apply(schema_enum!)]
#[derive(Default)]
pub enum FallbackBehavior {
	/// Route to the anchor (first) target when a recommendation is out of class.
	#[default]
	FallbackToRequested,
	/// Fail the request when a recommendation is out of class.
	RejectIfOutOfClass,
}

#[apply(schema_ser_schema!)]
pub struct SwitchyardTarget {
	pub model: String,
	pub tier: Tier,
	/// Context window in tokens. Unknown windows never satisfy `x-routing-min-context-window`.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub context_window: Option<u64>,
	// XDS-only resolution state. User-facing configuration does not expose this field.
	#[serde(skip_serializing_if = "std::ops::Not::not")]
	pub invalid: bool,
}

#[apply(schema_ser_schema!)]
pub struct SessionAffinity {
	/// Request header carrying the agent session identifier.
	pub header: String,
	#[serde(with = "crate::serdes::serde_dur")]
	#[cfg_attr(feature = "schema", schemars(with = "String"))]
	pub idle_timeout: Duration,
	pub max_sessions: usize,
}

impl SessionAffinity {
	pub fn new(header: String, idle_timeout: Option<Duration>, max_sessions: Option<usize>) -> Self {
		Self {
			header,
			idle_timeout: idle_timeout
				.filter(|d| !d.is_zero())
				.unwrap_or(DEFAULT_IDLE_TIMEOUT),
			max_sessions: max_sessions
				.filter(|m| *m > 0)
				.unwrap_or(DEFAULT_MAX_SESSIONS),
		}
	}
}

#[apply(schema_ser_schema!)]
pub struct StageRouterSettings {
	pub efficient_first: bool,
	pub confidence_threshold: f64,
	pub capable_hold_turns: u32,
}

impl Default for StageRouterSettings {
	fn default() -> Self {
		Self {
			efficient_first: false,
			confidence_threshold: 0.5,
			capable_hold_turns: 2,
		}
	}
}

/// Seam for libsy's routing-time model calls (e.g. a judge model). Intentionally not wired to a
/// real backend yet; the default fails every call, which the router treats as an evaluation error.
#[async_trait]
pub trait JudgeClient: Send + Sync + 'static {
	async fn call(&self, request: sp::Request, models: Vec<sp::ModelId>) -> sy::Result<sp::Response>;
}

pub struct UnconfiguredJudge;

#[async_trait]
impl JudgeClient for UnconfiguredJudge {
	async fn call(&self, _: sp::Request, _: Vec<sp::ModelId>) -> sy::Result<sp::Response> {
		Err(sy::LibsyError::AlgorithmError {
			message: "no judge model is configured for switchyard routing".to_string(),
		})
	}
}

#[derive(Clone)]
struct AffinityEntry {
	model: String,
	turns: u64,
	last_seen: Instant,
}

/// Shared, non-serialized state. Cloning a [`SwitchyardRouting`] shares it.
pub struct SwitchyardRuntime {
	algorithm: Arc<dyn sy::Algorithm>,
	judge: Arc<dyn JudgeClient>,
	affinity: Arc<Cache<String, AffinityEntry>>,
}

impl std::fmt::Debug for SwitchyardRuntime {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("SwitchyardRuntime")
			.field("algorithm", &self.algorithm.name())
			.finish_non_exhaustive()
	}
}

#[apply(schema_ser_schema!)]
pub struct SwitchyardRouting {
	pub strategy: SwitchyardStrategy,
	pub equivalence_class: String,
	/// The first valid target is the anchor.
	pub targets: Vec<SwitchyardTarget>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub session_affinity: Option<SessionAffinity>,
	pub fallback_behavior: FallbackBehavior,
	pub stage_router: StageRouterSettings,
	#[serde(skip)]
	#[cfg_attr(feature = "schema", schemars(skip))]
	runtime: Arc<SwitchyardRuntime>,
}

/// The routing outcome, stored in request extensions for response headers and telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRoutingDecision {
	pub original_model: String,
	/// Empty when the request was rejected.
	pub routed_model: String,
	pub decision: String,
	pub strategy: &'static str,
	pub equivalence_class: String,
	pub session_affinity_hit: bool,
}

#[derive(Debug)]
pub struct SwitchyardError {
	pub status: StatusCode,
	pub code: &'static str,
	pub message: String,
	pub decision: Option<ModelRoutingDecision>,
}

impl SwitchyardError {
	fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
		Self {
			status,
			code,
			message: message.into(),
			decision: None,
		}
	}
}

struct Recommendation {
	model: String,
	source: String,
}

impl SwitchyardRouting {
	pub fn new(
		equivalence_class: String,
		targets: Vec<SwitchyardTarget>,
		session_affinity: Option<SessionAffinity>,
		fallback_behavior: FallbackBehavior,
		stage_router: StageRouterSettings,
	) -> anyhow::Result<Self> {
		let mode = if stage_router.efficient_first {
			sy::PickerMode::EfficientFirst
		} else {
			sy::PickerMode::CapableFirst
		};
		let mut config = sy::StageRouterConfig::new(mode, stage_router.confidence_threshold);
		config.capable_hold_turns = stage_router.capable_hold_turns;
		let algorithm = sy::StageRouter::new(config).map_err(|e| anyhow::anyhow!("{e}"))?;
		let capacity = session_affinity
			.as_ref()
			.map_or(1, |a| a.max_sessions.max(1));
		Ok(Self {
			strategy: SwitchyardStrategy::StageRouter,
			equivalence_class,
			targets,
			session_affinity,
			fallback_behavior,
			stage_router,
			runtime: Arc::new(SwitchyardRuntime {
				algorithm: Arc::new(algorithm),
				judge: Arc::new(UnconfiguredJudge),
				affinity: Arc::new(Cache::new(capacity)),
			}),
		})
	}

	/// Replaces the judge used to serve libsy's routing-time model calls.
	pub fn with_judge(mut self, judge: Arc<dyn JudgeClient>) -> Self {
		self.runtime = Arc::new(SwitchyardRuntime {
			algorithm: self.runtime.algorithm.clone(),
			judge,
			affinity: self.runtime.affinity.clone(),
		});
		self
	}

	#[cfg(test)]
	pub fn with_algorithm(mut self, algorithm: Arc<dyn sy::Algorithm>) -> Self {
		self.runtime = Arc::new(SwitchyardRuntime {
			algorithm,
			judge: self.runtime.judge.clone(),
			affinity: self.runtime.affinity.clone(),
		});
		self
	}

	fn tier_of(&self, model: &str) -> Option<Tier> {
		self
			.targets
			.iter()
			.find(|t| t.model == model)
			.map(|t| t.tier)
	}

	fn session_id(&self, headers: &HeaderMap) -> Option<String> {
		let affinity = self.session_affinity.as_ref()?;
		let value = headers.get(affinity.header.as_str())?.to_str().ok()?.trim();
		(!value.is_empty() && value.len() <= MAX_SESSION_ID_LEN).then(|| value.to_string())
	}

	fn live_pin(&self, session: &str) -> Option<AffinityEntry> {
		let idle = self.session_affinity.as_ref()?.idle_timeout;
		let entry = self.runtime.affinity.get(session)?;
		(entry.last_seen.elapsed() <= idle).then_some(entry)
	}

	/// Runs libsy with a hard deadline. Any error or timeout is `None`.
	async fn evaluate(
		&self,
		request: sp::Request,
		candidates: &[&SwitchyardTarget],
	) -> Option<Recommendation> {
		let models_for = |tier: Tier| {
			candidates
				.iter()
				.filter(|t| t.tier == tier)
				.map(|t| sp::ModelId::new(t.model.clone()))
				.collect::<Vec<_>>()
		};
		let models = Arc::new(sy::RuntimeModels::new(HashMap::from([
			(sp::Category::Capable, models_for(Tier::Capable)),
			(sp::Category::Efficient, models_for(Tier::Efficient)),
			(
				sp::Category::Any,
				candidates
					.iter()
					.map(|t| sp::ModelId::new(t.model.clone()))
					.collect(),
			),
		])));
		let judge = self.runtime.judge.clone();
		// libsy emits its own tracing spans and records fields on them after creation, which the
		// gateway's log formatter does not support (it asserts in debug builds). Evaluate with
		// tracing disabled; the gateway records the routing decision itself.
		let run = sy::drive(
			self.runtime.algorithm.clone(),
			request,
			models,
			move |call| {
				let judge = judge.clone();
				async move {
					let result = judge.call(call.request.clone(), call.models.clone()).await;
					call.respond(result)
				}
			},
		)
		.with_subscriber(tracing::subscriber::NoSubscriber::default());
		let outcome = match tokio::time::timeout(EVALUATION_TIMEOUT, run).await {
			Ok(Ok(outcome)) => outcome,
			Ok(Err(err)) => {
				tracing::debug!(%err, "switchyard evaluation failed");
				return None;
			},
			Err(_) => {
				tracing::debug!("switchyard evaluation timed out");
				return None;
			},
		};
		let model = outcome.selected_model_id().ok()?.to_string();
		let source = outcome
			.metadata
			.as_ref()
			.and_then(|m| m.evidence.as_ref())
			.and_then(|e| e.get("source"))
			.and_then(Value::as_str)
			.unwrap_or("unknown")
			.to_string();
		Some(Recommendation { model, source })
	}

	/// Chooses the model for one request. See the module docs for the decision order.
	pub async fn resolve(
		&self,
		virtual_model: &str,
		headers: &HeaderMap,
		body: Option<&Value>,
	) -> Result<ModelRoutingDecision, SwitchyardError> {
		let decide = |routed: &str, decision: &str, hit: bool| ModelRoutingDecision {
			original_model: virtual_model.to_string(),
			routed_model: routed.to_string(),
			decision: decision.to_string(),
			strategy: STRATEGY_STAGE_ROUTER,
			equivalence_class: self.equivalence_class.clone(),
			session_affinity_hit: hit,
		};
		let valid: Vec<&SwitchyardTarget> = self.targets.iter().filter(|t| !t.invalid).collect();
		let Some(anchor) = valid.first().copied() else {
			return Err(SwitchyardError::new(
				StatusCode::NOT_FOUND,
				"virtual_model_target_not_found",
				format!("Virtual model {virtual_model} has no valid targets"),
			));
		};

		if bypass_requested(headers) {
			return Ok(decide(&anchor.model, "bypass", false));
		}

		let candidates = narrow_candidates(&valid, headers)?;
		let fallback_choice = || -> &SwitchyardTarget {
			candidates
				.iter()
				.find(|t| t.model == anchor.model)
				.or_else(|| candidates.iter().find(|t| t.tier == anchor.tier))
				.unwrap_or(&candidates[0])
		};

		let session = self.session_id(headers);
		let request = adapter::build_request(headers, body, session.as_deref());
		let capable = candidates.iter().any(|t| t.tier == Tier::Capable);
		let efficient = candidates.iter().any(|t| t.tier == Tier::Efficient);
		// Always run the heuristic so libsy's per-session state sees every turn.
		let recommendation = if capable && efficient {
			self.evaluate(request, &candidates).await
		} else {
			Some(Recommendation {
				model: candidates[0].model.clone(),
				source: "constrained".to_string(),
			})
		};
		let in_class = |m: &str| candidates.iter().any(|t| t.model == m);

		let pinned = session.as_deref().and_then(|s| self.live_pin(s));
		let mut chosen: Option<(String, String, bool)> = None;
		if let Some(pin) = pinned.as_ref().filter(|p| in_class(&p.model)) {
			let switch = recommendation.as_ref().filter(|r| {
				matches!(
					r.source.as_str(),
					"override" | "dimensions" | "capable_hold"
				) && r.model != pin.model
					&& in_class(&r.model)
			});
			chosen = Some(match switch {
				Some(r) => {
					let label = if self.tier_of(&r.model) == Some(Tier::Capable) {
						"escalate"
					} else {
						"de-escalate"
					};
					(r.model.clone(), label.to_string(), false)
				},
				None => (pin.model.clone(), "session_affinity".to_string(), true),
			});
		}
		let (model, label, hit) = match chosen {
			Some(c) => c,
			None => match recommendation {
				Some(r) if in_class(&r.model) => (r.model, r.source, false),
				Some(_) => match self.fallback_behavior {
					FallbackBehavior::FallbackToRequested => (
						fallback_choice().model.clone(),
						"fallback_to_requested".to_string(),
						false,
					),
					FallbackBehavior::RejectIfOutOfClass => {
						let mut err = SwitchyardError::new(
							StatusCode::SERVICE_UNAVAILABLE,
							"model_routing_out_of_class",
							format!(
								"Virtual model {virtual_model} recommended a model outside equivalence class {}",
								self.equivalence_class
							),
						);
						err.decision = Some(decide("", "rejected_out_of_class", false));
						return Err(err);
					},
				},
				None => (
					fallback_choice().model.clone(),
					"evaluation_error".to_string(),
					false,
				),
			},
		};

		if let Some(session) = session {
			let turns = pinned.map_or(0, |p| p.turns) + 1;
			self.runtime.affinity.insert(
				session,
				AffinityEntry {
					model: model.clone(),
					turns,
					last_seen: Instant::now(),
				},
			);
		}
		Ok(decide(&model, &label, hit))
	}
}

fn bypass_requested(headers: &HeaderMap) -> bool {
	let header = |name: &str| {
		headers
			.get(name)
			.and_then(|v| v.to_str().ok())
			.map(str::trim)
	};
	header(HEADER_ROUTING_MODE).is_some_and(|v| v.eq_ignore_ascii_case("strict"))
		|| header(HEADER_ALLOW_DYNAMIC).is_some_and(|v| v.eq_ignore_ascii_case("false"))
}

/// Applies the client capability floor to the valid targets.
fn narrow_candidates<'a>(
	valid: &[&'a SwitchyardTarget],
	headers: &HeaderMap,
) -> Result<Vec<&'a SwitchyardTarget>, SwitchyardError> {
	let mut candidates = valid.to_vec();
	if headers.contains_key(HEADER_ALLOWED_MODELS) {
		let allowed: Vec<String> = headers
			.get_all(HEADER_ALLOWED_MODELS)
			.iter()
			.filter_map(|v| v.to_str().ok())
			.flat_map(|v| {
				v.trim()
					.trim_start_matches('[')
					.trim_end_matches(']')
					.split(',')
					.map(|m| {
						m.trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace())
							.to_string()
					})
					.collect::<Vec<_>>()
			})
			.filter(|m| !m.is_empty())
			.collect();
		candidates.retain(|t| allowed.contains(&t.model));
	}
	if let Some(value) = headers.get(HEADER_MIN_CONTEXT) {
		let min = value
			.to_str()
			.ok()
			.and_then(|v| v.trim().parse::<u64>().ok())
			.ok_or_else(|| {
				SwitchyardError::new(
					StatusCode::BAD_REQUEST,
					"invalid_routing_header",
					format!("{HEADER_MIN_CONTEXT} must be an unsigned integer"),
				)
			})?;
		candidates.retain(|t| t.context_window.is_some_and(|w| w >= min));
	}
	if candidates.is_empty() {
		return Err(SwitchyardError::new(
			StatusCode::BAD_REQUEST,
			"routing_constraints_unsatisfiable",
			"No target in the equivalence class satisfies the routing constraints",
		));
	}
	Ok(candidates)
}
