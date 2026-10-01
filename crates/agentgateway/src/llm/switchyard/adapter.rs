//! Adapts an already-parsed client JSON body into a `switchyard_protocol::Request`.
//!
//! The result is used only to obtain a model recommendation; it is never forwarded upstream.

use ::http::HeaderMap;
use serde_json::Value;
use switchyard_protocol as sp;

/// Builds the libsy request for OpenAI chat completions and Anthropic messages bodies.
/// Other shapes (Responses API, path or multipart models) yield an empty `LlmRequest`, which
/// StageRouter treats as "no signal" and falls open.
pub(super) fn build_request(
	headers: &HeaderMap,
	body: Option<&Value>,
	session_id: Option<&str>,
) -> sp::Request {
	let mut metadata = sp::Metadata::from_headers(headers);
	if let Some(session_id) = session_id {
		metadata.session_id = Some(session_id.to_string());
	}
	let mut llm_request = sp::LlmRequest::default();
	if let Some(body) = body.and_then(Value::as_object) {
		let mut anthropic = body.contains_key("system");
		llm_request.model = body
			.get("model")
			.and_then(Value::as_str)
			.map(ToString::to_string);
		llm_request.stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
		if let Some(system) = body.get("system") {
			let content = text_blocks(system);
			if !content.is_empty() {
				llm_request.instructions.push(sp::InstructionBlock {
					role: sp::Role::System,
					content,
				});
			}
		}
		for message in body
			.get("messages")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
		{
			anthropic |= convert_message(message, &mut llm_request);
		}
		llm_request.tools = body
			.get("tools")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.filter_map(convert_tool)
			.collect();
		metadata.wire_format = Some(if anthropic {
			sp::WireFormat::AnthropicMessages
		} else {
			sp::WireFormat::OpenAiChat
		});
	}
	sp::Request {
		llm_request,
		raw_request: None,
		metadata: Some(metadata),
	}
}

fn role(role: &str) -> sp::Role {
	match role {
		"system" => sp::Role::System,
		"developer" => sp::Role::Developer,
		"assistant" => sp::Role::Assistant,
		"tool" => sp::Role::Tool,
		_ => sp::Role::User,
	}
}

fn text(text: &str) -> sp::ContentBlock {
	sp::ContentBlock::Text {
		text: text.to_string(),
	}
}

/// Text from a string or an array of text parts; non-text parts are dropped.
fn text_blocks(value: &Value) -> Vec<sp::ContentBlock> {
	match value {
		Value::String(s) => vec![text(s)],
		Value::Array(parts) => parts
			.iter()
			.filter_map(|p| match p {
				Value::String(s) => Some(text(s)),
				_ => p
					.get("text")
					.and_then(Value::as_str)
					.filter(|_| {
						p.get("type")
							.and_then(Value::as_str)
							.is_none_or(|t| t == "text")
					})
					.map(text),
			})
			.collect(),
		_ => vec![],
	}
}

/// Returns true when the message used Anthropic-only block types.
fn convert_message(message: &Value, out: &mut sp::LlmRequest) -> bool {
	let role_name = message
		.get("role")
		.and_then(Value::as_str)
		.unwrap_or("user");
	let role = role(role_name);
	let mut anthropic = false;
	let mut content = Vec::new();
	match message.get("content") {
		Some(Value::Array(blocks)) => {
			for block in blocks {
				match block.get("type").and_then(Value::as_str) {
					Some("tool_use") => {
						anthropic = true;
						content.push(sp::ContentBlock::ToolCall(sp::ToolCall {
							id: str_field(block, "id"),
							name: str_field(block, "name"),
							arguments: block.get("input").cloned().unwrap_or(Value::Null),
						}));
					},
					Some("tool_result") => {
						anthropic = true;
						content.push(sp::ContentBlock::ToolResult(sp::ToolResult {
							tool_call_id: str_field(block, "tool_use_id"),
							content: block.get("content").map(text_blocks).unwrap_or_default(),
							is_error: block.get("is_error").and_then(Value::as_bool),
						}));
					},
					_ => content.extend(text_blocks(&Value::Array(vec![block.clone()]))),
				}
			}
		},
		Some(other) => content.extend(text_blocks(other)),
		None => {},
	}
	if role_name == "tool" {
		// OpenAI tool message: the whole content is the result of `tool_call_id`.
		content = vec![sp::ContentBlock::ToolResult(sp::ToolResult {
			tool_call_id: str_field(message, "tool_call_id"),
			content,
			is_error: None,
		})];
	}
	for call in message
		.get("tool_calls")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
	{
		let function = call.get("function").unwrap_or(call);
		let arguments = match function.get("arguments") {
			Some(Value::String(s)) => {
				serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
			},
			Some(v) => v.clone(),
			None => Value::Null,
		};
		content.push(sp::ContentBlock::ToolCall(sp::ToolCall {
			id: str_field(call, "id"),
			name: str_field(function, "name"),
			arguments,
		}));
	}
	match role {
		sp::Role::System | sp::Role::Developer => {
			if !content.is_empty() {
				out
					.instructions
					.push(sp::InstructionBlock { role, content });
			}
		},
		_ => out.messages.push(sp::Message { role, content }),
	}
	anthropic
}

fn convert_tool(tool: &Value) -> Option<sp::ToolDefinition> {
	let function = tool.get("function").unwrap_or(tool);
	Some(sp::ToolDefinition {
		name: function.get("name")?.as_str()?.to_string(),
		description: function
			.get("description")
			.and_then(Value::as_str)
			.map(ToString::to_string),
		parameters: function
			.get("parameters")
			.or_else(|| function.get("input_schema"))
			.cloned()
			.unwrap_or(Value::Null),
		strict: function.get("strict").and_then(Value::as_bool),
	})
}

fn str_field(value: &Value, key: &str) -> String {
	value
		.get(key)
		.and_then(Value::as_str)
		.unwrap_or_default()
		.to_string()
}
