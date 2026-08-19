use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    CreateElicitationRequest, CreateElicitationResponse, ElicitationAction as AcpElicitationAction,
    ElicitationFormMode, ElicitationSchema, ElicitationSessionScope, Meta, SessionId,
    CLIENT_METHOD_NAMES,
};
use agent_client_protocol::{
    Client, ConnectionTo, JsonRpcMessage, JsonRpcRequest, JsonRpcResponse, UntypedMessage,
};
use tracing::warn;

use crate::action_required_manager::ElicitationOutcome;
use crate::agents::Agent;

pub(super) struct FormElicitation {
    session_id: SessionId,
    elicitation_id: String,
    message: String,
    requested_schema: serde_json::Value,
    meta: Meta,
    recovered: bool,
}

impl FormElicitation {
    pub(super) fn new(
        session_id: SessionId,
        elicitation_id: String,
        message: String,
        requested_schema: serde_json::Value,
        meta: Meta,
        recovered: bool,
    ) -> Self {
        Self {
            session_id,
            elicitation_id,
            message,
            requested_schema,
            meta,
            recovered,
        }
    }
}

impl super::GooseAcpAgent {
    pub(super) async fn handle_form_elicitation(
        &self,
        cx: &ConnectionTo<Client>,
        agent: &Arc<Agent>,
        elicitation: FormElicitation,
    ) -> Result<(), agent_client_protocol::Error> {
        if self.supports_acp_elicitation() {
            self.send_form_elicitation(cx, agent, elicitation).await?;
        } else {
            warn!(
                session_id = %elicitation.session_id.0.as_ref(),
                elicitation_id = %elicitation.elicitation_id,
                "ACP client does not support form elicitation"
            );
            self.cancel_form_elicitation(
                agent,
                elicitation.session_id.0.as_ref(),
                &elicitation.elicitation_id,
                elicitation.recovered,
            )
            .await;
        }

        Ok(())
    }

    async fn send_form_elicitation(
        &self,
        cx: &ConnectionTo<Client>,
        agent: &Arc<Agent>,
        elicitation: FormElicitation,
    ) -> Result<(), agent_client_protocol::Error> {
        let session_id = elicitation.session_id.0.as_ref().to_string();
        let elicitation_id = elicitation.elicitation_id;
        if elicitation
            .requested_schema
            .get("url")
            .and_then(|url| url.as_str())
            .is_some()
        {
            warn!(
                session_id = %session_id,
                elicitation_id = %elicitation_id,
                "ACP URL elicitation is not supported"
            );
            finish_form_elicitation(
                agent,
                &session_id,
                &elicitation_id,
                ElicitationOutcome::Cancel,
                elicitation.recovered,
            )
            .await;
            return Ok(());
        }

        let requested_schema: ElicitationSchema =
            match serde_json::from_value(elicitation.requested_schema) {
                Ok(schema) => schema,
                Err(error) => {
                    finish_form_elicitation(
                        agent,
                        &session_id,
                        &elicitation_id,
                        ElicitationOutcome::Cancel,
                        elicitation.recovered,
                    )
                    .await;
                    return Err(agent_client_protocol::Error::internal_error()
                        .data(format!("Failed to parse ACP elicitation schema: {error}")));
                }
            };
        let has_live_waiter = agent
            .has_pending_elicitation(&session_id, &elicitation_id)
            .await;
        let mut meta = elicitation.meta;
        add_elicitation_meta(
            &mut meta,
            &elicitation_id,
            elicitation.recovered,
            if has_live_waiter {
                "response"
            } else {
                "prompt"
            },
        );
        let request = CreateElicitationRequest::new(
            ElicitationFormMode::new(
                ElicitationSessionScope::new(session_id.clone()),
                requested_schema,
            ),
            elicitation.message,
        )
        .meta(meta);

        let callback_agent = Arc::clone(agent);
        let callback_session_id = session_id.clone();
        let callback_elicitation_id = elicitation_id.clone();
        cx.send_request(CreateElicitationRequestMessage(request))
            .on_receiving_result(move |result| async move {
                match result {
                    Ok(response) => {
                        finish_form_elicitation(
                            &callback_agent,
                            &callback_session_id,
                            &callback_elicitation_id,
                            elicitation_response_from_acp(response.0),
                            elicitation.recovered,
                        )
                        .await;
                    }
                    Err(error) => {
                        warn!(
                            error = %error,
                            session_id = %callback_session_id,
                            elicitation_id = %callback_elicitation_id,
                            "ACP elicitation request disconnected; preserving pending response"
                        );
                    }
                }

                Ok(())
            })?;

        Ok(())
    }

    async fn cancel_form_elicitation(
        &self,
        agent: &Arc<Agent>,
        session_id: &str,
        elicitation_id: &str,
        recovered: bool,
    ) {
        finish_form_elicitation(
            agent,
            session_id,
            elicitation_id,
            ElicitationOutcome::Cancel,
            recovered,
        )
        .await;
    }
}

fn add_elicitation_meta(
    meta: &mut Meta,
    elicitation_id: &str,
    recovered: bool,
    continuation: &str,
) {
    // `response` means the original provider call is still blocked and this ACP
    // response resumes it directly. `prompt` means only the persisted question
    // survived, so the client must continue the session with a new user prompt.
    let goose = meta
        .entry("goose".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !goose.is_object() {
        *goose = serde_json::Value::Object(serde_json::Map::new());
    }
    let goose = goose
        .as_object_mut()
        .expect("goose elicitation metadata was initialized as an object");
    goose.insert(
        "elicitationId".to_string(),
        serde_json::Value::String(elicitation_id.to_string()),
    );
    goose.insert("recovered".to_string(), serde_json::Value::Bool(recovered));
    goose.insert(
        "continuation".to_string(),
        serde_json::Value::String(continuation.to_string()),
    );
}

#[derive(Debug, Clone)]
struct CreateElicitationRequestMessage(CreateElicitationRequest);

impl JsonRpcMessage for CreateElicitationRequestMessage {
    fn matches_method(method: &str) -> bool {
        method == CLIENT_METHOD_NAMES.elicitation_create
    }

    fn method(&self) -> &str {
        CLIENT_METHOD_NAMES.elicitation_create
    }

    fn to_untyped_message(&self) -> Result<UntypedMessage, agent_client_protocol::Error> {
        UntypedMessage::new(CLIENT_METHOD_NAMES.elicitation_create, &self.0)
    }

    fn parse_message(
        method: &str,
        params: &impl serde::Serialize,
    ) -> Result<Self, agent_client_protocol::Error> {
        if !Self::matches_method(method) {
            return Err(agent_client_protocol::Error::method_not_found());
        }

        Ok(Self(agent_client_protocol::util::json_cast_params(params)?))
    }
}

impl JsonRpcRequest for CreateElicitationRequestMessage {
    type Response = CreateElicitationResponseMessage;
}

#[derive(Debug, Clone)]
struct CreateElicitationResponseMessage(CreateElicitationResponse);

impl JsonRpcResponse for CreateElicitationResponseMessage {
    fn into_json(self, _method: &str) -> Result<serde_json::Value, agent_client_protocol::Error> {
        serde_json::to_value(self.0).map_err(agent_client_protocol::Error::into_internal_error)
    }

    fn from_value(
        _method: &str,
        value: serde_json::Value,
    ) -> Result<Self, agent_client_protocol::Error> {
        Ok(Self(agent_client_protocol::util::json_cast(&value)?))
    }
}

pub(super) fn client_supports_form_elicitation(
    args: &agent_client_protocol::schema::v1::InitializeRequest,
) -> bool {
    args.client_capabilities
        .elicitation
        .as_ref()
        .and_then(|elicitation| elicitation.form.as_ref())
        .is_some()
}

fn elicitation_response_from_acp(response: CreateElicitationResponse) -> ElicitationOutcome {
    match response.action {
        AcpElicitationAction::Accept(action) => {
            let content = serde_json::to_value(action.content.unwrap_or_default())
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
            ElicitationOutcome::Accept(content)
        }
        AcpElicitationAction::Decline => ElicitationOutcome::Decline,
        AcpElicitationAction::Cancel => ElicitationOutcome::Cancel,
        action => {
            warn!(?action, "Unsupported ACP elicitation action");
            ElicitationOutcome::Cancel
        }
    }
}

async fn finish_form_elicitation(
    agent: &Arc<Agent>,
    session_id: &str,
    elicitation_id: &str,
    response: ElicitationOutcome,
    recovered: bool,
) {
    match agent
        .submit_elicitation_response(session_id, elicitation_id, response.clone(), None)
        .await
    {
        Ok(true) => {}
        Ok(false) if recovered => {
            if let Err(error) = agent
                .record_recovered_elicitation_response(session_id, elicitation_id, &response)
                .await
            {
                warn!(
                    error = %error,
                    session_id = %session_id,
                    elicitation_id = %elicitation_id,
                    "Failed to record recovered ACP elicitation response"
                );
            }
        }
        Ok(false) => {
            warn!(
                session_id = %session_id,
                elicitation_id = %elicitation_id,
                "ACP elicitation response no longer has a live waiter"
            );
        }
        Err(error) => {
            warn!(
                error = %error,
                session_id = %session_id,
                elicitation_id = %elicitation_id,
                "Failed to submit ACP elicitation response"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovered_elicitation_metadata_declares_prompt_continuation() {
        let mut meta = Meta::new();
        add_elicitation_meta(&mut meta, "acp-provider:question-1", true, "prompt");

        assert_eq!(
            meta.get("goose"),
            Some(&serde_json::json!({
                "elicitationId": "acp-provider:question-1",
                "recovered": true,
                "continuation": "prompt"
            }))
        );
    }

    #[test]
    fn elicitation_metadata_preserves_existing_goose_fields() {
        let mut meta = Meta::from_iter([(
            "goose".to_string(),
            serde_json::json!({ "messageId": "message-1" }),
        )]);
        add_elicitation_meta(&mut meta, "question-1", false, "response");

        assert_eq!(meta["goose"]["messageId"], "message-1");
        assert_eq!(meta["goose"]["elicitationId"], "question-1");
        assert_eq!(meta["goose"]["recovered"], false);
        assert_eq!(meta["goose"]["continuation"], "response");
    }
}
