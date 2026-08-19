use std::collections::{BTreeMap, HashSet};

use agent_client_protocol::schema::v1::{
    CreateElicitationRequest, CreateElicitationResponse, ElicitationAction,
    ElicitationContentValue, ElicitationFormMode, ElicitationSchema, ElicitationSessionScope,
    EnumOption, MultiSelectPropertySchema, StringPropertySchema,
};
use agent_client_protocol::{JsonRpcRequest, JsonRpcResponse};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "cursor/ask_question", response = CursorAskQuestionResponse)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CursorAskQuestionRequest {
    pub(crate) tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    pub(crate) questions: Vec<CursorQuestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CursorQuestion {
    pub(crate) id: String,
    pub(crate) prompt: String,
    pub(crate) options: Vec<CursorQuestionOption>,
    #[serde(default)]
    pub(crate) allow_multiple: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CursorQuestionOption {
    pub(crate) id: String,
    pub(crate) label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
pub(crate) struct CursorAskQuestionResponse {
    pub(crate) outcome: CursorAskQuestionOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "lowercase")]
pub(crate) enum CursorAskQuestionOutcome {
    Answered { answers: Vec<CursorQuestionAnswer> },
    Skipped,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CursorQuestionAnswer {
    pub(crate) question_id: String,
    pub(crate) selected_option_ids: Vec<String>,
}

impl CursorAskQuestionRequest {
    pub(crate) fn to_elicitation_request(
        &self,
        session_id: &str,
    ) -> Option<CreateElicitationRequest> {
        if self.questions.is_empty() {
            return None;
        }

        let mut question_ids = HashSet::new();
        let mut schema = ElicitationSchema::new().title(self.title.clone());

        for question in &self.questions {
            if question.id.is_empty()
                || question.options.is_empty()
                || !question_ids.insert(question.id.as_str())
            {
                return None;
            }

            let mut option_ids = HashSet::new();
            let options = question
                .options
                .iter()
                .map(|option| {
                    option_ids
                        .insert(option.id.as_str())
                        .then(|| EnumOption::new(option.id.clone(), option.label.clone()))
                })
                .collect::<Option<Vec<_>>>()?;

            schema = if question.allow_multiple {
                schema.property(
                    question.id.clone(),
                    MultiSelectPropertySchema::titled(options).title(question.prompt.clone()),
                    true,
                )
            } else {
                schema.property(
                    question.id.clone(),
                    StringPropertySchema::new()
                        .title(question.prompt.clone())
                        .one_of(options),
                    true,
                )
            };
        }

        let message = self.title.clone().unwrap_or_else(|| {
            if self.questions.len() == 1 {
                "Cursor has a question".to_string()
            } else {
                "Cursor has a few questions".to_string()
            }
        });

        Some(CreateElicitationRequest::new(
            ElicitationFormMode::new(ElicitationSessionScope::new(session_id.to_string()), schema),
            message,
        ))
    }

    pub(crate) fn response_from_elicitation(
        &self,
        response: CreateElicitationResponse,
    ) -> CursorAskQuestionResponse {
        let outcome = match response.action {
            ElicitationAction::Accept(accept) => {
                let Some(content) = accept.content else {
                    return CursorAskQuestionResponse::cancelled();
                };
                let Some(answers) = self.answers_from_content(&content) else {
                    return CursorAskQuestionResponse::cancelled();
                };
                CursorAskQuestionOutcome::Answered { answers }
            }
            ElicitationAction::Decline => CursorAskQuestionOutcome::Skipped,
            ElicitationAction::Cancel => CursorAskQuestionOutcome::Cancelled,
            _ => CursorAskQuestionOutcome::Cancelled,
        };

        CursorAskQuestionResponse { outcome }
    }

    fn answers_from_content(
        &self,
        content: &BTreeMap<String, ElicitationContentValue>,
    ) -> Option<Vec<CursorQuestionAnswer>> {
        self.questions
            .iter()
            .map(|question| {
                let selected_option_ids = match content.get(&question.id)? {
                    ElicitationContentValue::String(value) if !question.allow_multiple => {
                        vec![value.clone()]
                    }
                    ElicitationContentValue::StringArray(values) if question.allow_multiple => {
                        values.clone()
                    }
                    _ => return None,
                };

                Some(CursorQuestionAnswer {
                    question_id: question.id.clone(),
                    selected_option_ids,
                })
            })
            .collect()
    }
}

impl CursorAskQuestionResponse {
    pub(crate) fn cancelled() -> Self {
        Self {
            outcome: CursorAskQuestionOutcome::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::{
        ElicitationAcceptAction, ElicitationPropertySchema, MultiSelectItems,
    };

    fn request() -> CursorAskQuestionRequest {
        CursorAskQuestionRequest {
            tool_call_id: "tool-1".to_string(),
            title: Some("Shape the implementation".to_string()),
            questions: vec![
                CursorQuestion {
                    id: "direction".to_string(),
                    prompt: "Which direction?".to_string(),
                    options: vec![
                        CursorQuestionOption {
                            id: "native".to_string(),
                            label: "Native ACP".to_string(),
                        },
                        CursorQuestionOption {
                            id: "prose".to_string(),
                            label: "Prose".to_string(),
                        },
                    ],
                    allow_multiple: false,
                },
                CursorQuestion {
                    id: "priorities".to_string(),
                    prompt: "What matters?".to_string(),
                    options: vec![
                        CursorQuestionOption {
                            id: "quality".to_string(),
                            label: "Quality".to_string(),
                        },
                        CursorQuestionOption {
                            id: "speed".to_string(),
                            label: "Speed".to_string(),
                        },
                    ],
                    allow_multiple: true,
                },
            ],
        }
    }

    #[test]
    fn converts_multiple_questions_without_inventing_options() {
        let request = request();
        let elicitation = request.to_elicitation_request("cursor-session").unwrap();
        let ElicitationFormMode {
            requested_schema, ..
        } = match elicitation.mode {
            agent_client_protocol::schema::v1::ElicitationMode::Form(form) => form,
            _ => panic!("expected form elicitation"),
        };

        let ElicitationPropertySchema::String(direction) =
            &requested_schema.properties["direction"]
        else {
            panic!("expected single-select property");
        };
        assert_eq!(
            direction
                .one_of
                .as_ref()
                .unwrap()
                .iter()
                .map(|option| option.value.as_str())
                .collect::<Vec<_>>(),
            vec!["native", "prose"]
        );

        let ElicitationPropertySchema::Array(priorities) =
            &requested_schema.properties["priorities"]
        else {
            panic!("expected multi-select property");
        };
        let MultiSelectItems::Titled(items) = &priorities.items else {
            panic!("expected titled multi-select items");
        };
        assert_eq!(
            items
                .options
                .iter()
                .map(|option| option.value.as_str())
                .collect::<Vec<_>>(),
            vec!["quality", "speed"]
        );
    }

    #[test]
    fn maps_accepted_values_back_to_cursor_ids_in_question_order() {
        let request = request();
        let response = CreateElicitationResponse::new(ElicitationAction::Accept(
            ElicitationAcceptAction::new().content(BTreeMap::from([
                (
                    "priorities".to_string(),
                    ElicitationContentValue::StringArray(vec![
                        "quality".to_string(),
                        "speed".to_string(),
                    ]),
                ),
                (
                    "direction".to_string(),
                    ElicitationContentValue::String("native".to_string()),
                ),
            ])),
        ));

        assert_eq!(
            request.response_from_elicitation(response).outcome,
            CursorAskQuestionOutcome::Answered {
                answers: vec![
                    CursorQuestionAnswer {
                        question_id: "direction".to_string(),
                        selected_option_ids: vec!["native".to_string()],
                    },
                    CursorQuestionAnswer {
                        question_id: "priorities".to_string(),
                        selected_option_ids: vec!["quality".to_string(), "speed".to_string()],
                    },
                ],
            }
        );
    }

    #[test]
    fn rejects_ambiguous_question_and_option_ids() {
        let mut duplicate_questions = request();
        duplicate_questions.questions[1].id = "direction".to_string();
        assert!(duplicate_questions
            .to_elicitation_request("cursor-session")
            .is_none());

        let mut duplicate_options = request();
        duplicate_options.questions[0].options[1].id = "native".to_string();
        assert!(duplicate_options
            .to_elicitation_request("cursor-session")
            .is_none());
    }

    #[test]
    fn maps_decline_and_cancel_to_cursor_outcomes() {
        let request = request();
        assert_eq!(
            request
                .response_from_elicitation(CreateElicitationResponse::new(
                    ElicitationAction::Decline,
                ))
                .outcome,
            CursorAskQuestionOutcome::Skipped
        );
        assert_eq!(
            request
                .response_from_elicitation(CreateElicitationResponse::new(
                    ElicitationAction::Cancel,
                ))
                .outcome,
            CursorAskQuestionOutcome::Cancelled
        );
    }
}
