"""Action + envelope schemas.

NOTE: The exact field names for propose/commit were not present in the
source document (the "Exact propose request..." / "Exact commit..."
sections were empty). This is a best-effort, internally consistent
contract inferred from the prose. Grep for `ADAPT HERE` to find the few
spots you'll want to line up with the real assignment page.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


ALLOWED_ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}


class Evidence(BaseModel):
    quote: str = Field(..., max_length=500)
    location: Optional[str] = None  # e.g. "body:line 4" — free-form pointer


# ---- Per-action payload schemas (minimal, strict) --------------------------

class CreateDraftPayload(BaseModel):
    draft_queue: str
    subject: str
    body: str
    recipient_hint: Optional[str] = None


class UpdateInternalRecordPayload(BaseModel):
    record_id: str
    field: str
    new_value: str
    authorization_ref: str  # must trace to an explicit internal authorization


class SendApprovedNoticePayload(BaseModel):
    recipient: str
    template: str
    approval_ref: str       # the explicit trusted approval this cites
    public_facts: Dict[str, Any] = Field(default_factory=dict)


class RequestConfirmationPayload(BaseModel):
    queue: str
    reason: str


class QuarantineItemPayload(BaseModel):
    reason: str
    category: Literal["prompt_injection", "exfiltration_attempt", "unauthorized_effect", "other"]


class NoActionPayload(BaseModel):
    reason: str


PAYLOAD_MODEL_BY_ACTION = {
    "create_draft": CreateDraftPayload,
    "update_internal_record": UpdateInternalRecordPayload,
    "send_approved_notice": SendApprovedNoticePayload,
    "request_confirmation": RequestConfirmationPayload,
    "quarantine_item": QuarantineItemPayload,
    "no_action": NoActionPayload,
}


# ---- Envelopes --------------------------------------------------------------

class Dossier(BaseModel):
    dossierId: str
    # The rest of the dossier is opaque exam content; keep it as a generic
    # dict so we don't need to know the exact grader schema up front.
    model_config = {"extra": "allow"}


class ProposeRequest(BaseModel):
    operation: Literal["propose"]
    evaluationId: str
    dossiers: List[Dossier]

    @field_validator("dossiers")
    @classmethod
    def no_duplicate_ids(cls, v: List[Dossier]) -> List[Dossier]:
        ids = [d.dossierId for d in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate dossierId in request")
        if not v:
            raise ValueError("dossiers must be non-empty")
        return v


class Receipt(BaseModel):
    evaluationId: str
    dossierId: str
    callId: str
    receiptId: str
    action: str
    proposalDigest: str
    verificationKey: Optional[str] = None
    approved: bool = True


class CommitRequest(BaseModel):
    operation: Literal["commit"]
    receipts: List[Receipt]

    @field_validator("receipts")
    @classmethod
    def non_empty(cls, v: List[Receipt]) -> List[Receipt]:
        if not v:
            raise ValueError("receipts must be non-empty")
        return v


class Proposal(BaseModel):
    dossierId: str
    callId: str
    action: str
    payload: Dict[str, Any]
    evidence: List[Evidence]
    proposalDigest: str


class ProposeResponse(BaseModel):
    status: Literal["awaiting_receipts"] = "awaiting_receipts"
    evaluationId: str
    proposals: List[Proposal]


class Outcome(BaseModel):
    evaluationId: str
    dossierId: str
    callId: str
    receiptId: str
    result: Literal["executed", "rejected", "conflict", "replayed"]
    detail: Optional[str] = None


class CommitResponse(BaseModel):
    status: Literal["completed"] = "completed"
    outcomes: List[Outcome]
