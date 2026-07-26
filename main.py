import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import store
from .canon import canonical_json, content_fingerprint, call_id_for, input_digest
from .llm import decide
from .schemas import ProposeRequest, CommitRequest, ALLOWED_ACTIONS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mailroom")

MAX_BODY_BYTES = 2 * 1024 * 1024      # generous request bound; response is capped separately
MAX_RESPONSE_BYTES = 512 * 1024

app = FastAPI()


@app.on_event("startup")
def _startup():
    store.init_db()


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": code, "message": message},
        media_type="application/json",
    )


def _bounded_json_response(payload: Dict[str, Any]) -> JSONResponse:
    body = canonical_json(payload)
    if len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        # Should not happen in practice; fail loudly rather than truncate silently.
        return _err(500, "response_too_large", "internal: response exceeded 512 KiB bound")
    return Response(content=body, status_code=200, media_type="application/json")


# ---- propose -----------------------------------------------------------------

def _handle_propose(raw: Dict[str, Any]):
    try:
        req = ProposeRequest.model_validate(raw)
    except ValidationError as e:
        return _err(422, "invalid_propose_request", str(e))

    dossier_ids_sorted = sorted(d.dossierId for d in req.dossiers)
    dossier_set_hash = content_fingerprint(
        [ {**d.model_dump()} for d in sorted(req.dossiers, key=lambda x: x.dossierId) ]
    )

    existing = store.get_evaluation(req.evaluationId)
    if existing is not None:
        if existing["dossierSetHash"] != dossier_set_hash:
            return _err(409, "evaluation_conflict",
                        "evaluationId already used with different dossier content")
        # exact replay: return the persisted proposals verbatim
        return _bounded_json_response({
            "status": "awaiting_receipts",
            "evaluationId": req.evaluationId,
            "proposals": existing["proposals"],
        })

    proposals = []
    for d in req.dossiers:
        body = d.model_dump(exclude={"dossierId"})
        fp = content_fingerprint(body)
        cached = store.get_cached_decision(d.dossierId, fp)
        if cached is not None:
            decision = cached
        else:
            raw_decision = decide(d.dossierId, body)
            call_id = call_id_for(d.dossierId, fp)
            digest = input_digest(d.dossierId, raw_decision["action"], raw_decision["target"],
                                   raw_decision["payload"], raw_decision["evidence"])
            store.put_cached_decision(d.dossierId, fp, call_id, raw_decision["action"],
                                       raw_decision["target"], raw_decision["payload"],
                                       raw_decision["evidence"], digest)
            decision = {
                "callId": call_id,
                "action": raw_decision["action"],
                "target": raw_decision["target"],
                "payload": raw_decision["payload"],
                "evidence": raw_decision["evidence"],
                "inputDigest": digest,
            }
        proposals.append({
            "dossierId": d.dossierId,
            "callId": decision["callId"],
            "action": decision["action"],
            "target": decision["target"],
            "payload": decision["payload"],
            "evidence": decision["evidence"],
            "inputDigest": decision["inputDigest"],
        })

    store.put_evaluation(req.evaluationId, dossier_set_hash, proposals)

    return _bounded_json_response({
        "status": "awaiting_receipts",
        "evaluationId": req.evaluationId,
        "proposals": proposals,
    })


# ---- commit ------------------------------------------------------------------

def _execute_effect(action: str, target: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Perform the (simulated) side effect for an approved action.

    Only ever called after: schema validation, evaluation lookup, and
    receipt/digest verification all succeeded. Real deployments would wire
    these into an actual draft store / CRM / mailer; here we record an
    auditable effect object rather than assume a live external system.
    """
    return {"action": action, "applied": True, "target": target, "payload": payload}


def _handle_commit(raw: Dict[str, Any]):
    try:
        req = CommitRequest.model_validate(raw)
    except ValidationError as e:
        return _err(422, "invalid_commit_request", str(e))

    outcomes = []
    for r in req.receipts:
        evaluation = store.get_evaluation(r.evaluationId)
        if evaluation is None:
            outcomes.append({
                "evaluationId": r.evaluationId, "dossierId": r.dossierId,
                "callId": r.callId, "receiptId": r.receiptId,
                "result": "rejected", "detail": "unknown evaluationId",
            })
            continue

        matching = next((p for p in evaluation["proposals"] if p["callId"] == r.callId), None)
        if matching is None:
            outcomes.append({
                "evaluationId": r.evaluationId, "dossierId": r.dossierId,
                "callId": r.callId, "receiptId": r.receiptId,
                "result": "rejected", "detail": "unknown callId for this evaluation",
            })
            continue

        if matching["action"] != r.action or matching["inputDigest"] != r.inputDigest:
            outcomes.append({
                "evaluationId": r.evaluationId, "dossierId": matching["dossierId"],
                "callId": r.callId, "receiptId": r.receiptId,
                "result": "rejected", "detail": "action/inputDigest mismatch vs persisted proposal",
            })
            continue

        # idempotent replay check
        prior = store.get_receipt(r.evaluationId, r.callId)
        if prior is not None:
            outcomes.append({
                "evaluationId": r.evaluationId, "dossierId": matching["dossierId"],
                "callId": r.callId, "receiptId": prior["receiptId"],
                "result": "replayed" if prior["result"] == "executed" else prior["result"],
                "detail": prior["detail"],
            })
            continue

        if not r.approved or matching["action"] not in ALLOWED_ACTIONS:
            store.put_receipt(r.evaluationId, r.callId, r.receiptId, r.verificationKey,
                               "rejected", "not approved", None)
            outcomes.append({
                "evaluationId": r.evaluationId, "dossierId": matching["dossierId"],
                "callId": r.callId, "receiptId": r.receiptId,
                "result": "rejected", "detail": "not approved by grader",
            })
            continue

        effect = _execute_effect(matching["action"], matching["target"], matching["payload"])
        store.put_receipt(r.evaluationId, r.callId, r.receiptId, r.verificationKey,
                           "executed", None, effect)
        outcomes.append({
            "evaluationId": r.evaluationId, "dossierId": matching["dossierId"],
            "callId": r.callId, "receiptId": r.receiptId,
            "result": "executed", "detail": None,
        })

    response = {"status": "completed", "outcomes": outcomes}
    return _bounded_json_response(response)


# ---- single public endpoint ---------------------------------------------------

@app.post("/v1/mailroom/actions")
async def mailroom_actions(request: Request):
    body_bytes = await request.body()
    if len(body_bytes) > MAX_BODY_BYTES:
        return _err(413, "payload_too_large", "request body exceeds bound")

    try:
        raw = json.loads(body_bytes)
    except json.JSONDecodeError:
        return _err(400, "malformed_json", "request body is not valid JSON")

    if not isinstance(raw, dict) or "operation" not in raw:
        return _err(400, "missing_operation", "request must include an 'operation' field")

    op = raw.get("operation")
    if op == "propose":
        return _handle_propose(raw)
    if op == "commit":
        return _handle_commit(raw)
    return _err(400, "invalid_operation", f"unknown operation: {op!r}")


@app.get("/healthz")
def healthz():
    return {"ok": True}
