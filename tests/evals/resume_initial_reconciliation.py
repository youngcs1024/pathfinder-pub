"""A-only bounded timeout reservations and independently replayable scoring calls."""

import json
from decimal import Decimal
from time import monotonic

from app.llm.factory import LLMProviderError
from app.llm.invocations import LOCKED_CHAT_MODEL, chat_request_hash
from app.llm.ports import ChatMessage, ChatModelResult
from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiment_budget import CHAT_HEADROOM_CNY, ExperimentRecorder, admit
from tests.evals.resume_experiments import preserve
from tests.evals.resume_live_budget import ledger_identity

SCORE_NODES = frozenset(
    f"score_{stage}" for stage in ("initial", "correction", "review", "unit", "unit_correction")
)
POLICY = {
    "version": "a_terminal_score_timeout_v1",
    "maximum_new_timeouts": 2,
    "reservation_cny": str(CHAT_HEADROOM_CNY),
    "authorization": (
        "2026-09-29 user approved at most two A scoring terminal timeouts including "
        "existing timeout; original shared caps remain"
    ),
}


def call_identity(messages, metadata):
    return quality_identity_digest(
        {
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": [],
            "metadata": metadata,
        }
    )


def safe_call_path(root, path):
    relative = path.relative_to(root)
    require(relative.parts[0] == "a" and not path.is_symlink(), "unsafe_score_call")
    require(path.resolve().is_relative_to((root / "a").resolve()), "unsafe_score_call")
    return str(relative)


class ARecorder(ExperimentRecorder):
    active_score_call = None

    def enable_policy(self):
        preserve(self.root / "a-score-timeout-policy.json", {**POLICY, "binding": self.binding})

    def policy_enabled(self):
        path = self.root / "a-score-timeout-policy.json"
        if not path.exists():
            require(not (self.root / "a-score-timeouts").exists(), "a_policy_removed")
            return False
        require(read_private_json(path) == {**POLICY, "binding": self.binding}, "a_policy_changed")
        return True

    def link_call(self, row, path, identity):
        require(row.graph_node in SCORE_NODES, "not_a_score_call")
        started = read_private_json(path)
        require(started["identity"] == identity, "score_identity_changed")
        directory = self.root / "a-score-calls"
        directory.mkdir(mode=0o700, exist_ok=True)
        value = {
            "binding": quality_identity_digest(self.binding),
            "invocation_id": str(row.id),
            "request_hash": row.request_hash,
            "graph_node": row.graph_node,
            "started_path": safe_call_path(self.root, path),
            "started_digest": quality_identity_digest(started),
            "identity": identity,
        }
        receipt = directory / f"{row.id}.json"
        if receipt.exists():
            old = read_private_json(receipt)
            require(
                all(
                    old[k] == value[k]
                    for k in ("binding", "invocation_id", "request_hash", "graph_node", "identity")
                ),
                "score_call_link_changed",
            )
        else:
            preserve(receipt, value)

    async def prepare(self, attempt):
        await super().prepare(attempt)
        if self.active_score_call is not None:
            row = next(r for r in await self.rows() if r.id == attempt.invocation_id)
            self.link_call(row, *self.active_score_call)

    async def measurement(self):
        usage = await super().measurement()
        if not self.policy_enabled():
            return usage
        rows = await self.rows()
        receipts = self.root / "a-score-timeouts"
        eligible = []
        for row in rows:
            link = self.root / "a-score-calls" / f"{row.id}.json"
            if not (
                row.status == "failed"
                and row.error_category == "provider_timeout"
                and row.estimated_cost is None
                and row.token_usage is None
                and row.provider == "qwen"
                and row.invocation_kind == "chat"
                and row.graph_node in SCORE_NODES
                and link.exists()
            ):
                continue
            saved = read_private_json(link)
            require(
                saved["binding"] == quality_identity_digest(self.binding)
                and saved["request_hash"] == row.request_hash
                and saved["graph_node"] == row.graph_node
                and saved["invocation_id"] == str(row.id),
                "a_timeout_link_changed",
            )
            started_path = self.root / saved["started_path"]
            safe_call_path(self.root, started_path)
            require(
                quality_identity_digest(read_private_json(started_path)) == saved["started_digest"],
                "a_timeout_start_changed",
            )
            eligible.append((row, saved))
        require(len(eligible) <= POLICY["maximum_new_timeouts"], "a_timeout_limit")
        expected = {}
        for row, link in eligible:
            value = {
                "policy_digest": quality_identity_digest({**POLICY, "binding": self.binding}),
                "call": link,
                "ledger": ledger_identity(row),
                "reserved_cost_cny": str(CHAT_HEADROOM_CNY),
            }
            receipts.mkdir(mode=0o700, exist_ok=True)
            preserve(receipts / f"{row.id}.json", value)
            expected[str(row.id)] = value
        if receipts.exists():
            require(
                {p.stem for p in receipts.glob("*.json")} == expected.keys(),
                "a_timeout_receipt_changed",
            )
        extra = CHAT_HEADROOM_CNY * len(expected)
        usage["a_score_timeouts"] = expected
        usage["a_reserved_unknown_attempts"] = len(expected)
        usage["reserved_unknown_attempts"] += len(expected)
        usage["reserved_cost_cny"] = str(Decimal(usage["reserved_cost_cny"]) + extra)
        usage["budget_occupied_cny"] = str(Decimal(usage["budget_occupied_cny"]) + extra)
        usage["remaining_admission_cny"] = str(Decimal(usage["remaining_admission_cny"]) - extra)
        from tests.evals.resume_initial_interruption import reservation

        return reservation(self.root, self.binding, rows, usage)

    async def failure_measurement(self):
        # Diagnostic only: never used by admission. Keep the hard two-timeout rejection.
        from tests.evals.product_acceptance_contracts import AcceptanceError

        try:
            return await self.measurement()
        except AcceptanceError as exc:
            usage = await super().measurement()
            return {
                **usage,
                "admission": "BLOCKED",
                "a_reservation_validation_error": str(exc),
                "reservation_audit_complete": False,
            }

    async def finalize(self, attempt, outcome):
        if self.active_score_call is None:
            return await super().finalize(attempt, outcome)
        try:
            await self.delegate.finalize(attempt, outcome)
            # Budget/accounting stays fail-closed. The scoring journal checks source after
            # saving a successful response; source changes cannot discard known output.
            await self.checked_usage(after=True)
        except BaseException:
            self.stopped = True
            raise

    async def check_admission(self, *, after=False, kind="chat"):
        self.source_check()
        return await self.checked_usage(after=after, kind=kind)

    async def checked_usage(self, *, after=False, kind="chat"):
        usage = await self.measurement()
        # Reuse the original fail-closed budget arithmetic after separating validated A exceptions.
        checked = dict(usage)
        checked["unfinished"] -= usage.get("reserved_unfinished_attempts", 0)
        count = usage.get("a_reserved_unknown_attempts", 0)
        checked["unknown_cost"] -= count
        checked["unknown_usage"] -= count
        checked["reserved_unknown_attempts"] -= count
        extra = CHAT_HEADROOM_CNY * count
        checked["reserved_cost_cny"] = str(Decimal(usage["reserved_cost_cny"]) - extra)
        checked["known_cost_cny"] = str(Decimal(usage["known_cost_cny"]) + extra)
        admit(checked, self.budget, after=after, kind=kind)
        return usage


async def reconcile_started(recorder, path, messages, metadata, expected_id=None):
    started = read_private_json(path)
    identity = call_identity(messages, metadata)
    require(started["identity"] == identity, "score_identity_changed")
    request_hash = chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=())
    candidates = [
        r
        for r in await recorder.audit()
        if str(r.id) not in started["before"]["invocation_ids"]
        and r.request_hash == request_hash
        and r.graph_node == metadata["graph_node"]
        and (expected_id is None or str(r.id) == expected_id)
    ]
    require(len(candidates) == 1, "score_reconciliation_ambiguous")
    row = candidates[0]
    require(
        row.status == "failed"
        and row.error_category == "provider_timeout"
        and row.token_usage is None
        and row.estimated_cost is None,
        "score_reconciliation_not_timeout",
    )
    recorder.link_call(row, path, identity)
    usage = await recorder.check_admission(after=True)
    require(str(row.id) in usage.get("a_score_timeouts", {}), "score_timeout_not_reserved")
    return str(row.id)


async def scoring_call(factory, recorder, context, directory, payload, stage, prompt):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    messages = (
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    )
    metadata = {
        "task": "resume_initial_review",
        "graph_node": f"score_{stage}",
        "prompt_version": quality_identity_digest(prompt),
    }
    identity = call_identity(messages, metadata)
    for number in range(2):
        stem = f"call-{number:02}"
        started = directory / f"{stem}-started.json"
        response = directory / f"{stem}-response.json"
        terminal = directory / f"{stem}-terminal.json"
        if response.exists():
            saved = read_private_json(response)
            require(saved["identity"] == identity, "response_identity_changed")
            await recorder.check_admission(after=True)
            return ChatModelResult.model_validate_json(json.dumps(saved["response"]))
        if started.exists():
            saved_terminal = read_private_json(terminal) if terminal.exists() else None
            if saved_terminal is not None:
                require(
                    saved_terminal["status"] == "reserved_timeout"
                    and saved_terminal["identity"] == identity
                    and len(saved_terminal["invocation_ids"]) == 1,
                    "score_terminal_changed",
                )
            invocation_id = await reconcile_started(
                recorder,
                started,
                messages,
                metadata,
                saved_terminal["invocation_ids"][0] if saved_terminal else None,
            )
            preserve(
                terminal,
                {
                    "status": "reserved_timeout",
                    "identity": identity,
                    "invocation_ids": [invocation_id],
                    "stage": stage,
                    "retry": number,
                },
            )
            continue
        before = await recorder.check_admission()
        preserve(started, {"identity": identity, "before": before})
        recorder.active_score_call = (started, identity)
        began = monotonic()
        try:
            result = await factory.create_chat_model(context).invoke(messages, (), metadata)
        except BaseException as exc:
            rows = await recorder.audit()
            ids = sorted(str(r.id) for r in rows if str(r.id) not in before["invocation_ids"])
            if isinstance(exc, LLMProviderError) and exc.category == "provider_timeout":
                after = await recorder.check_admission(after=True)
                require(
                    len(ids) == 1 and ids[0] in after.get("a_score_timeouts", {}),
                    "score_failure_requires_reconciliation",
                )
                preserve(
                    terminal,
                    {
                        "status": "reserved_timeout",
                        "identity": identity,
                        "invocation_ids": ids,
                        "stage": stage,
                        "retry": number,
                    },
                )
                continue
            preserve(
                terminal,
                {
                    "status": "failed",
                    "identity": identity,
                    "invocation_ids": ids,
                    "stage": stage,
                    "retry": number,
                },
            )
            raise
        finally:
            recorder.active_score_call = None
        # Save the returned response before later admission/source checks can interrupt journaling.
        rows = await recorder.rows()
        ids = sorted(str(r.id) for r in rows if str(r.id) not in before["invocation_ids"])
        preserve(
            response,
            {
                "identity": identity,
                "response": result.model_dump(mode="json"),
                "elapsed_seconds": monotonic() - began,
                "invocation_ids": ids,
            },
        )
        preserve(
            terminal,
            {
                "status": "completed",
                "identity": identity,
                "invocation_ids": ids,
                "stage": stage,
                "retry": number,
            },
        )
        await recorder.check_admission(after=True)
        return result
    # Both bounded attempts timed out; move to the next semantic stage, never fabricate a score.
    return ChatModelResult(content=None, finish_status="incomplete")


async def reconcile_legacy(recorder, origin):
    from tests.evals import resume_initial_scoring as scoring

    recorder.enable_policy()
    reconciled = []
    for path in sorted(origin.glob("formal/*/a-score-v4/batch-*/*/call-00-started.json")):
        if path.with_name("call-00-response.json").exists():
            continue
        stage_path = path.parent
        protocol = read_private_json(stage_path.parent.parent / "protocol.json")
        index = int(stage_path.parent.name.removeprefix("batch-"))
        # The stored JSON is key-sorted; regenerate the original wire insertion order.
        # Never guess a request hash by accepting equivalent but different JSON strings.
        from tests.evals.resume_experiment_scoring import profile_evidence
        from tests.evals.resume_experiments import load_inputs

        sample_path = stage_path.parent.parent.parent
        generated = read_private_json(sample_path / "generation.json")
        case_id = generated["sample"]["case_id"]
        annotation = read_private_json(origin / "formal/annotations.json")[case_id]
        case = next(c for c in load_inputs(recorder.root).cases if c.case_id == case_id)
        packets, mapping = scoring.packets_for(
            generated["output"]["content"],
            read_private_json(recorder.root / "inputs/facts.json")["facts"],
            profile_evidence(read_private_json(recorder.root / "inputs/profile.json")),
            annotation,
            case.jd,
        )
        require(
            packets == protocol["packets"] and mapping == protocol["mapping"],
            "legacy_score_packet_changed",
        )
        payload = dict(packets[index])
        stage = stage_path.name
        if stage == "correction":
            prev = stage_path.parent / "initial"
            payload.update(
                previous_response=read_private_json(prev / "call-00-response.json")["response"][
                    "content"
                ]
                or "",
                validation_error=read_private_json(prev / "validation.json")["error"],
            )
        elif stage == "review":
            payload["review_instruction"] = (
                "Independently adjudicate the original evidence and candidate; "
                "do not assume any earlier assessment."
            )
        prompt = scoring.prompt_for(payload)
        messages = (
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        )
        metadata = {
            "task": "resume_initial_review",
            "graph_node": f"score_{stage}",
            "prompt_version": quality_identity_digest(prompt),
        }
        reconciled.append(await reconcile_started(recorder, path, messages, metadata))
    return {
        "status": "PASS",
        "reconciled": reconciled,
        "usage": await recorder.check_admission(after=True),
    }
