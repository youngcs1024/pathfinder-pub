"""E8.3 manual product evidence, separate from production and accepted baselines."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from app.domain.runs import CURRENT_GRAPH_VERSION
from app.llm.invocations import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from tests.evals.contracts import EvalContractModel, EvalDigest
from tests.evals.quality_dataset import load_quality_dataset, quality_identity_digest
from tests.evals.quality_experiment_binding import (
    ROOT,
    encoded,
    file_inventory,
    git,
    no_links,
    read_json,
    source_identity,
    write_new,
)
from tests.evals.quality_generation_support import checked_directory, secret_markers

DATASET = ROOT / "evals/datasets/quality_v1"
CASES = ("mixed_alpha", "gap_alpha")


class AcceptanceError(Exception):
    """Only fixed categories, never exception payloads, leave the command boundary."""


def require(condition, category):
    if not condition:
        raise AcceptanceError(category)


class Budget(EvalContractModel):
    cost_admission_budget_cny: Decimal = Decimal("5.00")
    unknown_attempt_reserve_cny: Decimal = Decimal("0.10")
    provider_attempt_cap: Literal[100] = 100
    input_token_cap: Literal[500000] = 500000
    output_token_cap: Literal[200000] = 200000


class Manifest(EvalContractModel):
    artifact_kind: Literal["e83_product_manifest_v1"] = "e83_product_manifest_v1"
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: dict[str, EvalDigest]
    dataset_digest: EvalDigest
    case_ids: tuple[Literal["mixed_alpha"], Literal["gap_alpha"]] = CASES
    repeat: Literal[1] = 1
    graph_version: Literal["pathfinder-research-v6"] = "pathfinder-research-v6"
    chat_model: str = LOCKED_CHAT_MODEL
    embedding_model: str = LOCKED_EMBEDDING_MODEL
    embedding_dimension: Literal[1536] = 1536
    pricing_version: str = QWEN_BEIJING_PRICING_VERSION
    web_mode: Literal["frozen_fixture"] = "frozen_fixture"
    auth_mode: Literal["fake"] = "fake"
    trace_mode: Literal["off"] = "off"
    budget: Budget = Budget()


def bound_files():
    return (
        git(
            ROOT,
            "ls-files",
            "src",
            "migrations",
            "tests/evals",
            "evals",
            "uv.lock",
            "pyproject.toml",
            ".python-version",
            "alembic.ini",
        )
        .decode()
        .splitlines()
    )


def prepare_manifest():
    sha = git(ROOT, "rev-parse", "HEAD").decode().strip()
    return Manifest(
        source_sha=sha,
        source_tree=source_identity(ROOT, sha),
        files=file_inventory(ROOT, bound_files()),
        dataset_digest=load_quality_dataset(DATASET).manifest_digest,
    )


def verify_manifest(manifest):
    require(manifest.budget == Budget(), "budget_changed")
    require(
        manifest.graph_version == CURRENT_GRAPH_VERSION
        and manifest.chat_model == LOCKED_CHAT_MODEL
        and manifest.embedding_model == LOCKED_EMBEDDING_MODEL
        and manifest.pricing_version == QWEN_BEIJING_PRICING_VERSION,
        "version_changed",
    )
    require(source_identity(ROOT, manifest.source_sha) == manifest.source_tree, "source_changed")
    require(set(manifest.files) == set(bound_files()), "source_changed")
    require(file_inventory(ROOT, manifest.files) == manifest.files, "source_changed")
    require(
        load_quality_dataset(DATASET).manifest_digest == manifest.dataset_digest, "data_changed"
    )


def new_directory(path):
    path = no_links(path)
    require(not path.is_relative_to(ROOT) and not path.exists(), "unsafe_directory")
    path.mkdir(mode=0o700, parents=False)
    checked_directory(path)
    return path


def publish(path, value, markers=()):
    raw = encoded(value).decode()
    require(
        not any(marker and marker in raw for marker in (*markers, *secret_markers())),
        "unsafe_artifact",
    )
    return write_new(path, value)


def read_private_json(path):
    path = no_links(path)
    info = path.stat()
    require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0, "unsafe_artifact")
    checked_directory(path.parent)
    return read_json(path)


def review_digest(review):
    return quality_identity_digest(review)


def validate_review(review, *, now=None):
    from app.api.schemas.action_intents import ActionIntentReviewResponse

    value = ActionIntentReviewResponse.model_validate_json(encoded(review))
    request = value.approval_request
    require(
        value.status == "proposed" and request.status == "pending" and value.decision is None,
        "approval_not_pending",
    )
    require(request.expires_at > (now or datetime.now(UTC)), "approval_expired")
    require(
        (
            value.args_digest,
            value.target_digest,
            value.approval_binding_digest,
            value.approval_binding_version,
        )
        == (
            request.args_digest,
            request.target_digest,
            request.approval_binding_digest,
            request.approval_binding_version,
        ),
        "approval_binding_mismatch",
    )
    return value


class Decision(EvalContractModel):
    artifact_kind: Literal["e83_human_decision_v1"] = "e83_human_decision_v1"
    case_id: Literal["mixed_alpha", "gap_alpha"]
    review_digest: EvalDigest
    decision: Literal["approve", "reject"]
    reviewed_exact_content: Literal[True]
    acknowledged_at: datetime


def write_decision(root, case_id, decision, expected_digest):
    require(case_id in CASES, "invalid_case")
    require(not (root / "report.json").exists(), "execution_closed")
    review = read_private_json(root / f"{case_id}-review.json")
    validate_review(review)
    require(review_digest(review) == expected_digest, "review_changed")
    value = Decision(
        case_id=case_id,
        decision=decision,
        review_digest=expected_digest,
        reviewed_exact_content=True,
        acknowledged_at=datetime.now(UTC),
    )
    publish(root / f"{case_id}-decision.json", value)
