"""Versioned execution payloads, independent of HTTP, persistence and runtime threads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from app.domain.research import ResearchOutputV1, ResearchOutputV2, ResearchRequestV1
from app.domain.resume_generation import GenerationCandidateV1
from app.domain.resume_revision import RevisionCandidateV1


class RunMode(StrEnum):
    RESEARCH = "research"
    APPLICATION = "application"
    MATERIAL_PREPARATION = "material_preparation"
    RESUME_GENERATION = "resume_generation"
    RESUME_REVISION = "resume_revision"


LEGACY_GRAPH_VERSION = "pathfinder-research-v6"
LEGACY_RUN_MODES = frozenset({RunMode.RESEARCH, RunMode.APPLICATION})


class _ResumeEnvelopeV1[Payload: BaseModel](BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )
    schema_version: Literal[1] = 1
    mode: Literal[RunMode.MATERIAL_PREPARATION, RunMode.RESUME_GENERATION, RunMode.RESUME_REVISION]
    payload: Payload

    @model_validator(mode="after")
    def require_concrete_payload(self):
        schema = type(self).model_fields["payload"].annotation
        if (
            not isinstance(schema, type)
            or not issubclass(schema, BaseModel)
            or schema is BaseModel
            or type(self.payload) is not schema
            or schema.model_config.get("extra") != "forbid"
            or schema.model_config.get("frozen") is not True
            or schema.model_config.get("strict") is not True
        ):
            raise ValueError("execution payload requires a concrete strict immutable schema")
        return self


class ResumeRunInputV1[Payload: BaseModel](_ResumeEnvelopeV1[Payload]):
    """Payload schemas are introduced by their first real consumer, not reserved here."""


class ResumeRunOutputV1[Payload: BaseModel](_ResumeEnvelopeV1[Payload]):
    """A typed result, never an unvalidated dictionary or a successful placeholder."""


class MaterialPreparationInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    import_id: UUID
    project_id: UUID
    source_ids: tuple[UUID, ...]


class MaterialPreparationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    import_id: UUID
    snapshot_ids: tuple[UUID, ...]
    file_count: int
    document_count: int


class MaterialPreparationRunInputV1(ResumeRunInputV1[MaterialPreparationInputV1]):
    mode: Literal[RunMode.MATERIAL_PREPARATION] = RunMode.MATERIAL_PREPARATION
    payload: MaterialPreparationInputV1


class MaterialPreparationRunOutputV1(ResumeRunOutputV1[MaterialPreparationResultV1]):
    mode: Literal[RunMode.MATERIAL_PREPARATION] = RunMode.MATERIAL_PREPARATION
    payload: MaterialPreparationResultV1


class ResumeGenerationInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: UUID


class ResumeGenerationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: UUID
    version_id: UUID | None
    artifact_id: UUID | None
    outcome: Literal["draft", "needs_input"]
    questions: tuple[str, ...]


class ResumeGenerationRunInputV1(ResumeRunInputV1[ResumeGenerationInputV1]):
    mode: Literal[RunMode.RESUME_GENERATION] = RunMode.RESUME_GENERATION
    payload: ResumeGenerationInputV1


class ResumeGenerationCandidateOutputV1(ResumeRunOutputV1[GenerationCandidateV1]):
    """Worker-only result; the completion transaction converts it to a receipt."""

    mode: Literal[RunMode.RESUME_GENERATION] = RunMode.RESUME_GENERATION
    payload: GenerationCandidateV1


class ResumeGenerationRunOutputV1(ResumeRunOutputV1[ResumeGenerationResultV1]):
    mode: Literal[RunMode.RESUME_GENERATION] = RunMode.RESUME_GENERATION
    payload: ResumeGenerationResultV1


class ResumeRevisionInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: UUID
    feedback_id: UUID
    base_version_id: UUID | None


class ResumeRevisionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: UUID
    feedback_id: UUID
    version_id: UUID | None
    artifact_id: UUID | None
    outcome: Literal["draft", "needs_input"]
    questions: tuple[str, ...]


class ResumeRevisionRunInputV1(ResumeRunInputV1[ResumeRevisionInputV1]):
    mode: Literal[RunMode.RESUME_REVISION] = RunMode.RESUME_REVISION
    payload: ResumeRevisionInputV1


class ResumeRevisionCandidateOutputV1(ResumeRunOutputV1[RevisionCandidateV1]):
    mode: Literal[RunMode.RESUME_REVISION] = RunMode.RESUME_REVISION
    payload: RevisionCandidateV1


class ResumeRevisionRunOutputV1(ResumeRunOutputV1[ResumeRevisionResultV1]):
    mode: Literal[RunMode.RESUME_REVISION] = RunMode.RESUME_REVISION
    payload: ResumeRevisionResultV1


type RunInput = ResearchRequestV1 | ResumeRunInputV1
type RunOutput = ResearchOutputV1 | ResearchOutputV2 | ResumeRunOutputV1


@dataclass(frozen=True, slots=True)
class RunContractV1:
    mode: RunMode
    graph_version: str
    input_model: type[ResearchRequestV1] | type[ResumeRunInputV1]
    output_models: tuple[
        type[ResearchOutputV1] | type[ResearchOutputV2] | type[ResumeRunOutputV1], ...
    ]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode, RunMode)
            or not isinstance(self.graph_version, str)
            or not isinstance(self.output_models, tuple)
            or not self.output_models
            or not isinstance(self.input_model, type)
            or any(not isinstance(model, type) for model in self.output_models)
        ):
            raise ValueError("invalid execution contract")
        legacy = self.mode in LEGACY_RUN_MODES
        if legacy:
            if (
                self.input_model is not ResearchRequestV1
                or any(
                    model not in (ResearchOutputV1, ResearchOutputV2)
                    for model in self.output_models
                )
                or self.graph_version not in {f"pathfinder-research-v{i}" for i in range(1, 7)}
            ):
                raise ValueError("invalid legacy contract")
        elif (
            re.fullmatch(r"pathfinder-resume-v[1-9][0-9]*", self.graph_version) is None
            or not issubclass(self.input_model, ResumeRunInputV1)
            or any(not issubclass(model, ResumeRunOutputV1) for model in self.output_models)
        ):
            raise ValueError("invalid resume contract")
        versions = [model.model_fields["schema_version"].default for model in self.output_models]
        if len(versions) != len(set(versions)):
            raise ValueError("ambiguous result schema versions")

    def decode_input(self, value: object) -> RunInput:
        if not isinstance(value, dict) or type(value.get("schema_version", 1)) is not int:
            raise ValueError("invalid input schema version")
        result = self.input_model.model_validate_json(_json(value), strict=True)
        if isinstance(result, ResearchRequestV1):
            if result.include_application_draft is not (self.mode is RunMode.APPLICATION):
                raise ValueError("input mode mismatch")
        elif result.mode is not self.mode:
            raise ValueError("input mode mismatch")
        return result

    def decode_output(self, value: object) -> RunOutput:
        if not isinstance(value, dict):
            raise ValueError("invalid result")
        version = value.get("schema_version", 1 if self.mode in LEGACY_RUN_MODES else None)
        model = next(
            (m for m in self.output_models if m.model_fields["schema_version"].default == version),
            None,
        )
        if model is None or type(version) is not int:
            raise ValueError("unknown result schema version")
        result = model.model_validate_json(_json(value), strict=True)
        if isinstance(result, ResumeRunOutputV1) and result.mode is not self.mode:
            raise ValueError("output mode mismatch")
        return result


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


LEGACY_READ_CONTRACTS = tuple(
    RunContractV1(
        mode,
        f"pathfinder-research-v{version}",
        ResearchRequestV1,
        (ResearchOutputV1, ResearchOutputV2),
    )
    for version in range(1, 7)
    for mode in (RunMode.RESEARCH, RunMode.APPLICATION)
)
# The v1 material contract remains readable, but only the v2 extractor runs.
MATERIAL_V1_READ_CONTRACT = RunContractV1(
    RunMode.MATERIAL_PREPARATION,
    "pathfinder-resume-v1",
    MaterialPreparationRunInputV1,
    (MaterialPreparationRunOutputV1,),
)
EXECUTION_CONTRACTS: tuple[RunContractV1, ...] = (
    RunContractV1(
        RunMode.MATERIAL_PREPARATION,
        "pathfinder-resume-v2",
        MaterialPreparationRunInputV1,
        (MaterialPreparationRunOutputV1,),
    ),
    RunContractV1(
        RunMode.RESUME_GENERATION,
        "pathfinder-resume-v3",
        ResumeGenerationRunInputV1,
        (ResumeGenerationRunOutputV1,),
    ),
    RunContractV1(
        RunMode.RESUME_REVISION,
        "pathfinder-resume-v4",
        ResumeRevisionRunInputV1,
        (ResumeRevisionRunOutputV1,),
    ),
)
READ_CONTRACTS = (*LEGACY_READ_CONTRACTS, MATERIAL_V1_READ_CONTRACT, *EXECUTION_CONTRACTS)


def find_run_contract(
    contracts: tuple[RunContractV1, ...], graph_version: str, mode: RunMode
) -> RunContractV1:
    matches = [
        item for item in contracts if item.graph_version == graph_version and item.mode is mode
    ]
    if len(matches) != 1:
        raise ValueError("unsupported execution contract")
    return matches[0]
