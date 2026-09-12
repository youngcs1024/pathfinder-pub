from pydantic import BaseModel, ConfigDict


class ProblemDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    title: str
    status: int
    detail: str
    request_id: str
