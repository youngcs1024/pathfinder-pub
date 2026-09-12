from typing import Literal

from pydantic import BaseModel, ConfigDict


class UIConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    auth_mode: Literal["fake", "supabase"]
    supabase_url: str | None
    supabase_publishable_key: str | None
