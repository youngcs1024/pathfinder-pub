"""Strict request identity parsing for the Run creation route."""

import re
from collections.abc import Sequence
from uuid import UUID

from app.api.errors import InvalidIdempotencyKeyError


def parse_idempotency_key(values: Sequence[str]) -> UUID | None:
    """Parse all occurrences, never a lossy single-value Header lookup."""
    if isinstance(values, (str, bytes)):
        raise InvalidIdempotencyKeyError()
    if not values:
        return None
    if len(values) != 1:
        raise InvalidIdempotencyKeyError()
    value = values[0]
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            value,
        )
        is None
    ):
        raise InvalidIdempotencyKeyError()
    return UUID(value)
