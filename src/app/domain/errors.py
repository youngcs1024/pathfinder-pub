class DomainError(Exception):
    """Base class for errors owned by domain policy."""


class DomainUnavailableError(DomainError):
    """Storage is unavailable; this does not assert that a write did not commit."""

    def __init__(self, *, commit_outcome_unknown: bool = False) -> None:
        self.commit_outcome_unknown = commit_outcome_unknown
        super().__init__("The service is temporarily unavailable.")


class DomainForbiddenError(DomainError):
    """The caller is authenticated but domain authorization denies the operation."""


class DomainNotFoundError(DomainError):
    """A requested domain resource is unavailable to the caller."""


class DomainConflictError(DomainError):
    """A request conflicts with the current domain state."""


class DomainValidationError(DomainError):
    """A request violates a domain rule."""


class InvalidEventCursorError(DomainValidationError):
    """An SSE cursor cannot identify a committed event position for the run."""


class DomainInvariantError(DomainError):
    """Persisted or computed state violates a domain invariant."""
