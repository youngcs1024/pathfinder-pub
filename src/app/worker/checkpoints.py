"""Checkpoint lifecycle is composed by worker/main through the db adapter.

Kept as a module boundary marker for the worker package; it intentionally owns no
database or LangGraph imports.
"""

type CheckpointHandle = object
