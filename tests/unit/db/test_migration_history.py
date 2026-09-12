from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_ROOT = PROJECT_ROOT / "src" / "app" / "db" / "migrations" / "versions"

APPLIED_MIGRATION_HASHES = {
    "0001_identity_baseline.py": (
        "eecbf23122000589aa17de4da293d386671a8b33444186dce914228a0fa86faf"
    ),
    "0002_llm_invocations.py": ("34744b710774cbd75952f875ebd9ac4fd4ed9c766dfb310e4cd66959781cfc8a"),
    "0003_openai_adapters.py": ("c6c248fd17aa6139b568491b880dfc4550989b375beeb7d3849351606ba530b5"),
    "0004_llm_costs.py": ("1893ad27bfa62713ed4cd5105b99618c9160408c0a51bf9fcfe5c739a2b00f60"),
    "0005_langfuse_tracing.py": (
        "3eadc46b91bdaae661818cd456eb0ff2141cb4d58bd1337b4d87af1a866aa54b"
    ),
    "0006_qwen_provider_profile.py": (
        "227e8256c0bf3ae91d15c66f9e7e84816ab58b7e364d933dd4ac825a1c9ff77b"
    ),
    "0007_persistent_run_state.py": (
        "c2e967127b3b7c973656acaeb43726d75f42f87e7e067dc48630ecdcbc7d67f8"
    ),
    "0008_rag_document_storage.py": (
        "c01ab448a20e8b3663d2841e9afae147c38fca0090a1aabad90b6ca88af405a4"
    ),
    "0009_gate5_graph_rag.py": ("267c03a21245beabfbde30b826e2f3fb14f3ace8b8fd2f5a76451f01c88eebb9"),
    "0010_gate6_action_proposals.py": (
        "817093fcf720f0c6d4f91b51538a0abdab4e77398d908ab550b4abcb00f49391"
    ),
    "0011_gate6_graph_interrupt.py": (
        "ee5b71d1d4008867104b8b639cf4f8be4d6c0597778bc7f677cbd5f725862503"
    ),
    "0012_gate6_approval_decisions.py": (
        "86f00733fd1ea4c2af20ffd5186678d19069eb8837a3427c2258d6ca69134299"
    ),
}


def test_applied_migrations_are_byte_for_byte_immutable() -> None:
    actual = {
        filename: sha256((MIGRATION_ROOT / filename).read_bytes()).hexdigest()
        for filename in APPLIED_MIGRATION_HASHES
    }

    assert actual == APPLIED_MIGRATION_HASHES
