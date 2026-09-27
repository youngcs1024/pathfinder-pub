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
    "0013_gate6_mock_action_execution.py": (
        "a45d7efeb8e31be2f84ee5c144ec422a5634c0bc21b7a33235d39523e5498122"
    ),
    "0014_gate6_action_recovery.py": (
        "4e3108c8a4a1db40a2fc20a3094a338b2fb1ab050a334aaa009f24bd6d36adfb"
    ),
    "0015_e3_run_request_identity.py": (
        "5582d4480f0f138959f7178a1cae2b67a672a82f5392d16edd2f7bba4af30d7a"
    ),
    "0016_r1_execution_contracts.py": (
        "d79b5683f2e8fe50e969727eeea168749dbea39505c8ef427a20f95d601eed9f"
    ),
    "0017_r12_resume_commands.py": (
        "c5907cb01a365e99bcdaaa17a8fd93049e40e14ae52956b84b56da4071403d04"
    ),
    "0018_r21_material_snapshots.py": (
        "c8533f58a091f74ed88f744192ca906c50da96b5251096caabe98d5fe6f13555"
    ),
    "0019_r21_material_line_ranges.py": (
        "5d8b97d2d25ad9fcd87e09492b61bb7dd7fba22f79e52684682fc502618aab62"
    ),
    "0020_r22_project_facts.py": (
        "87e9aa9d16ca083e062be32f5c70e01f107979b3b129d6eecd37b90a32c73e15"
    ),
    "0021_r31_resume_profiles.py": (
        "7392b9f091a91b26332830929726aef53267ddb6d82794349d79311cc8805df7"
    ),
    "0022_r32_resume_tex_artifacts.py": (
        "b4ef8e6b1ec4d9e568c1560cf0fecbf9132b8f2bd3185da175dcc60991f1feea"
    ),
    "0023_r41_resume_generation.py": (
        "51d162e500174263aebcc3e79d31036a32af66dec2c455648a77f5b66145bf76"
    ),
    "0024_r51_resume_revision.py": (
        "7ff93d627a67e00d3b911a4599379250ae43a4ce0ec13e6630f9c9d21ad889ba"
    ),
    "0025_r52_resume_confirmations.py": (
        "461e8249cc6fa675bb60a96b0dbd6d7c4bcc1f3f1b6faa992d382b1af5ea6fbb"
    ),
}


def test_applied_migrations_are_byte_for_byte_immutable() -> None:
    actual = {
        filename: sha256((MIGRATION_ROOT / filename).read_bytes()).hexdigest()
        for filename in APPLIED_MIGRATION_HASHES
    }

    assert actual == APPLIED_MIGRATION_HASHES
