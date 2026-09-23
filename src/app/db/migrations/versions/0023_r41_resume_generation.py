"""Fixed job snapshots and first-draft sessions, versions and coverage."""

# Raw SQL keeps constraint names and composite keys visible in the migration.
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op

revision = "0023_r41_resume_generation"
down_revision = "0022_r32_resume_tex_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = """
        CREATE TABLE job_snapshots (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            source text NOT NULL CHECK (source IN ('paste','upload')),
            filename text,
            jd_text text NOT NULL CHECK (octet_length(jd_text) BETWEEN 1 AND 32768),
            jd_sha256 text NOT NULL CHECK (jd_sha256 ~ '^[0-9a-f]{64}$'),
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_job_snapshots_workspace_id_id UNIQUE (workspace_id,id)
        );
        CREATE INDEX ix_job_snapshots_workspace ON job_snapshots(workspace_id);
        CREATE TABLE resume_sessions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            owner_user_id uuid NOT NULL,
            profile_version_id uuid NOT NULL,
            preference_version_id uuid NOT NULL,
            job_snapshot_id uuid NOT NULL,
            run_id uuid NOT NULL,
            current_version_id uuid,
            revision integer NOT NULL DEFAULT 0,
            repair_count integer NOT NULL DEFAULT 0,
            override_json jsonb NOT NULL,
            budget_json jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_resume_sessions_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_sessions_workspace_id_run_id UNIQUE (workspace_id,run_id),
            FOREIGN KEY (workspace_id,owner_user_id) REFERENCES workspace_memberships(workspace_id,user_id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,profile_version_id) REFERENCES resume_profile_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,preference_version_id) REFERENCES resume_preference_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,job_snapshot_id) REFERENCES job_snapshots(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,run_id) REFERENCES runs(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_sessions_state CHECK (revision >= 0 AND repair_count BETWEEN 0 AND 1),
            CONSTRAINT ck_resume_sessions_override_json CHECK (jsonb_typeof(override_json) = 'object'),
            CONSTRAINT ck_resume_sessions_budget_json CHECK (jsonb_typeof(budget_json) = 'object')
        );
        CREATE INDEX ix_resume_sessions_workspace_owner ON resume_sessions(workspace_id,owner_user_id);
        CREATE TABLE resume_session_projects (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            project_id uuid NOT NULL,
            CONSTRAINT uq_resume_session_projects_workspace_id_session_id_project_id
                UNIQUE (workspace_id,session_id,project_id),
            FOREIGN KEY (workspace_id,session_id) REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,project_id) REFERENCES material_projects(workspace_id,id) ON DELETE RESTRICT
        );
        CREATE TABLE resume_session_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            project_id uuid NOT NULL,
            fact_version_id uuid NOT NULL,
            CONSTRAINT uq_resume_session_facts_workspace_id_session_id_fact_version_id
                UNIQUE (workspace_id,session_id,fact_version_id),
            FOREIGN KEY (workspace_id,session_id) REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,project_id) REFERENCES material_projects(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,fact_version_id) REFERENCES material_fact_versions(workspace_id,id) ON DELETE RESTRICT
        );
        CREATE INDEX ix_resume_session_facts_workspace_session ON resume_session_facts(workspace_id,session_id);
        CREATE TABLE job_requirements (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            ordinal integer NOT NULL,
            kind text NOT NULL CHECK (kind IN ('explicit','preferred','inferred')),
            start_offset integer NOT NULL,
            end_offset integer NOT NULL,
            quote text NOT NULL,
            inference_basis text,
            CONSTRAINT uq_job_requirements_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_job_requirements_workspace_id_session_id_ordinal
                UNIQUE (workspace_id,session_id,ordinal),
            FOREIGN KEY (workspace_id,session_id) REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_job_requirements_offsets CHECK (start_offset >= 0 AND end_offset > start_offset)
        );
        CREATE INDEX ix_job_requirements_workspace_session ON job_requirements(workspace_id,session_id);
        CREATE TABLE resume_versions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            version integer NOT NULL,
            artifact_id uuid NOT NULL,
            content_json jsonb NOT NULL,
            validation_json jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_resume_versions_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_versions_workspace_id_session_id_version
                UNIQUE (workspace_id,session_id,version),
            FOREIGN KEY (workspace_id,session_id) REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,artifact_id) REFERENCES resume_tex_artifacts(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_versions_content CHECK (version >= 1 AND jsonb_typeof(content_json) = 'object'),
            CONSTRAINT ck_resume_versions_validation CHECK (jsonb_typeof(validation_json) = 'object')
        );
        CREATE INDEX ix_resume_versions_workspace_session ON resume_versions(workspace_id,session_id);
        ALTER TABLE resume_sessions ADD CONSTRAINT fk_resume_sessions_current_version
            FOREIGN KEY (workspace_id,current_version_id) REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT;
        CREATE TABLE requirement_coverage (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            version_id uuid NOT NULL,
            requirement_id uuid NOT NULL,
            support text NOT NULL CHECK (support IN ('supported','partial','no_support_found')),
            verification text NOT NULL CHECK (verification IN ('needs_human_review','confirmed_gap','material_insufficient','unchecked')),
            reason text NOT NULL,
            item_ids_json jsonb NOT NULL CHECK (jsonb_typeof(item_ids_json) = 'array'),
            CONSTRAINT uq_requirement_coverage_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_requirement_coverage_workspace_id_version_id_requirement_id
                UNIQUE (workspace_id,version_id,requirement_id),
            FOREIGN KEY (workspace_id,version_id) REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,requirement_id) REFERENCES job_requirements(workspace_id,id) ON DELETE RESTRICT
        );
        CREATE INDEX ix_requirement_coverage_workspace_version ON requirement_coverage(workspace_id,version_id);
        CREATE TABLE requirement_coverage_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            coverage_id uuid NOT NULL,
            fact_version_id uuid NOT NULL,
            CONSTRAINT uq_requirement_coverage_facts_workspace_id_coverage_id__c63f
                UNIQUE (workspace_id,coverage_id,fact_version_id),
            FOREIGN KEY (workspace_id,coverage_id) REFERENCES requirement_coverage(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,fact_version_id) REFERENCES material_fact_versions(workspace_id,id) ON DELETE RESTRICT
        );
    """
    for statement in statements.split(";"):
        if statement.strip():
            op.execute(sa.text(statement))
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1','pathfinder-research-v2',"
        "'pathfinder-research-v3','pathfinder-research-v4','pathfinder-research-v5',"
        "'pathfinder-research-v6','pathfinder-resume-v1','pathfinder-resume-v2',"
        "'pathfinder-resume-v3')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    for table in (
        "requirement_coverage_facts",
        "requirement_coverage",
        "resume_versions",
        "job_requirements",
        "resume_session_facts",
        "resume_session_projects",
        "resume_sessions",
        "job_snapshots",
    ):
        if bind.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar_one():
            raise RuntimeError("resume generation history cannot be safely downgraded")
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1','pathfinder-research-v2',"
        "'pathfinder-research-v3','pathfinder-research-v4','pathfinder-research-v5',"
        "'pathfinder-research-v6','pathfinder-resume-v1','pathfinder-resume-v2')",
    )
    op.drop_constraint("fk_resume_sessions_current_version", "resume_sessions", type_="foreignkey")
    for table in (
        "requirement_coverage_facts",
        "requirement_coverage",
        "resume_versions",
        "job_requirements",
        "resume_session_facts",
        "resume_session_projects",
        "resume_sessions",
        "job_snapshots",
    ):
        op.drop_table(table)
