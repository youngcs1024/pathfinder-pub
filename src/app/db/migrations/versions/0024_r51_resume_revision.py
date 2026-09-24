"""R5.1 feedback, user attestations, scoped revision history and execution v4."""

import sqlalchemy as sa
from alembic import op

revision = "0024_r51_resume_revision"
down_revision = "0023_r41_resume_generation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = """
        ALTER TABLE resume_sessions ADD COLUMN latest_run_id uuid;
        UPDATE resume_sessions SET latest_run_id = run_id;
        ALTER TABLE resume_sessions ALTER COLUMN latest_run_id SET NOT NULL;
        ALTER TABLE resume_sessions ADD CONSTRAINT fk_resume_sessions_latest_run
            FOREIGN KEY (workspace_id,latest_run_id) REFERENCES runs(workspace_id,id)
            ON DELETE RESTRICT;
        ALTER TABLE resume_commands ADD CONSTRAINT uq_resume_commands_workspace_id_id
            UNIQUE (workspace_id,id);
        ALTER TABLE resume_versions ADD COLUMN parent_version_id uuid;
        ALTER TABLE resume_versions ADD COLUMN feedback_id uuid;
        ALTER TABLE resume_versions ADD COLUMN diff_json jsonb NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE resume_versions ADD COLUMN impact_json jsonb NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE resume_versions ADD CONSTRAINT ck_resume_versions_diff
            CHECK (jsonb_typeof(diff_json) = 'array');
        ALTER TABLE resume_versions ADD CONSTRAINT ck_resume_versions_impact
            CHECK (jsonb_typeof(impact_json) = 'array');
        ALTER TABLE resume_versions ADD CONSTRAINT fk_resume_versions_parent
            FOREIGN KEY (workspace_id,parent_version_id)
            REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT;
        CREATE TABLE resume_feedback (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            command_id uuid NOT NULL,
            run_id uuid,
            base_version_id uuid,
            target_version_id uuid,
            kind text NOT NULL CHECK (kind IN
                ('answer','fact','preference','content','lock','fact_review')),
            scope text NOT NULL,
            request_json jsonb NOT NULL CHECK (jsonb_typeof(request_json) = 'object'),
            normalized_json jsonb NOT NULL CHECK (jsonb_typeof(normalized_json) = 'object'),
            questions_json jsonb NOT NULL CHECK (jsonb_typeof(questions_json) = 'array'),
            repair_count integer NOT NULL DEFAULT 0 CHECK (repair_count BETWEEN 0 AND 1),
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_resume_feedback_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_feedback_workspace_id_command_id UNIQUE (workspace_id,command_id),
            CONSTRAINT uq_resume_feedback_workspace_id_run_id UNIQUE (workspace_id,run_id),
            FOREIGN KEY (workspace_id,session_id)
                REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,run_id)
                REFERENCES runs(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,base_version_id)
                REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,target_version_id)
                REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT
        );
        CREATE INDEX ix_resume_feedback_workspace_session
            ON resume_feedback(workspace_id,session_id);
        ALTER TABLE resume_versions ADD CONSTRAINT fk_resume_versions_feedback
            FOREIGN KEY (workspace_id,feedback_id)
            REFERENCES resume_feedback(workspace_id,id) ON DELETE RESTRICT;
        CREATE TABLE resume_session_preference_versions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            version integer NOT NULL,
            preferences_json jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_resume_session_preference_versions_workspace_id_id
                UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_session_preference_versions_workspace_id_session_id_version
                UNIQUE (workspace_id,session_id,version),
            FOREIGN KEY (workspace_id,session_id)
                REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_session_preference_versions_value
                CHECK (version >= 1 AND jsonb_typeof(preferences_json) = 'object')
        );
        CREATE INDEX ix_resume_session_preferences_workspace_session
            ON resume_session_preference_versions(workspace_id,session_id);
        INSERT INTO resume_session_preference_versions
            (id,workspace_id,session_id,version,preferences_json)
        SELECT gen_random_uuid(),workspace_id,id,1,
            jsonb_build_object('override',override_json,'locked_item_ids','[]'::jsonb)
        FROM resume_sessions;
        CREATE TABLE resume_user_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            project_id uuid NOT NULL,
            scope_session_id uuid,
            current_version integer NOT NULL CHECK (current_version >= 1),
            supersedes_material_version_id uuid,
            supersedes_user_version_id uuid,
            CONSTRAINT uq_resume_user_facts_workspace_id_id UNIQUE (workspace_id,id),
            FOREIGN KEY (workspace_id,project_id)
                REFERENCES material_projects(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,scope_session_id,project_id)
                REFERENCES resume_session_projects(workspace_id,session_id,project_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,supersedes_material_version_id)
                REFERENCES material_fact_versions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_user_facts_one_prior
                CHECK (supersedes_material_version_id IS NULL OR
                       supersedes_user_version_id IS NULL)
        );
        CREATE INDEX ix_resume_user_facts_workspace_project
            ON resume_user_facts(workspace_id,project_id);
        CREATE TABLE resume_user_fact_versions (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            fact_id uuid NOT NULL,
            version integer NOT NULL,
            claim text NOT NULL,
            kind text NOT NULL,
            conditions_json jsonb NOT NULL,
            review_status text NOT NULL,
            created_by_user_id uuid NOT NULL,
            attested_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_resume_user_fact_versions_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_user_fact_versions_workspace_id_fact_id_version
                UNIQUE (workspace_id,fact_id,version),
            FOREIGN KEY (workspace_id,fact_id)
                REFERENCES resume_user_facts(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,created_by_user_id)
                REFERENCES workspace_memberships(workspace_id,user_id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_user_fact_versions_claim
                CHECK (version >= 1 AND char_length(claim) BETWEEN 1 AND 2000),
            CONSTRAINT ck_resume_user_fact_versions_kind
                CHECK (kind IN ('implementation','plan','experiment','personal_statement')),
            CONSTRAINT ck_resume_user_fact_versions_review
                CHECK (review_status IN ('pending','confirmed','rejected')),
            CONSTRAINT ck_resume_user_fact_versions_conditions
                CHECK (jsonb_typeof(conditions_json) = 'object')
        );
        ALTER TABLE resume_user_facts ADD CONSTRAINT fk_resume_user_facts_supersedes_user
            FOREIGN KEY (workspace_id,supersedes_user_version_id)
            REFERENCES resume_user_fact_versions(workspace_id,id) ON DELETE RESTRICT;
        CREATE TABLE resume_session_user_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            fact_version_id uuid NOT NULL,
            CONSTRAINT uq_resume_session_user_facts_workspace_id_session_id_fact_version_id
                UNIQUE (workspace_id,session_id,fact_version_id),
            FOREIGN KEY (workspace_id,session_id)
                REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,fact_version_id)
                REFERENCES resume_user_fact_versions(workspace_id,id) ON DELETE RESTRICT
        );
        CREATE TABLE resume_version_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            version_id uuid NOT NULL,
            material_fact_version_id uuid,
            user_fact_version_id uuid,
            CONSTRAINT uq_resume_version_facts_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_version_facts_material
                UNIQUE (workspace_id,version_id,material_fact_version_id),
            CONSTRAINT uq_resume_version_facts_user
                UNIQUE (workspace_id,version_id,user_fact_version_id),
            FOREIGN KEY (workspace_id,version_id)
                REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,material_fact_version_id)
                REFERENCES material_fact_versions(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,user_fact_version_id)
                REFERENCES resume_user_fact_versions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT ck_resume_version_facts_one_fact
                CHECK ((material_fact_version_id IS NULL) <>
                       (user_fact_version_id IS NULL))
        );
        CREATE INDEX ix_resume_version_facts_workspace_version
            ON resume_version_facts(workspace_id,version_id);
        INSERT INTO resume_version_facts
            (id,workspace_id,version_id,material_fact_version_id)
        SELECT gen_random_uuid(),v.workspace_id,v.id,f.fact_version_id
        FROM resume_versions v
        JOIN resume_session_facts f
            ON f.workspace_id=v.workspace_id AND f.session_id=v.session_id;
        CREATE TABLE requirement_coverage_user_facts (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
            coverage_id uuid NOT NULL,
            fact_version_id uuid NOT NULL,
            CONSTRAINT uq_requirement_coverage_user_facts
                UNIQUE (workspace_id,coverage_id,fact_version_id),
            FOREIGN KEY (workspace_id,coverage_id)
                REFERENCES requirement_coverage(workspace_id,id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id,fact_version_id)
                REFERENCES resume_user_fact_versions(workspace_id,id) ON DELETE RESTRICT
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
        "'pathfinder-resume-v3','pathfinder-resume-v4')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM resume_feedback) OR "
            "EXISTS (SELECT 1 FROM resume_user_facts) OR "
            "EXISTS (SELECT 1 FROM resume_versions WHERE parent_version_id IS NOT NULL) OR "
            "EXISTS (SELECT 1 FROM runs WHERE graph_version='pathfinder-resume-v4')"
        )
    ).scalar_one():
        raise RuntimeError("resume revision history cannot be safely downgraded")
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1','pathfinder-research-v2',"
        "'pathfinder-research-v3','pathfinder-research-v4','pathfinder-research-v5',"
        "'pathfinder-research-v6','pathfinder-resume-v1','pathfinder-resume-v2',"
        "'pathfinder-resume-v3')",
    )
    op.drop_table("requirement_coverage_user_facts")
    op.drop_table("resume_version_facts")
    op.drop_table("resume_session_user_facts")
    op.drop_constraint(
        "fk_resume_user_facts_supersedes_user", "resume_user_facts", type_="foreignkey"
    )
    op.drop_table("resume_user_fact_versions")
    op.drop_table("resume_user_facts")
    op.drop_table("resume_session_preference_versions")
    op.drop_constraint("fk_resume_versions_feedback", "resume_versions", type_="foreignkey")
    op.drop_table("resume_feedback")
    op.drop_constraint("fk_resume_versions_parent", "resume_versions", type_="foreignkey")
    for column in ("impact_json", "diff_json", "feedback_id", "parent_version_id"):
        op.drop_column("resume_versions", column)
    op.drop_constraint("uq_resume_commands_workspace_id_id", "resume_commands", type_="unique")
    op.drop_constraint("fk_resume_sessions_latest_run", "resume_sessions", type_="foreignkey")
    op.drop_column("resume_sessions", "latest_run_id")
