\set ON_ERROR_STOP on

BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;

WITH
params AS (
    SELECT
        :'fixture_run_id'::uuid AS fixture_run_id,
        :'fixture_document_id'::uuid AS fixture_document_id
),
expected_public_tables(table_name) AS (
    VALUES
        ('action_intents'),
        ('alembic_version'),
        ('approval_decisions'),
        ('approval_requests'),
        ('conversations'),
        ('document_chunks'),
        ('documents'),
        ('llm_invocations'),
        ('messages'),
        ('mock_submissions'),
        ('run_events'),
        ('run_jobs'),
        ('runs'),
        ('tool_invocations'),
        ('users'),
        ('workspace_memberships'),
        ('workspaces')
),
expected_checkpoint_tables(table_name) AS (
    VALUES
        ('checkpoint_blobs'),
        ('checkpoint_migrations'),
        ('checkpoint_writes'),
        ('checkpoints')
),
public_table_presence AS (
    SELECT
        expected.table_name,
        to_regclass(format('public.%I', expected.table_name)) IS NOT NULL AS present
    FROM expected_public_tables AS expected
),
checkpoint_table_presence AS (
    SELECT
        expected.table_name,
        to_regclass(format('pathfinder_checkpoint.%I', expected.table_name)) IS NOT NULL
            AS present
    FROM expected_checkpoint_tables AS expected
),
run_request_identity_schema AS (
    SELECT
        (
            SELECT count(*) = 3
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'runs'
              AND is_nullable = 'YES' AND column_default IS NULL
              AND ((column_name = 'client_request_id' AND data_type = 'uuid')
                OR (column_name = 'create_request_digest' AND data_type = 'text')
                OR (column_name = 'create_request_version' AND data_type = 'integer'))
        ) AND EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'public.runs'::regclass
              AND conname = 'ck_runs_create_request_identity'
              AND contype = 'c' AND convalidated
        ) AND EXISTS (
            SELECT 1 FROM pg_constraint AS constraint_row
            JOIN pg_index AS index_row ON index_row.indexrelid = constraint_row.conindid
            WHERE constraint_row.conrelid = 'public.runs'::regclass
              AND constraint_row.conname = 'uq_runs_workspace_creator_request'
              AND constraint_row.contype = 'u' AND constraint_row.convalidated
              AND index_row.indisvalid AND NOT index_row.indnullsnotdistinct
        ) AS valid
),
fixture_run AS (
    SELECT run.*
    FROM runs AS run
    JOIN params ON params.fixture_run_id = run.id
),
fixture_document AS (
    SELECT document.*
    FROM documents AS document
    JOIN params ON params.fixture_document_id = document.id
),
fixture_action AS (
    SELECT action.*
    FROM action_intents AS action
    JOIN params ON params.fixture_run_id = action.run_id
),
fixture_request AS (
    SELECT request.*
    FROM approval_requests AS request
    JOIN params ON params.fixture_run_id = request.run_id
),
fixture_decision AS (
    SELECT decision.*
    FROM approval_decisions AS decision
    JOIN fixture_request AS request ON request.id = decision.approval_request_id
),
fixture_mock AS (
    SELECT submission.*
    FROM mock_submissions AS submission
    JOIN params ON params.fixture_run_id = submission.run_id
),
fixture_events AS (
    SELECT event.*
    FROM run_events AS event
    JOIN params ON params.fixture_run_id = event.run_id
),
fixture_tool_invocations AS (
    SELECT invocation.*
    FROM tool_invocations AS invocation
    JOIN params ON params.fixture_run_id = invocation.run_id
),
fixture_llm_invocations AS (
    SELECT invocation.*
    FROM llm_invocations AS invocation
    JOIN params ON params.fixture_run_id = invocation.run_id
),
snapshot AS (
    SELECT jsonb_build_object(
        'schema', jsonb_build_object(
            'alembic_version', (SELECT version_num FROM alembic_version),
            'public_tables', (
                SELECT jsonb_agg(
                    jsonb_build_object('name', table_name, 'present', present)
                    ORDER BY table_name
                )
                FROM public_table_presence
            ),
            'checkpoint_tables', (
                SELECT jsonb_agg(
                    jsonb_build_object('name', table_name, 'present', present)
                    ORDER BY table_name
                )
                FROM checkpoint_table_presence
            ),
            'runs_has_thread_id', EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'runs'
                  AND column_name = 'thread_id'
            )
        ),
        'global_counts', jsonb_build_object(
            'users', (SELECT count(*) FROM users),
            'workspaces', (SELECT count(*) FROM workspaces),
            'workspace_memberships', (SELECT count(*) FROM workspace_memberships),
            'documents', (SELECT count(*) FROM documents),
            'document_chunks', (SELECT count(*) FROM document_chunks),
            'runs', (SELECT count(*) FROM runs),
            'run_jobs', (SELECT count(*) FROM run_jobs),
            'run_events', (SELECT count(*) FROM run_events),
            'action_intents', (SELECT count(*) FROM action_intents),
            'approval_requests', (SELECT count(*) FROM approval_requests),
            'approval_decisions', (SELECT count(*) FROM approval_decisions),
            'tool_invocations', (SELECT count(*) FROM tool_invocations),
            'mock_submissions', (SELECT count(*) FROM mock_submissions),
            'llm_invocations', (SELECT count(*) FROM llm_invocations),
            'checkpoints', (SELECT count(*) FROM pathfinder_checkpoint.checkpoints)
        ),
        'fixture', jsonb_build_object(
            'workspace', (
                SELECT jsonb_build_object(
                    'id', workspace.id,
                    'kind', workspace.kind,
                    'created_by_user_id', workspace.created_by_user_id
                )
                FROM workspaces AS workspace
                JOIN fixture_run AS run ON run.workspace_id = workspace.id
            ),
            'memberships', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'user_id', membership.user_id,
                            'role', membership.role,
                            'revoked', membership.revoked_at IS NOT NULL
                        )
                        ORDER BY membership.user_id
                    ),
                    '[]'::jsonb
                )
                FROM workspace_memberships AS membership
                JOIN fixture_run AS run ON run.workspace_id = membership.workspace_id
            ),
            'document', (
                SELECT jsonb_build_object(
                    'id', document.id,
                    'workspace_id', document.workspace_id,
                    'created_by_user_id', document.created_by_user_id,
                    'content_hash', document.content_hash,
                    'normalization_version', document.normalization_version,
                    'chunking_version', document.chunking_version,
                    'embedding_model', document.embedding_model,
                    'chunk_count', (
                        SELECT count(*)
                        FROM document_chunks AS chunk
                        WHERE chunk.workspace_id = document.workspace_id
                          AND chunk.document_id = document.id
                    ),
                    'chunks', (
                        SELECT coalesce(
                            jsonb_agg(
                                jsonb_build_object(
                                    'id', chunk.id,
                                    'ordinal', chunk.ordinal,
                                    'content_hash', chunk.content_hash,
                                    'token_count', chunk.token_count,
                                    'embedding_model', chunk.embedding_model
                                )
                                ORDER BY chunk.ordinal
                            ),
                            '[]'::jsonb
                        )
                        FROM document_chunks AS chunk
                        WHERE chunk.workspace_id = document.workspace_id
                          AND chunk.document_id = document.id
                    )
                )
                FROM fixture_document AS document
            ),
            'run', (
                SELECT jsonb_build_object(
                    'id', run.id,
                    'workspace_id', run.workspace_id,
                    'created_by_user_id', run.created_by_user_id,
                    'resume_document_id', run.resume_document_id,
                    'status', run.status,
                    'graph_version', run.graph_version,
                    'next_event_seq', run.next_event_seq
                )
                FROM fixture_run AS run
            ),
            'job', (
                SELECT jsonb_build_object(
                    'id', job.id,
                    'workspace_id', job.workspace_id,
                    'originating_actor_user_id', job.originating_actor_user_id,
                    'run_id', job.run_id,
                    'status', job.status,
                    'attempt', job.attempt,
                    'resume_approval_request_id', job.resume_approval_request_id
                )
                FROM run_jobs AS job
                JOIN params ON params.fixture_run_id = job.run_id
            ),
            'events', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'seq', event.seq,
                            'type', event.type,
                            'actor_user_id', event.actor_user_id
                        )
                        ORDER BY event.seq
                    ),
                    '[]'::jsonb
                )
                FROM fixture_events AS event
            ),
            'action_intents', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', action.id,
                            'originating_actor_user_id', action.originating_actor_user_id,
                            'action_key', action.action_key,
                            'action_revision', action.action_revision,
                            'status', action.status,
                            'effect', action.effect,
                            'tool_name', action.tool_name,
                            'args_digest', action.args_digest,
                            'target_digest', action.target_digest,
                            'approval_binding_version', action.approval_binding_version,
                            'approval_binding_digest', action.approval_binding_digest,
                            'idempotency_key', action.idempotency_key
                        )
                        ORDER BY action.action_key, action.action_revision, action.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_action AS action
            ),
            'approval_requests', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', request.id,
                            'action_intent_id', request.action_intent_id,
                            'status', request.status,
                            'version', request.version,
                            'args_digest', request.args_digest,
                            'target_digest', request.target_digest,
                            'approval_binding_version', request.approval_binding_version,
                            'approval_binding_digest', request.approval_binding_digest
                        )
                        ORDER BY request.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_request AS request
            ),
            'approval_decisions', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', decision.id,
                            'approval_request_id', decision.approval_request_id,
                            'decision', decision.decision,
                            'actor_user_id', decision.actor_user_id
                        )
                        ORDER BY decision.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_decision AS decision
            ),
            'mock_submissions', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', submission.id,
                            'action_intent_id', submission.action_intent_id,
                            'idempotency_key', submission.idempotency_key,
                            'payload_digest', submission.payload_digest
                        )
                        ORDER BY submission.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_mock AS submission
            ),
            'tool_invocations', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', invocation.id,
                            'action_intent_id', invocation.action_intent_id,
                            'tool_name', invocation.tool_name,
                            'effect', invocation.effect,
                            'args_digest', invocation.args_digest,
                            'status', invocation.status,
                            'attempt', invocation.attempt
                        )
                        ORDER BY invocation.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_tool_invocations AS invocation
            ),
            'llm_invocations', (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_object(
                            'id', invocation.id,
                            'invocation_kind', invocation.invocation_kind,
                            'provider', invocation.provider,
                            'model', invocation.model,
                            'graph_node', invocation.graph_node,
                            'status', invocation.status
                        )
                        ORDER BY invocation.id
                    ),
                    '[]'::jsonb
                )
                FROM fixture_llm_invocations AS invocation
            ),
            'checkpoint_count', (
                SELECT count(*)
                FROM pathfinder_checkpoint.checkpoints AS checkpoint
                JOIN params ON checkpoint.thread_id = params.fixture_run_id::text
            )
        ),
        'checks', jsonb_build_object(
            'run_request_identity_schema', (SELECT valid FROM run_request_identity_schema),
            'all_public_tables_present', (
                SELECT bool_and(present) FROM public_table_presence
            ),
            'all_checkpoint_tables_present', (
                SELECT bool_and(present) FROM checkpoint_table_presence
            ),
            'runs_has_no_thread_id', NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'runs'
                  AND column_name = 'thread_id'
            ),
            'fixture_run_exactly_one', (SELECT count(*) = 1 FROM fixture_run),
            'fixture_document_exactly_one', (SELECT count(*) = 1 FROM fixture_document),
            'fixture_document_matches_run', (
                SELECT count(*) = 1
                FROM fixture_run AS run
                JOIN fixture_document AS document
                  ON document.workspace_id = run.workspace_id
                 AND document.id = run.resume_document_id
            ),
            'fixture_document_has_chunks', (
                SELECT count(*) > 0
                FROM document_chunks AS chunk
                JOIN fixture_document AS document
                  ON document.workspace_id = chunk.workspace_id
                 AND document.id = chunk.document_id
            ),
            'document_representation_identities_unique', NOT EXISTS (
                SELECT 1
                FROM documents
                GROUP BY
                    workspace_id,
                    content_hash,
                    normalization_version,
                    chunking_version,
                    embedding_model
                HAVING count(*) <> 1
            ),
            'fixture_run_completed', (
                SELECT count(*) = 1 FROM fixture_run WHERE status = 'completed'
            ),
            'fixture_job_done', (
                SELECT count(*) = 1
                FROM run_jobs AS job
                JOIN params ON params.fixture_run_id = job.run_id
                WHERE job.status = 'done'
            ),
            'fixture_events_contiguous', (
                SELECT count(*) > 0
                   AND min(seq) = 1
                   AND max(seq) = count(*)
                FROM fixture_events
            ),
            'fixture_next_event_seq_correct', (
                SELECT count(*) = 1
                FROM fixture_run AS run
                WHERE run.next_event_seq = (
                    SELECT coalesce(max(event.seq), 0) + 1 FROM fixture_events AS event
                )
            ),
            'fixture_action_succeeded', (
                SELECT count(*) = 1
                FROM fixture_action
                WHERE status = 'succeeded'
                  AND action_key = 'submit_application'
                  AND effect = 'irreversible'
                  AND tool_name = 'submit_mock_application'
            ),
            'fixture_request_consumed', (
                SELECT count(*) = 1 FROM fixture_request WHERE status = 'consumed'
            ),
            'fixture_decision_approved', (
                SELECT count(*) = 1 FROM fixture_decision WHERE decision = 'approve'
            ),
            'fixture_binding_consistent', (
                SELECT count(*) = 1
                FROM fixture_action AS action
                JOIN fixture_request AS request
                  ON request.action_intent_id = action.id
                 AND request.args_digest = action.args_digest
                 AND request.target_digest = action.target_digest
                 AND request.approval_binding_version = action.approval_binding_version
                 AND request.approval_binding_digest = action.approval_binding_digest
            ),
            'fixture_actor_audit_consistent', (
                SELECT count(*) = 1
                FROM fixture_run AS run
                JOIN run_jobs AS job
                  ON job.workspace_id = run.workspace_id
                 AND job.run_id = run.id
                 AND job.originating_actor_user_id = run.created_by_user_id
                JOIN workspace_memberships AS membership
                  ON membership.workspace_id = run.workspace_id
                 AND membership.user_id = run.created_by_user_id
                 AND membership.revoked_at IS NULL
            ),
            'fixture_mock_exactly_once', (
                SELECT count(*) = 1
                FROM fixture_mock AS submission
                JOIN fixture_action AS action
                  ON action.id = submission.action_intent_id
                 AND action.idempotency_key = submission.idempotency_key
            ),
            'fixture_irreversible_invocation_bound', (
                SELECT count(*) = 1
                FROM fixture_tool_invocations AS invocation
                JOIN fixture_action AS action
                  ON action.id = invocation.action_intent_id
                 AND action.args_digest = invocation.args_digest
                WHERE invocation.effect = 'irreversible'
                  AND invocation.status = 'succeeded'
            ),
            'fixture_llm_invocations_fake_and_terminal', (
                SELECT count(*) > 0
                   AND bool_and(provider = 'fake')
                   AND bool_and(status IN ('succeeded', 'failed'))
                FROM fixture_llm_invocations
            ),
            'fixture_checkpoint_present', (
                SELECT count(*) > 0
                FROM pathfinder_checkpoint.checkpoints AS checkpoint
                JOIN params ON checkpoint.thread_id = params.fixture_run_id::text
            ),
            'source_is_quiescent', NOT EXISTS (
                SELECT 1 FROM runs WHERE status IN ('queued', 'running', 'waiting_approval')
            ) AND NOT EXISTS (
                SELECT 1 FROM run_jobs WHERE status IN ('queued', 'leased')
            )
        )
    ) AS value
)
SELECT value::text FROM snapshot;

COMMIT;
