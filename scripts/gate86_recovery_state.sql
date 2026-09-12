\set ON_ERROR_STOP on

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;

WITH
params AS (
    SELECT
        :'baseline_run_id'::uuid AS baseline_run_id,
        :'waiting_run_id'::uuid AS waiting_run_id,
        :'leased_run_id'::uuid AS leased_run_id,
        :'queued_run_id'::uuid AS queued_run_id
),
fixture_ids(name, run_id) AS (
    SELECT 'baseline', baseline_run_id FROM params
    UNION ALL SELECT 'waiting', waiting_run_id FROM params
    UNION ALL SELECT 'leased', leased_run_id FROM params
    UNION ALL SELECT 'queued', queued_run_id FROM params
),
fixture_rows AS (
    SELECT
        fixture.name,
        run.id AS run_id,
        jsonb_build_object(
            'run', jsonb_build_object(
                'id', run.id,
                'workspace_id', run.workspace_id,
                'status', run.status,
                'graph_version', run.graph_version,
                'next_event_seq', run.next_event_seq
            ),
            'job', jsonb_build_object(
                'id', job.id,
                'status', job.status,
                'attempt', job.attempt,
                'max_attempts', job.max_attempts,
                'available_at', job.available_at,
                'leased_by', job.leased_by,
                'owner_present', job.owner_token IS NOT NULL,
                'lease_expires_at', job.lease_expires_at,
                'resume_approval_request_id', job.resume_approval_request_id
            ),
            'events', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', event.id,
                        'seq', event.seq,
                        'type', event.type,
                        'version', event.version
                    ) ORDER BY event.seq
                ), '[]'::jsonb)
                FROM run_events AS event
                WHERE event.workspace_id = run.workspace_id AND event.run_id = run.id
            ),
            'approval_requests', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', request.id,
                        'action_intent_id', request.action_intent_id,
                        'status', request.status,
                        'version', request.version,
                        'expires_at', request.expires_at,
                        'args_digest', request.args_digest,
                        'target_digest', request.target_digest,
                        'approval_binding_version', request.approval_binding_version,
                        'approval_binding_digest', request.approval_binding_digest
                    ) ORDER BY request.id
                ), '[]'::jsonb)
                FROM approval_requests AS request
                WHERE request.workspace_id = run.workspace_id AND request.run_id = run.id
            ),
            'decision_count', (
                SELECT count(*)
                FROM approval_decisions AS decision
                JOIN approval_requests AS request
                  ON request.workspace_id = decision.workspace_id
                 AND request.id = decision.approval_request_id
                WHERE request.workspace_id = run.workspace_id AND request.run_id = run.id
            ),
            'approve_decision_count', (
                SELECT count(*)
                FROM approval_decisions AS decision
                JOIN approval_requests AS request
                  ON request.workspace_id = decision.workspace_id
                 AND request.id = decision.approval_request_id
                WHERE request.workspace_id = run.workspace_id
                  AND request.run_id = run.id
                  AND decision.decision = 'approve'
            ),
            'actions', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', action.id,
                        'status', action.status,
                        'action_key', action.action_key,
                        'action_revision', action.action_revision,
                        'tool_name', action.tool_name,
                        'effect', action.effect,
                        'args_digest', action.args_digest,
                        'target_digest', action.target_digest,
                        'approval_binding_version', action.approval_binding_version,
                        'approval_binding_digest', action.approval_binding_digest,
                        'idempotency_key', action.idempotency_key
                    ) ORDER BY action.action_revision, action.id
                ), '[]'::jsonb)
                FROM action_intents AS action
                WHERE action.workspace_id = run.workspace_id AND action.run_id = run.id
            ),
            'mock_submissions', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', submission.id,
                        'action_intent_id', submission.action_intent_id,
                        'idempotency_key', submission.idempotency_key,
                        'payload_digest', submission.payload_digest
                    ) ORDER BY submission.id
                ), '[]'::jsonb)
                FROM mock_submissions AS submission
                WHERE submission.workspace_id = run.workspace_id
                  AND submission.run_id = run.id
            ),
            'tool_invocations', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', invocation.id,
                        'action_intent_id', invocation.action_intent_id,
                        'tool_name', invocation.tool_name,
                        'effect', invocation.effect,
                        'status', invocation.status,
                        'attempt', invocation.attempt
                    ) ORDER BY invocation.id
                ), '[]'::jsonb)
                FROM tool_invocations AS invocation
                WHERE invocation.workspace_id = run.workspace_id
                  AND invocation.run_id = run.id
            ),
            'llm_invocations', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'id', invocation.id,
                        'invocation_kind', invocation.invocation_kind,
                        'provider', invocation.provider,
                        'model', invocation.model,
                        'graph_node', invocation.graph_node,
                        'status', invocation.status
                    ) ORDER BY invocation.id
                ), '[]'::jsonb)
                FROM llm_invocations AS invocation
                WHERE invocation.workspace_id = run.workspace_id
                  AND invocation.run_id = run.id
            ),
            'checkpoints', (
                SELECT coalesce(jsonb_agg(
                    jsonb_build_object(
                        'checkpoint_ns', checkpoint.checkpoint_ns,
                        'checkpoint_id', checkpoint.checkpoint_id
                    ) ORDER BY checkpoint.checkpoint_ns, checkpoint.checkpoint_id
                ), '[]'::jsonb)
                FROM pathfinder_checkpoint.checkpoints AS checkpoint
                WHERE checkpoint.thread_id = run.id::text
            ),
            'checks', jsonb_build_object(
                'events_contiguous', (
                    SELECT count(*) > 0 AND min(event.seq) = 1 AND max(event.seq) = count(*)
                    FROM run_events AS event
                    WHERE event.workspace_id = run.workspace_id AND event.run_id = run.id
                ),
                'next_event_seq_correct', run.next_event_seq = (
                    SELECT coalesce(max(event.seq), 0) + 1
                    FROM run_events AS event
                    WHERE event.workspace_id = run.workspace_id AND event.run_id = run.id
                )
            )
        ) AS value
    FROM fixture_ids AS fixture
    LEFT JOIN runs AS run ON run.id = fixture.run_id
    LEFT JOIN run_jobs AS job ON job.run_id = run.id AND job.workspace_id = run.workspace_id
),
snapshot AS (
    SELECT jsonb_build_object(
        'version', 1,
        'captured_at', statement_timestamp(),
        'schema_revision', (SELECT version_num FROM alembic_version),
        'fixtures', coalesce(jsonb_object_agg(name, value ORDER BY name), '{}'::jsonb),
        'checks', jsonb_build_object(
            'all_fixture_runs_present', count(run_id) = 4,
            'all_fixture_names_present', count(*) = 4,
            'all_events_contiguous', bool_and((value #>> '{checks,events_contiguous}')::boolean),
            'all_next_event_seq_correct',
                bool_and((value #>> '{checks,next_event_seq_correct}')::boolean)
        )
    ) AS value
    FROM fixture_rows
)
SELECT value::text FROM snapshot;
