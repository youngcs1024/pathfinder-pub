\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;

WITH
parameters AS (
    SELECT
        :'actor_user_id'::uuid AS actor_user_id,
        :'workspace_id'::uuid AS workspace_id,
        :'document_id'::uuid AS document_id,
        :'run_id'::uuid AS run_id,
        :'cost_cap_cny'::numeric AS cost_cap_cny
),
selected_run AS (
    SELECT run.*
    FROM runs AS run
    JOIN parameters AS parameter
      ON run.id = parameter.run_id
     AND run.workspace_id = parameter.workspace_id
),
selected_document AS (
    SELECT document.*
    FROM documents AS document
    JOIN parameters AS parameter
      ON document.id = parameter.document_id
     AND document.workspace_id = parameter.workspace_id
),
selected_job AS (
    SELECT job.*
    FROM run_jobs AS job
    JOIN parameters AS parameter
      ON job.run_id = parameter.run_id
     AND job.workspace_id = parameter.workspace_id
),
selected_action AS (
    SELECT action.*
    FROM action_intents AS action
    JOIN parameters AS parameter
      ON action.run_id = parameter.run_id
     AND action.workspace_id = parameter.workspace_id
    WHERE action.action_key = 'submit_application'
),
facts AS (
    SELECT
        (SELECT count(*) FROM users AS actor, parameters AS parameter
          WHERE actor.id = parameter.actor_user_id) AS actor_count,
        (SELECT count(*) FROM workspace_memberships AS membership, parameters AS parameter
          WHERE membership.workspace_id = parameter.workspace_id
            AND membership.user_id = parameter.actor_user_id
            AND membership.revoked_at IS NULL) AS active_membership_count,
        (SELECT count(*) FROM selected_run) AS run_count,
        (SELECT count(*) FROM selected_run AS run, parameters AS parameter
          WHERE run.created_by_user_id = parameter.actor_user_id
            AND run.resume_document_id = parameter.document_id
            AND run.mode = 'application') AS run_identity_count,
        (SELECT count(*) FROM selected_document) AS document_count,
        (SELECT count(*) FROM selected_document AS document, parameters AS parameter
          WHERE document.created_by_user_id = parameter.actor_user_id
            AND document.embedding_model = 'qwen-beijing-text-embedding-v4-1536-v1')
            AS document_profile_count,
        (SELECT count(*) FROM document_chunks AS chunk, parameters AS parameter
          WHERE chunk.workspace_id = parameter.workspace_id
            AND chunk.document_id = parameter.document_id
            AND chunk.embedding_model = 'qwen-beijing-text-embedding-v4-1536-v1') AS chunk_count,
        (SELECT count(*) FROM tool_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.tool_name = 'retrieve_documents'
            AND invocation.effect = 'read_only'
            AND invocation.status = 'succeeded') AS rag_invocation_count,
        (SELECT count(*) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id
            AND event.type = 'rag.retrieved'
            AND event.payload -> 'document_ids' ? parameter.document_id::text)
            AS bound_rag_event_count,
        (SELECT count(*) FROM tool_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.tool_name = 'search_web'
            AND invocation.effect = 'read_only'
            AND invocation.status = 'succeeded') AS search_invocation_count,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'qwen') AS qwen_invocation_count,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'fake') AS fake_invocation_count,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.status = 'started') AS nonterminal_llm_count,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'qwen'
            AND NOT (
                (invocation.invocation_kind = 'chat'
                 AND invocation.model = 'qwen3.6-flash-2026-04-16')
                OR
                (invocation.invocation_kind = 'embedding'
                 AND invocation.model = 'text-embedding-v4')
            )) AS wrong_qwen_profile_count,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'qwen'
            AND invocation.status = 'succeeded'
            AND (invocation.currency IS DISTINCT FROM 'CNY'
                 OR invocation.pricing_version IS NULL
                 OR invocation.estimated_cost IS NULL)) AS unknown_cost_count,
        (SELECT coalesce(sum(invocation.estimated_cost), 0)
           FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'qwen') AS total_cost_cny,
        (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.provider = 'qwen'
            AND invocation.trace_ids IS NULL) AS missing_trace_count,
        (SELECT count(*) FROM pathfinder_checkpoint.checkpoints AS checkpoint,
                              parameters AS parameter
          WHERE checkpoint.thread_id = parameter.run_id::text) AS checkpoint_count,
        (SELECT count(*) FROM selected_action) AS action_count,
        (SELECT count(*) FROM selected_action AS action
          WHERE action.status = 'succeeded'
            AND action.effect = 'irreversible'
            AND action.tool_name = 'submit_mock_application'
            AND action.idempotency_key = action.id::text) AS succeeded_action_count,
        (SELECT count(*) FROM approval_requests AS request
          JOIN selected_action AS action
            ON request.workspace_id = action.workspace_id
           AND request.run_id = action.run_id
           AND request.action_intent_id = action.id
          WHERE request.status = 'consumed') AS consumed_request_count,
        (SELECT count(*) FROM approval_requests AS request
          JOIN selected_action AS action
            ON request.workspace_id = action.workspace_id
           AND request.run_id = action.run_id
           AND request.action_intent_id = action.id
          WHERE request.args_digest = action.args_digest
            AND request.target_digest = action.target_digest
            AND request.approval_binding_version = action.approval_binding_version
            AND request.approval_binding_digest = action.approval_binding_digest)
            AS exact_binding_count,
        (SELECT count(*) FROM approval_decisions AS decision
          JOIN approval_requests AS request
            ON request.workspace_id = decision.workspace_id
           AND request.id = decision.approval_request_id
          JOIN selected_action AS action
            ON action.workspace_id = request.workspace_id
           AND action.run_id = request.run_id
           AND action.id = request.action_intent_id
          WHERE decision.decision = 'approve') AS approve_decision_count,
        (SELECT count(*) FROM tool_invocations AS invocation
          JOIN selected_action AS action
            ON invocation.workspace_id = action.workspace_id
           AND invocation.run_id = action.run_id
           AND invocation.action_intent_id = action.id
          WHERE invocation.tool_name = 'submit_mock_application'
            AND invocation.effect = 'irreversible'
            AND invocation.status = 'succeeded') AS irreversible_invocation_count,
        (SELECT count(*) FROM tool_invocations AS invocation, parameters AS parameter
          WHERE invocation.workspace_id = parameter.workspace_id
            AND invocation.run_id = parameter.run_id
            AND invocation.status = 'outcome_unknown') AS unknown_tool_count,
        (SELECT count(*) FROM selected_action AS action
          WHERE action.status = 'outcome_unknown') AS unknown_action_count,
        (
            (SELECT count(*) FROM selected_action AS action, parameters AS parameter
              WHERE action.originating_actor_user_id <> parameter.actor_user_id)
            +
            (SELECT count(*) FROM tool_invocations AS invocation, parameters AS parameter
              WHERE invocation.workspace_id = parameter.workspace_id
                AND invocation.run_id = parameter.run_id
                AND invocation.originating_actor_user_id <> parameter.actor_user_id)
            +
            (SELECT count(*) FROM llm_invocations AS invocation, parameters AS parameter
              WHERE invocation.workspace_id = parameter.workspace_id
                AND invocation.run_id = parameter.run_id
                AND invocation.actor_user_id <> parameter.actor_user_id)
            +
            (SELECT count(*) FROM mock_submissions AS submission, parameters AS parameter
              WHERE submission.workspace_id = parameter.workspace_id
                AND submission.run_id = parameter.run_id
                AND submission.originating_actor_user_id <> parameter.actor_user_id)
        ) AS tenant_actor_mismatch_count,
        (SELECT count(*) FROM mock_submissions AS submission
          JOIN selected_action AS action
            ON submission.workspace_id = action.workspace_id
           AND submission.run_id = action.run_id
           AND submission.action_intent_id = action.id
           AND submission.idempotency_key = action.idempotency_key) AS mock_count,
        (SELECT count(*) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id) AS event_count,
        (SELECT coalesce(min(event.seq), 0) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id) AS first_event_seq,
        (SELECT coalesce(max(event.seq), 0) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id) AS last_event_seq,
        (SELECT count(DISTINCT event.seq) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id) AS distinct_event_seq_count,
        (SELECT count(*) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id
            AND event.type = 'approval.decided') AS approval_event_count,
        (SELECT count(*) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id
            AND event.type = 'action.completed') AS action_event_count,
        (SELECT count(*) FROM run_events AS event, parameters AS parameter
          WHERE event.workspace_id = parameter.workspace_id
            AND event.run_id = parameter.run_id
            AND event.type = 'run.completed') AS completed_event_count,
        (SELECT count(*) FROM selected_run AS run
          WHERE run.status = 'completed'
            AND run.error_category IS NULL
            AND run.result_json IS NOT NULL) AS completed_run_count,
        (SELECT count(*) FROM selected_job AS job
          WHERE job.status = 'done'
            AND job.leased_by IS NULL
            AND job.owner_token IS NULL
            AND job.lease_expires_at IS NULL) AS done_job_count,
        (SELECT count(*) FROM selected_job AS job WHERE job.status = 'dead') AS dead_job_count,
        (SELECT count(*) FROM selected_run AS run
          WHERE run.next_event_seq = (
              SELECT coalesce(max(event.seq), 0) + 1
              FROM run_events AS event
              WHERE event.workspace_id = run.workspace_id AND event.run_id = run.id
          )) AS next_event_seq_count
),
checks AS (
    SELECT * FROM (
        SELECT 'actor_exists', actor_count = 1 FROM facts UNION ALL
        SELECT 'membership_active', active_membership_count = 1 FROM facts UNION ALL
        SELECT 'run_identity', run_count = 1 AND run_identity_count = 1 FROM facts UNION ALL
        SELECT 'document_identity', document_count = 1 AND document_profile_count = 1 FROM facts UNION ALL
        SELECT 'document_chunks', chunk_count > 0 FROM facts UNION ALL
        SELECT 'rag_evidence', rag_invocation_count > 0 AND bound_rag_event_count > 0 FROM facts UNION ALL
        SELECT 'search_success', search_invocation_count > 0 FROM facts UNION ALL
        SELECT 'qwen_profile', qwen_invocation_count > 0 AND fake_invocation_count = 0
            AND nonterminal_llm_count = 0 AND wrong_qwen_profile_count = 0 FROM facts UNION ALL
        SELECT 'known_cost', unknown_cost_count = 0 FROM facts UNION ALL
        SELECT 'cost_cap', total_cost_cny <= (SELECT cost_cap_cny FROM parameters) FROM facts UNION ALL
        SELECT 'trace_evidence', missing_trace_count = 0 AND qwen_invocation_count > 0 FROM facts UNION ALL
        SELECT 'checkpoint_identity', checkpoint_count > 0 FROM facts UNION ALL
        SELECT 'exact_approval', action_count = 1 AND consumed_request_count = 1
            AND exact_binding_count = 1 AND approve_decision_count = 1 FROM facts UNION ALL
        SELECT 'action_success', succeeded_action_count = 1 AND irreversible_invocation_count = 1
            AND unknown_tool_count = 0 AND unknown_action_count = 0 FROM facts UNION ALL
        SELECT 'tenant_actor_chain', tenant_actor_mismatch_count = 0 FROM facts UNION ALL
        SELECT 'mock_exactly_one', mock_count = 1 FROM facts UNION ALL
        SELECT 'events_contiguous', event_count > 0 AND first_event_seq = 1
            AND last_event_seq = event_count AND distinct_event_seq_count = event_count
            AND next_event_seq_count = 1 FROM facts UNION ALL
        SELECT 'required_events', approval_event_count > 0 AND action_event_count > 0
            AND completed_event_count = 1 FROM facts UNION ALL
        SELECT 'terminal', completed_run_count = 1 AND done_job_count = 1
            AND dead_job_count = 0 FROM facts
    ) AS values(name, ok)
),
evidence AS (
    SELECT jsonb_build_object(
        'version', 1,
        'actor_user_id', parameter.actor_user_id,
        'workspace_id', parameter.workspace_id,
        'document_id', parameter.document_id,
        'run_id', parameter.run_id,
        'cost_cap_cny', parameter.cost_cap_cny,
        'facts', to_jsonb(facts),
        'actions', (
            SELECT coalesce(jsonb_agg(jsonb_build_object(
                'id', action.id,
                'status', action.status,
                'args_digest', action.args_digest,
                'target_digest', action.target_digest,
                'approval_binding_version', action.approval_binding_version,
                'approval_binding_digest', action.approval_binding_digest,
                'idempotency_key', action.idempotency_key
            ) ORDER BY action.id), '[]'::jsonb) FROM selected_action AS action
        ),
        'model_invocations', (
            SELECT coalesce(jsonb_agg(jsonb_build_object(
                'id', invocation.id,
                'kind', invocation.invocation_kind,
                'provider', invocation.provider,
                'model', invocation.model,
                'status', invocation.status,
                'pricing_version', invocation.pricing_version,
                'currency', invocation.currency,
                'estimated_cost', invocation.estimated_cost,
                'trace_ids', invocation.trace_ids
            ) ORDER BY invocation.created_at, invocation.id), '[]'::jsonb)
            FROM llm_invocations AS invocation
            WHERE invocation.workspace_id = parameter.workspace_id
              AND invocation.run_id = parameter.run_id
        ),
        'events', (
            SELECT coalesce(jsonb_agg(jsonb_build_object(
                'seq', event.seq, 'type', event.type, 'version', event.version
            ) ORDER BY event.seq), '[]'::jsonb)
            FROM run_events AS event
            WHERE event.workspace_id = parameter.workspace_id
              AND event.run_id = parameter.run_id
        ),
        'checks', (SELECT jsonb_object_agg(name, ok ORDER BY name) FROM checks),
        'accepted', (SELECT bool_and(ok) FROM checks)
    ) AS value
    FROM parameters AS parameter CROSS JOIN facts
)
SELECT value::text AS evidence, (SELECT bool_and(ok) FROM checks)::text AS accepted
FROM evidence
\gset gate87_

\echo :gate87_evidence
\if :gate87_accepted
COMMIT;
\else
ROLLBACK;
\quit 3
\endif
