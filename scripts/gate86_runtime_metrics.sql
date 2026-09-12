\set ON_ERROR_STOP on

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;

WITH
observation AS (
    SELECT coalesce(
        nullif(current_setting('pathfinder.gate86_observed_at', true), '')::timestamptz,
        statement_timestamp()
    ) AS observed_at
),
queue_metrics AS (
    SELECT
        count(*) FILTER (WHERE job.status = 'queued')::bigint AS queued_total,
        count(*) FILTER (
            WHERE job.status = 'queued' AND job.available_at <= observation.observed_at
        )::bigint AS ready_queued_total,
        floor(extract(epoch FROM (
            observation.observed_at
            - min(job.created_at) FILTER (WHERE job.status = 'queued')
        )))::bigint AS oldest_queued_age_seconds,
        floor(extract(epoch FROM (
            observation.observed_at
            - min(job.created_at) FILTER (
                WHERE job.status = 'queued' AND job.available_at <= observation.observed_at
            )
        )))::bigint AS oldest_ready_queued_age_seconds,
        count(*) FILTER (WHERE job.status = 'leased')::bigint AS leased_total,
        count(*) FILTER (
            WHERE job.status = 'leased'
              AND job.lease_expires_at <= observation.observed_at
        )::bigint AS expired_leases_total,
        count(*) FILTER (WHERE job.status = 'dead')::bigint AS dead_total
    FROM observation
    LEFT JOIN run_jobs AS job ON true
    GROUP BY observation.observed_at
),
recent_runs AS (
    SELECT run.*
    FROM runs AS run
    CROSS JOIN observation
    WHERE run.status IN ('completed', 'failed', 'cancelled')
      AND run.finished_at >= observation.observed_at - interval '1 hour'
      AND run.finished_at <= observation.observed_at
),
run_metrics AS (
    SELECT
        count(*) FILTER (WHERE status = 'completed')::bigint AS completed,
        count(*) FILTER (WHERE status = 'failed')::bigint AS failed,
        count(*) FILTER (WHERE status = 'cancelled')::bigint AS cancelled,
        count(*) FILTER (WHERE status IN ('completed', 'failed'))::bigint
            AS failure_sample_count,
        CASE
            WHEN count(*) FILTER (WHERE status IN ('completed', 'failed')) = 0 THEN NULL
            ELSE (
                count(*) FILTER (WHERE status = 'failed')::numeric
                / count(*) FILTER (WHERE status IN ('completed', 'failed'))
            )
        END AS failure_rate
    FROM recent_runs
),
run_latency_samples AS (
    SELECT extract(epoch FROM (finished_at - started_at)) * 1000.0 AS latency_ms
    FROM recent_runs
    WHERE started_at IS NOT NULL
),
run_latency AS (
    SELECT
        count(*)::bigint AS terminal_run_count,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
    FROM run_latency_samples
),
recent_llm AS (
    SELECT invocation.*
    FROM llm_invocations AS invocation
    CROSS JOIN observation
    WHERE invocation.created_at >= observation.observed_at - interval '1 hour'
      AND invocation.created_at <= observation.observed_at
),
llm_errors AS (
    SELECT coalesce(jsonb_object_agg(error_category, error_count), '{}'::jsonb) AS value
    FROM (
        SELECT error_category, count(*)::bigint AS error_count
        FROM recent_llm
        WHERE status = 'failed' AND error_category IS NOT NULL
        GROUP BY error_category
        ORDER BY error_category
    ) AS grouped
),
llm_metrics AS (
    SELECT
        count(*) FILTER (WHERE status = 'succeeded')::bigint AS succeeded,
        count(*) FILTER (WHERE status = 'failed')::bigint AS failed,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL) AS p50_ms,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL) AS p95_ms
    FROM recent_llm
),
recent_tools AS (
    SELECT invocation.*
    FROM tool_invocations AS invocation
    CROSS JOIN observation
    WHERE invocation.created_at >= observation.observed_at - interval '1 hour'
      AND invocation.created_at <= observation.observed_at
),
tool_metrics AS (
    SELECT
        count(*) FILTER (WHERE status = 'succeeded')::bigint AS succeeded,
        count(*) FILTER (WHERE status = 'failed')::bigint AS failed,
        count(*) FILTER (WHERE status = 'outcome_unknown')::bigint AS outcome_unknown,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL) AS p50_ms,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE latency_ms IS NOT NULL) AS p95_ms
    FROM recent_tools
),
cost_metrics AS (
    SELECT
        count(*) FILTER (
            WHERE invocation.estimated_cost IS NOT NULL
              AND invocation.currency = 'CNY'
        )::bigint AS priced_attempts,
        sum(invocation.estimated_cost) FILTER (
            WHERE invocation.estimated_cost IS NOT NULL
              AND invocation.currency = 'CNY'
        ) AS known_estimated_cny_total,
        count(*) FILTER (
            WHERE invocation.provider <> 'fake'
              AND invocation.status = 'succeeded'
              AND invocation.estimated_cost IS NULL
        )::bigint AS unpriced_non_fake_succeeded_attempts
    FROM llm_invocations AS invocation
    CROSS JOIN observation
    WHERE invocation.created_at >= observation.observed_at - interval '24 hours'
      AND invocation.created_at <= observation.observed_at
)
SELECT jsonb_build_object(
    'version', 1,
    'observed_at', observation.observed_at,
    'windows', jsonb_build_object('runs_latency_errors', '1 hour', 'cost', '24 hours'),
    'queue', jsonb_build_object(
        'queued_total', queue_metrics.queued_total,
        'ready_queued_total', queue_metrics.ready_queued_total,
        'oldest_queued_age_seconds', queue_metrics.oldest_queued_age_seconds,
        'oldest_ready_queued_age_seconds', queue_metrics.oldest_ready_queued_age_seconds,
        'leased_total', queue_metrics.leased_total,
        'expired_leases_total', queue_metrics.expired_leases_total,
        'dead_total', queue_metrics.dead_total
    ),
    'runs', jsonb_build_object(
        'completed', run_metrics.completed,
        'failed', run_metrics.failed,
        'cancelled', run_metrics.cancelled,
        'failure_sample_count', run_metrics.failure_sample_count,
        'failure_rate', run_metrics.failure_rate
    ),
    'latency', jsonb_build_object(
        'terminal_run_count', run_latency.terminal_run_count,
        'run_p50_ms', run_latency.p50_ms,
        'run_p95_ms', run_latency.p95_ms,
        'llm_p50_ms', llm_metrics.p50_ms,
        'llm_p95_ms', llm_metrics.p95_ms,
        'tool_p50_ms', tool_metrics.p50_ms,
        'tool_p95_ms', tool_metrics.p95_ms
    ),
    'llm', jsonb_build_object(
        'succeeded', llm_metrics.succeeded,
        'failed', llm_metrics.failed,
        'error_categories', llm_errors.value
    ),
    'tools', jsonb_build_object(
        'succeeded', tool_metrics.succeeded,
        'failed', tool_metrics.failed,
        'outcome_unknown', tool_metrics.outcome_unknown
    ),
    'cost_24h', jsonb_build_object(
        'priced_attempts', cost_metrics.priced_attempts,
        'known_estimated_cny_total', cost_metrics.known_estimated_cny_total,
        'unpriced_non_fake_succeeded_attempts',
            cost_metrics.unpriced_non_fake_succeeded_attempts
    ),
    'storage', jsonb_build_object(
        'database_bytes', pg_database_size(current_database())
    )
)::text
FROM observation
CROSS JOIN queue_metrics
CROSS JOIN run_metrics
CROSS JOIN run_latency
CROSS JOIN llm_metrics
CROSS JOIN llm_errors
CROSS JOIN tool_metrics
CROSS JOIN cost_metrics;
