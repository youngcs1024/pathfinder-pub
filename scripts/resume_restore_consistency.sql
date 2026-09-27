\set ON_ERROR_STOP on
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '30s';
WITH table_hashes AS (
    SELECT schemaname || '.' || tablename AS name,
           query_to_xml(format(
               'SELECT count(*) AS rows, md5(coalesce(string_agg(md5(to_jsonb(t)::text), '''' ORDER BY md5(to_jsonb(t)::text)), '''')) AS digest FROM %I.%I t',
               schemaname, tablename), false, true, '') AS value
    FROM pg_tables
    WHERE schemaname IN ('public', 'pathfinder_checkpoint')
), fixture AS (
    SELECT s.id AS session_id, s.workspace_id, s.owner_user_id, s.current_version_id,
           (SELECT jsonb_agg(jsonb_build_object(
               'version_id', v.id, 'artifact_id', v.artifact_id, 'tex_sha256', a.tex_sha256,
               'version', v.version, 'confirmed', EXISTS (
                   SELECT 1 FROM resume_confirmations c WHERE c.version_id = v.id
                   AND c.workspace_id = v.workspace_id AND c.tex_sha256 = a.tex_sha256
                   AND c.artifact_id = a.id
               )) ORDER BY v.version)
            FROM resume_versions v JOIN resume_tex_artifacts a
              ON a.id = v.artifact_id AND a.workspace_id = v.workspace_id
            WHERE v.session_id = s.id AND v.workspace_id = s.workspace_id) AS versions
    FROM resume_sessions s WHERE s.run_id = :'fixture_run_id'::uuid
)
SELECT jsonb_build_object(
    'schema', jsonb_build_object('alembic_version', (SELECT version_num FROM alembic_version)),
    'global_counts', (SELECT jsonb_object_agg(name, jsonb_build_object(
        'rows', ((xpath('/row/rows/text()', value))[1]::text)::bigint,
        'digest', (xpath('/row/digest/text()', value))[1]::text)) FROM table_hashes),
    'fixture', (SELECT to_jsonb(f) FROM fixture f),
    'checks', jsonb_build_object(
        'fixture_unique', (SELECT count(*) = 1 FROM fixture),
        'fixture_confirmed', EXISTS (SELECT 1 FROM fixture f JOIN resume_confirmations c
            ON c.session_id = f.session_id AND c.version_id = f.current_version_id
            AND c.workspace_id = f.workspace_id),
        'no_pending_runs', NOT EXISTS (SELECT 1 FROM runs WHERE status IN ('queued','running','waiting_approval')),
        'no_pending_jobs', NOT EXISTS (SELECT 1 FROM run_jobs WHERE status IN ('queued','leased')),
        'no_executing_actions', NOT EXISTS (SELECT 1 FROM action_intents WHERE status = 'executing'),
        'no_executing_tools', NOT EXISTS (SELECT 1 FROM tool_invocations WHERE status = 'executing'),
        'artifact_integrity', NOT EXISTS (SELECT 1 FROM resume_tex_artifacts
            WHERE encode(sha256(tex_bytes), 'hex') <> tex_sha256)
    )
);
COMMIT;
