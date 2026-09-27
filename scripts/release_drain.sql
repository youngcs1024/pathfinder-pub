-- Compatible with the frozen 0015 schema. Counts only; no business bodies or writes.
BEGIN READ ONLY;
SET LOCAL statement_timeout = '5s';
SELECT
    (SELECT count(*) FROM runs WHERE status IN ('queued', 'running', 'waiting_approval')),
    (SELECT count(*) FROM run_jobs WHERE status IN ('queued', 'leased')),
    (SELECT count(*) FROM action_intents WHERE status = 'executing'),
    (SELECT count(*) FROM tool_invocations WHERE status = 'executing'),
    (SELECT count(*) FROM runs WHERE status = 'waiting_approval');
COMMIT;
