#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly METRICS_SQL="$REPOSITORY_ROOT/scripts/gate86_runtime_metrics.sql"
readonly RECOVERY_SQL="$REPOSITORY_ROOT/scripts/gate86_recovery_state.sql"
readonly LOCK_ROOT="${TMPDIR:-/tmp}"

stage="initialization"
mode="${1:-}"
lock_directory=""
lock_acquired=0
work_directory=""
worker_started=0
outage_started=0
stop_attempt="not-required"
declare -a COMPOSE_COMMAND=()

log() {
    printf 'gate86: %s\n' "$*"
}

diagnose() {
    printf 'gate86: stage=%s mode=%s project=%s\n' \
        "$stage" "${mode:-missing}" "${COMPOSE_PROJECT_NAME:-missing}" >&2
    printf 'gate86: work_directory=%s outage_started=%s worker_started=%s\n' \
        "${work_directory:-not-created}" "$outage_started" "$worker_started" >&2
    printf 'gate86: worker_stop_attempt=%s\n' "$stop_attempt" >&2
    printf '%s\n' \
        'gate86: evidence and project state are retained; no automatic retry or data repair ran' >&2
    if [[ ${#COMPOSE_COMMAND[@]} -gt 0 ]]; then
        printf '%s\n' 'gate86: exact project container state follows (read-only)' >&2
        if ! "${COMPOSE_COMMAND[@]}" ps --all --format json >&2; then
            printf '%s\n' 'gate86: project container state query failed' >&2
        fi
    fi
}

fail() {
    local failed_stage="$stage"
    printf 'gate86: error: %s\n' "$1" >&2
    best_effort_stop_worker
    stage="$failed_stage"
    diagnose
    exit 1
}

best_effort_stop_worker() {
    if [[ "$worker_started" -eq 1 && ${#COMPOSE_COMMAND[@]} -gt 0 ]]; then
        stage="failure-safe-stop-worker"
        if "${COMPOSE_COMMAND[@]}" stop --timeout 30 worker >/dev/null 2>&1; then
            stop_attempt="worker-stopped"
        else
            stop_attempt="worker-stop-failed"
        fi
        worker_started=0
    fi
}

cleanup_lock() {
    if [[ "$lock_acquired" -eq 1 && -n "$lock_directory" ]]; then
        if [[ -f "$lock_directory/owner" ]]; then
            rm -f -- "$lock_directory/owner"
        fi
        rmdir -- "$lock_directory" 2>/dev/null || true
        lock_acquired=0
    fi
}

interrupted() {
    local signal_name="$1"
    local interrupted_stage="$stage"
    printf 'gate86: interrupted by %s during stage=%s; no later stage was started\n' \
        "$signal_name" "$interrupted_stage" >&2
    best_effort_stop_worker
    stage="$interrupted_stage"
    diagnose
    exit 130
}

unexpected_error() {
    local exit_code="$1"
    local failed_stage="$stage"
    printf 'gate86: unexpected command failure exit_code=%s\n' "$exit_code" >&2
    best_effort_stop_worker
    stage="$failed_stage"
    diagnose
    exit "$exit_code"
}

trap cleanup_lock EXIT
trap 'unexpected_error $?' ERR
trap 'interrupted SIGINT' INT
trap 'interrupted SIGTERM' TERM
trap 'interrupted SIGHUP' HUP

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
}

require_nonempty() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "$name is required"
}

require_uuid() {
    local name="$1" value="$2"
    python3 -c '
import sys, uuid
try:
    value = uuid.UUID(sys.argv[1])
except (ValueError, AttributeError):
    raise SystemExit(1)
raise SystemExit(0 if str(value) == sys.argv[1].lower() else 1)
' "$value" || fail "$name is not a canonical UUID"
}

validate_project_name() {
    [[ "$COMPOSE_PROJECT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] \
        || fail "COMPOSE_PROJECT_NAME contains unsafe characters"
}

validate_deadline() {
    PF_GATE86_DEADLINE_SECONDS="${PF_GATE86_DEADLINE_SECONDS:-180}"
    if [[ ! "$PF_GATE86_DEADLINE_SECONDS" =~ ^[0-9]+$ \
        || "$PF_GATE86_DEADLINE_SECONDS" -lt 30 \
        || "$PF_GATE86_DEADLINE_SECONDS" -gt 900 ]]; then
        fail "PF_GATE86_DEADLINE_SECONDS must be an integer between 30 and 900"
    fi
    export PF_GATE86_DEADLINE_SECONDS
}

validate_compose_files() {
    require_nonempty PF_GATE86_COMPOSE_FILES
    local item canonical
    IFS=':' read -r -a compose_files <<<"$PF_GATE86_COMPOSE_FILES"
    [[ "${#compose_files[@]}" -gt 0 ]] || fail "PF_GATE86_COMPOSE_FILES is empty"
    COMPOSE_COMMAND=(docker compose --project-name "$COMPOSE_PROJECT_NAME")
    for item in "${compose_files[@]}"; do
        [[ -n "$item" ]] || fail "PF_GATE86_COMPOSE_FILES contains an empty entry"
        canonical="$(realpath -e -- "$item")" \
            || fail "a configured Compose file is missing"
        [[ -f "$canonical" ]] || fail "every configured Compose path must be a file"
        COMPOSE_COMMAND+=(--file "$canonical")
    done
}

validate_evidence_directory() {
    local lexical canonical repository mode_bits owner
    require_nonempty PF_GATE86_EVIDENCE_DIR
    lexical="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' \
        "$PF_GATE86_EVIDENCE_DIR")"
    canonical="$(realpath -e -- "$PF_GATE86_EVIDENCE_DIR")" \
        || fail "PF_GATE86_EVIDENCE_DIR must already exist"
    repository="$(realpath -e -- "$REPOSITORY_ROOT")"
    [[ -d "$canonical" ]] || fail "PF_GATE86_EVIDENCE_DIR must be a directory"
    if [[ "$lexical" == "$repository" || "$lexical" == "$repository/"* \
        || "$canonical" == "$repository" || "$canonical" == "$repository/"* ]]; then
        fail "PF_GATE86_EVIDENCE_DIR must be outside the repository"
    fi
    owner="$(stat -c '%u' -- "$canonical")"
    [[ "$owner" == "$(id -u)" ]] || fail "PF_GATE86_EVIDENCE_DIR must be current-user owned"
    mode_bits=$((8#$(stat -c '%a' -- "$canonical")))
    (( (mode_bits & 077) == 0 )) \
        || fail "PF_GATE86_EVIDENCE_DIR must not grant group or other permissions"
    PF_GATE86_EVIDENCE_DIR="$canonical"
    export PF_GATE86_EVIDENCE_DIR
}

initialize_common() {
    stage="static-preflight"
    case "$mode" in
        metrics|shared-host) ;;
        *) fail "usage: scripts/runtime_rehearsal.sh metrics|shared-host" ;;
    esac
    require_nonempty COMPOSE_PROJECT_NAME
    for command_name in docker id python3 realpath stat; do
        require_command "$command_name"
    done
    [[ -f "$METRICS_SQL" ]] || fail "runtime metrics SQL is missing"
    [[ -f "$RECOVERY_SQL" ]] || fail "recovery state SQL is missing"
    validate_project_name
    validate_deadline
    validate_compose_files
}

compose() {
    "${COMPOSE_COMMAND[@]}" "$@"
}

postgres_query_file() {
    local sql_file="$1"
    shift
    compose exec -T postgres sh -c \
        'psql -X -qAt --set ON_ERROR_STOP=on --single-transaction --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" "$@"' \
        gate86-psql "$@" <"$sql_file"
}

postgres_scalar() {
    local query="$1"
    compose exec -T postgres sh -c \
        'psql -X -qAt --set ON_ERROR_STOP=on --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --command "$1"' \
        gate86-psql "$query"
}

runtime_metrics_json() {
    local metrics_json disk_text
    metrics_json="$(postgres_query_file "$METRICS_SQL")"
    disk_text="$(compose exec -T postgres df -Pk /var/lib/postgresql/data)"
    python3 -c '
import json, sys
metrics = json.loads(sys.argv[1])
lines = [line.split() for line in sys.argv[2].splitlines() if line.strip()]
if len(lines) < 2 or len(lines[-1]) < 6:
    raise SystemExit("invalid PostgreSQL filesystem report")
row = lines[-1]
metrics["storage"]["filesystem"] = {
    "kilobytes_total": int(row[1]),
    "kilobytes_used": int(row[2]),
    "kilobytes_available": int(row[3]),
    "percent_used": int(row[4].rstrip("%")),
    "mountpoint": row[5],
}
print(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
' "$metrics_json" "$disk_text"
}

run_metrics_mode() {
    stage="metrics-read-only"
    compose config --quiet >/dev/null
    runtime_metrics_json
}

acquire_deployment_lock() {
    stage="deployment-mutation-lock"
    lock_directory="${LOCK_ROOT%/}/pathfinder-release-${COMPOSE_PROJECT_NAME}.lock"
    if ! mkdir -- "$lock_directory" 2>/dev/null; then
        fail "release/rehearsal lock exists: $lock_directory"
    fi
    lock_acquired=1
    printf 'pid=%s\noperation=gate86-shared-host\nproject=%s\nstarted_utc=%s\n' \
        "$$" "$COMPOSE_PROJECT_NAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$lock_directory/owner"
}

effective_metadata() {
    compose config --format json | python3 -c '
import json, sys
config = json.load(sys.stdin)
services = config.get("services", {})
required = {"postgres", "api", "worker", "caddy"}
if set(services) != required:
    raise SystemExit("effective Compose services are not exactly Pathfinder core")
ports = []
for item in services["caddy"].get("ports", []):
    if not isinstance(item, dict):
        raise SystemExit("Caddy port metadata is not structured")
    ports.append({
        "target": int(item["target"]),
        "published": int(item["published"]),
        "host_ip": item.get("host_ip", ""),
        "protocol": item.get("protocol", "tcp"),
    })
volumes = sorted(config.get("volumes", {}).keys())
print(json.dumps({
    "bindings": sorted(ports, key=lambda value: (value["target"], value["published"])),
    "volume_keys": volumes,
    "api_image": services["api"].get("image"),
    "worker_image": services["worker"].get("image"),
}, sort_keys=True, separators=(",", ":")))
'
}

one_container_id() {
    local service="$1" ids count
    ids="$(compose ps --all --quiet "$service")"
    count="$(awk 'NF {count += 1} END {print count + 0}' <<<"$ids")"
    [[ "$count" -eq 1 ]] || fail "$service cardinality must be exactly one (found $count)"
    awk 'NF {print; exit}' <<<"$ids"
}

container_state() {
    docker inspect --format \
        '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.Config.Image}}|{{.Image}}' \
        "$1"
}

require_recovered_service() {
    local service="$1" expected_ref="$2" expected_image_id="${3:-}"
    local id state health image_ref image_id
    id="$(one_container_id "$service")"
    IFS='|' read -r state health image_ref image_id <<<"$(container_state "$id")"
    [[ "$state" == "running" && "$health" == "healthy" ]] \
        || fail "$service did not recover healthy"
    [[ "$image_ref" == "$expected_ref" ]] \
        || fail "$service recovered with an unexpected image reference"
    if [[ -n "$expected_image_id" && "$image_id" != "$expected_image_id" ]]; then
        fail "$service recovered with an unexpected image identity"
    fi
}

actual_caddy_binding() {
    docker inspect --format '{{json .NetworkSettings.Ports}}' "$1" | python3 -c '
import json, sys
ports = json.load(sys.stdin)
values = []
for key, bindings in ports.items():
    target, protocol = key.split("/", 1)
    for binding in bindings or []:
        values.append({
            "target": int(target),
            "published": int(binding["HostPort"]),
            "host_ip": binding["HostIp"],
            "protocol": protocol,
        })
print(json.dumps(sorted(values, key=lambda value: (value["target"], value["published"])),
                 sort_keys=True, separators=(",", ":")))
'
}

validate_safe_binding() {
    python3 -c '
import json, sys
bindings = json.loads(sys.argv[1])
targets = {entry["target"] for entry in bindings}
if not bindings or not targets.issubset({80, 443}):
    raise SystemExit("Caddy may publish only container ports 80 and/or 443")
if any(entry["host_ip"] not in {"127.0.0.1", "::1"} for entry in bindings):
    raise SystemExit("shared-host Caddy binding must be loopback-only")
' "$1" || fail "unsafe shared-host Caddy binding"
}

resolve_volume_names() {
    local config_json="$1" expected_keys actual names
    expected_keys="$(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["volume_keys"]))' \
        "$config_json")"
    actual="$(docker volume ls --filter "label=com.docker.compose.project=$COMPOSE_PROJECT_NAME" \
        --format '{{.Name}}')"
    names="$(python3 -c '
import sys
project = sys.argv[1]
keys = {line for line in sys.argv[2].splitlines() if line}
actual = {line for line in sys.argv[3].splitlines() if line}
expected = {f"{project}_{key}" for key in keys}
if actual != expected:
    raise SystemExit("project volume set does not match effective Compose config")
print("\n".join(sorted(actual)))
' "$COMPOSE_PROJECT_NAME" "$expected_keys" "$actual")" \
        || fail "project durable volumes are incomplete or unexpected"
    printf '%s\n' "$names"
}

deployment_preflight() {
    stage="deployment-preflight"
    local config_json service id state health image_ref image_id
    local api_ref api_image_id worker_ref worker_image_id caddy_id
    compose config --quiet >/dev/null
    config_json="$(effective_metadata)"
    for service in postgres api worker caddy; do
        id="$(one_container_id "$service")"
        IFS='|' read -r state health image_ref image_id <<<"$(container_state "$id")"
        [[ "$state" == "running" ]] || fail "$service must be running"
        if [[ "$service" != "caddy" && "$health" != "healthy" ]]; then
            fail "$service must be healthy"
        fi
        case "$service" in
            api) api_ref="$image_ref"; api_image_id="$image_id" ;;
            worker) worker_ref="$image_ref"; worker_image_id="$image_id" ;;
            caddy) caddy_id="$id" ;;
        esac
    done
    [[ "$api_ref" == "$worker_ref" && "$api_image_id" == "$worker_image_id" ]] \
        || fail "API and worker application image identity differs"
    local configured_api configured_worker
    configured_api="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["api_image"])' "$config_json")"
    configured_worker="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["worker_image"])' "$config_json")"
    [[ "$configured_api" == "$api_ref" && "$configured_worker" == "$worker_ref" ]] \
        || fail "running application image differs from effective Compose config"

    local modes current_revision candidate_head active_json planned_binding current_binding
    modes="$(compose exec -T worker python -c \
        'from app.config import Settings; import json; s=Settings(); print(json.dumps([s.llm_mode,s.search_mode,s.auth_mode,s.trace_mode]))')"
    [[ "$modes" == '["fake", "fake", "fake", "off"]' \
        || "$modes" == '["fake","fake","fake","off"]' ]] \
        || fail "shared-host rehearsal requires fake/fake/fake/off"
    current_revision="$(postgres_scalar 'SELECT version_num FROM alembic_version')"
    candidate_head="$(compose exec -T api alembic heads | awk 'NF {print $1; exit}')"
    [[ -n "$current_revision" && "$current_revision" == "$candidate_head" ]] \
        || fail "database revision differs from candidate head"
    active_json="$(postgres_scalar \
        "SELECT json_build_object('active_runs',count(*) FILTER (WHERE status IN ('queued','running','waiting_approval')),'active_jobs',(SELECT count(*) FROM run_jobs WHERE status IN ('queued','leased')))::text FROM runs")"
    python3 -c '
import json,sys
value=json.loads(sys.argv[1])
raise SystemExit(0 if value == {"active_runs": 0, "active_jobs": 0} else 1)
' "$active_json" || fail "unrelated active run/job state exists"

    planned_binding="$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["bindings"],sort_keys=True,separators=(",",":")))' "$config_json")"
    current_binding="$(actual_caddy_binding "$caddy_id")"
    [[ "$planned_binding" == "$current_binding" ]] \
        || fail "effective/current Caddy binding mismatch"
    validate_safe_binding "$current_binding"
    PRE_CADDY_BINDING="$current_binding"
    APPLICATION_IMAGE_REF="$api_ref"
    APPLICATION_IMAGE_ID="$api_image_id"
    POSTGRES_IMAGE_REF="$(docker inspect --format '{{.Config.Image}}' "$(one_container_id postgres)")"
    POSTGRES_VERSION="$(compose exec -T postgres postgres --version)"
    MIGRATION_REVISION="$current_revision"
    PROJECT_VOLUMES="$(resolve_volume_names "$config_json")"
    export PRE_CADDY_BINDING APPLICATION_IMAGE_REF APPLICATION_IMAGE_ID
    export POSTGRES_IMAGE_REF POSTGRES_VERSION MIGRATION_REVISION PROJECT_VOLUMES
}

api_request() {
    local method="$1" path="$2" body="${3:-}"
    compose exec -T api python -c '
import sys, urllib.error, urllib.request
method, path, body = sys.argv[1:]
request = urllib.request.Request(
    "http://127.0.0.1:8000" + path,
    data=body.encode() if body else None,
    headers={"Accept": "application/json", "Content-Type": "application/json"},
    method=method,
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        sys.stdout.write(response.read().decode())
except urllib.error.HTTPError as error:
    sys.stdout.write(error.read().decode())
    raise SystemExit(1)
' "$method" "$path" "$body"
}

json_value() {
    local value="$1" expression="$2"
    python3 -c 'import json,sys; value=json.loads(sys.argv[1]); print(eval(sys.argv[2], {"__builtins__": {}}, {"value": value}))' \
        "$value" "$expression"
}

poll_run_status() {
    local workspace_id="$1" run_id="$2" expected="$3"
    local deadline status response
    deadline=$(( $(date +%s) + PF_GATE86_DEADLINE_SECONDS ))
    while (( $(date +%s) <= deadline )); do
        response="$(api_request GET "/api/v1/workspaces/$workspace_id/runs/$run_id")"
        status="$(json_value "$response" 'value["status"]')"
        if [[ "$status" == "$expected" ]]; then
            printf '%s\n' "$response"
            return 0
        fi
        if [[ "$status" =~ ^(completed|failed|cancelled)$ && "$status" != "$expected" ]]; then
            fail "run $run_id reached unexpected terminal status $status"
        fi
        sleep 0.5
    done
    fail "run $run_id did not reach $expected before deadline"
}

wait_worker_stopped() {
    local deadline running_ids
    deadline=$(( $(date +%s) + PF_GATE86_DEADLINE_SECONDS ))
    while (( $(date +%s) <= deadline )); do
        running_ids="$(compose ps --status running --quiet worker)"
        if [[ -z "$running_ids" ]]; then
            return 0
        fi
        sleep 0.5
    done
    fail "real worker did not stop before crash-window claim"
}

create_run() {
    local workspace_id="$1" mode_name="$2" query="$3" document_id="${4:-}"
    local body response
    body="$(python3 -c '
import json,sys
value={"mode":sys.argv[1],"query":sys.argv[2]}
if sys.argv[3]: value["resume_document_id"]=sys.argv[3]
print(json.dumps(value,separators=(",",":")))
' "$mode_name" "$query" "$document_id")"
    response="$(api_request POST "/api/v1/workspaces/$workspace_id/runs" "$body")"
    json_value "$response" 'value["run_id"]'
}

action_id_for_run() {
    postgres_scalar "SELECT id FROM action_intents WHERE run_id = '$1'::uuid"
}

approve_action() {
    local workspace_id="$1" action_id="$2" marker="$3"
    local review version body response readback
    review="$(api_request GET "/api/v1/workspaces/$workspace_id/action-intents/$action_id")"
    version="$(json_value "$review" 'value["approval_request"]["version"]')"
    body="$(python3 -c 'import json,sys; print(json.dumps({"decision":"approve","expected_version":int(sys.argv[1]),"reason":sys.argv[2]},separators=(",",":")))' \
        "$version" "$marker")"
    if response="$(api_request POST "/api/v1/workspaces/$workspace_id/action-intents/$action_id/decision" "$body")"; then
        [[ "$(json_value "$response" 'value["decision"]')" == "approve" ]] \
            || fail "decision response is not exact approve"
        return 0
    fi
    readback="$(api_request GET "/api/v1/workspaces/$workspace_id/action-intents/$action_id")"
    python3 -c '
import json,sys
value=json.loads(sys.argv[1]); decision=value.get("decision")
raise SystemExit(0 if decision and decision.get("decision")=="approve" else 1)
' "$readback" || fail "decision outcome is uncertain and persisted approve is absent"
}

create_document() {
    local workspace_id="$1" actor_id="$2" marker="$3" output
    output="$(compose exec -T api sh -c '
set -eu
document_path="/tmp/${3}.md"
printf "%s\n" "# Synthetic Gate 8.6 resume" "Marker: ${3}" "Backend systems research." >"$document_path"
python -m app.cli.ingest_documents --workspace-id "$1" --actor-user-id "$2" "$document_path"
status=$?
rm -f -- "$document_path"
exit "$status"
' gate86-ingest "$workspace_id" "$actor_id" "$marker")"
    awk -F= '/^document_id=/{print $2}' <<<"$output"
}

claim_crash_window() {
    local worker_marker="$1"
    compose run --rm --no-deps worker python -c '
import asyncio, json, sys
from datetime import UTC, datetime, timedelta
from app.config import Settings
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.session import create_database_engine, create_session_factory

async def main():
    engine=create_database_engine(Settings().database_url)
    try:
        store=SqlAlchemyWorkerJobStore(create_session_factory(engine), lambda attempt: timedelta(seconds=attempt))
        claimed=await store.claim_due_job(worker_id=sys.argv[1], now=datetime.now(UTC), lease_duration=timedelta(seconds=30))
        if claimed is None:
            raise SystemExit("no due job was claimed")
        print(json.dumps({"run_id":str(claimed.run_id),"attempt":claimed.attempt,"lease_expires_at":claimed.lease_expires_at.isoformat()}))
    finally:
        await engine.dispose()
asyncio.run(main())
' "$worker_marker"
}

recovery_snapshot() {
    postgres_query_file "$RECOVERY_SQL" \
        --set "baseline_run_id=$BASELINE_RUN_ID" \
        --set "waiting_run_id=$WAITING_RUN_ID" \
        --set "leased_run_id=$LEASED_RUN_ID" \
        --set "queued_run_id=$QUEUED_RUN_ID"
}

write_new_file() {
    local path="$1" value="$2"
    [[ ! -e "$path" ]] || fail "evidence path already exists: $path"
    printf '%s\n' "$value" >"$path"
    chmod 0600 "$path"
}

validate_snapshot() {
    local phase_name="$1" snapshot_path="$2" reference_path="${3:-}"
    python3 - "$phase_name" "$snapshot_path" "$reference_path" "$LEASED_BY" <<'PY'
import datetime as dt
import json
import os
import sys

phase, path, reference_path, leased_by = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
reference = json.load(open(reference_path, encoding="utf-8")) if reference_path else None
fixtures = value["fixtures"]
if not all(value["checks"].values()):
    raise SystemExit("generic recovery snapshot checks failed")

def one(name, field):
    items = fixtures[name][field]
    if len(items) != 1:
        raise SystemExit(f"{name} expected one {field}")
    return items[0]

def prefix(current, earlier):
    return current[:len(earlier)] == earlier

def exactly_one_succeeded_irreversible(fixture):
    values = [
        item for item in fixture["tool_invocations"]
        if item["effect"] == "irreversible" and item["status"] == "succeeded"
    ]
    return len(values) == 1

def mock_matches_action(fixture):
    if len(fixture["actions"]) != 1 or len(fixture["mock_submissions"]) != 1:
        return False
    return (
        fixture["actions"][0]["idempotency_key"]
        == fixture["mock_submissions"][0]["idempotency_key"]
    )

if phase == "pre":
    baseline = fixtures["baseline"]
    waiting = fixtures["waiting"]
    leased = fixtures["leased"]
    queued = fixtures["queued"]
    if baseline["run"]["status"] != "completed" or baseline["job"]["status"] != "done":
        raise SystemExit("baseline is not terminal")
    if one("baseline", "approval_requests")["status"] != "consumed":
        raise SystemExit("baseline request is not consumed")
    if (
        baseline["decision_count"] != 1
        or baseline["approve_decision_count"] != 1
        or one("baseline", "actions")["status"] != "succeeded"
    ):
        raise SystemExit("baseline approval/action is invalid")
    if (
        not mock_matches_action(baseline)
        or not exactly_one_succeeded_irreversible(baseline)
        or not baseline["checkpoints"]
    ):
        raise SystemExit("baseline Mock/checkpoint evidence is invalid")
    if waiting["run"]["status"] != "waiting_approval" or waiting["job"]["status"] != "done":
        raise SystemExit("waiting fixture is not paused")
    request = one("waiting", "approval_requests")
    if (
        request["status"] != "pending"
        or waiting["decision_count"] != 0
        or waiting["approve_decision_count"] != 0
    ):
        raise SystemExit("waiting request/decision is invalid")
    if one("waiting", "actions")["status"] != "proposed" or waiting["mock_submissions"]:
        raise SystemExit("waiting action/Mock state is invalid")
    if not waiting["checkpoints"]:
        raise SystemExit("waiting checkpoint is missing")
    expiry = dt.datetime.fromisoformat(request["expires_at"].replace("Z", "+00:00"))
    if expiry <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(
        seconds=int(os.environ["PF_GATE86_DEADLINE_SECONDS"]) + 30
    ):
        raise SystemExit("waiting approval expiry is too near")
    if leased["run"]["status"] != "queued" or leased["job"]["status"] != "leased":
        raise SystemExit("leased crash-window state is invalid")
    if leased["job"]["attempt"] != 1 or not leased["job"]["owner_present"]:
        raise SystemExit("leased crash-window identity is invalid")
    if leased["job"]["leased_by"] != leased_by:
        raise SystemExit("leased_by marker differs")
    lease_expiry = dt.datetime.fromisoformat(
        leased["job"]["lease_expires_at"].replace("Z", "+00:00")
    )
    if lease_expiry <= dt.datetime.now(dt.timezone.utc):
        raise SystemExit("leased crash-window lease is not initially live")
    if [event["type"] for event in leased["events"]] != ["run.created"]:
        raise SystemExit("leased crash window advanced execution")
    if leased["checkpoints"]:
        raise SystemExit("leased crash window unexpectedly checkpointed")
    if queued["run"]["status"] != "queued" or queued["job"]["status"] != "queued":
        raise SystemExit("queued fixture state is invalid")
    if queued["job"]["attempt"] != 0:
        raise SystemExit("queued fixture attempt is not zero")
elif phase in {"post-storage", "waiting-before-decision"}:
    if reference is None:
        raise SystemExit("reference snapshot is required")
    prior = reference["fixtures"]
    baseline = fixtures["baseline"]
    waiting = fixtures["waiting"]
    if (
        baseline["run"]["status"] != "completed"
        or baseline["job"]["status"] != "done"
        or one("baseline", "approval_requests")["status"] != "consumed"
        or baseline["decision_count"] != 1
        or baseline["approve_decision_count"] != 1
        or one("baseline", "actions")["status"] != "succeeded"
        or baseline["mock_submissions"] != prior["baseline"]["mock_submissions"]
        or not mock_matches_action(baseline)
        or not exactly_one_succeeded_irreversible(baseline)
    ):
        raise SystemExit("baseline durable state changed")
    stable_names = (
        ("baseline", "waiting", "leased", "queued")
        if phase == "post-storage"
        else ("baseline", "waiting")
    )
    for name in stable_names:
        if fixtures[name]["events"] != prior[name]["events"]:
            raise SystemExit(f"{name} event prefix changed")
        if fixtures[name]["checkpoints"] != prior[name]["checkpoints"]:
            raise SystemExit(f"{name} checkpoint identities changed")
        if fixtures[name]["llm_invocations"] != prior[name]["llm_invocations"]:
            raise SystemExit(f"{name} LLM invocation identities changed")
        if fixtures[name]["tool_invocations"] != prior[name]["tool_invocations"]:
            raise SystemExit(f"{name} tool invocation identities changed")
    if waiting["llm_invocations"] != prior["waiting"]["llm_invocations"]:
        raise SystemExit("waiting LLM invocations changed before decision")
    if waiting["tool_invocations"] != prior["waiting"]["tool_invocations"]:
        raise SystemExit("waiting tool invocations changed before decision")
    if (
        waiting["run"]["status"] != "waiting_approval"
        or waiting["job"]["status"] != "done"
        or one("waiting", "approval_requests")["status"] != "pending"
        or waiting["decision_count"] != 0
        or waiting["approve_decision_count"] != 0
        or one("waiting", "actions")["status"] != "proposed"
        or waiting["mock_submissions"]
    ):
        raise SystemExit("waiting state changed before decision")
    if phase == "post-storage":
        leased = fixtures["leased"]
        queued = fixtures["queued"]
        if leased["job"]["status"] != "leased" or leased["job"]["attempt"] != 1:
            raise SystemExit("leased state was reset before worker recovery")
        if not leased["job"]["owner_present"] or leased["job"]["leased_by"] != leased_by:
            raise SystemExit("leased identity changed during outage")
        expiry = dt.datetime.fromisoformat(leased["job"]["lease_expires_at"].replace("Z", "+00:00"))
        if expiry > dt.datetime.now(dt.timezone.utc):
            raise SystemExit("leased fixture is not expired at storage barrier")
        if queued["job"]["status"] != "queued" or queued["job"]["attempt"] != 0:
            raise SystemExit("queued fixture changed during outage")
elif phase == "final":
    if reference is None:
        raise SystemExit("reference snapshot is required")
    prior = reference["fixtures"]
    baseline, waiting, leased, queued = (fixtures[name] for name in ("baseline", "waiting", "leased", "queued"))
    if (
        baseline["run"]["status"] != "completed"
        or baseline["job"]["status"] != "done"
        or one("baseline", "approval_requests")["status"] != "consumed"
        or baseline["decision_count"] != 1
        or baseline["approve_decision_count"] != 1
        or one("baseline", "actions")["status"] != "succeeded"
        or baseline["mock_submissions"] != prior["baseline"]["mock_submissions"]
        or baseline["llm_invocations"] != prior["baseline"]["llm_invocations"]
        or baseline["tool_invocations"] != prior["baseline"]["tool_invocations"]
        or not mock_matches_action(baseline)
        or not exactly_one_succeeded_irreversible(baseline)
    ):
        raise SystemExit("baseline completed action or Mock state changed")
    if queued["run"]["status"] != "completed" or queued["job"]["status"] != "done" or queued["job"]["attempt"] != 1:
        raise SystemExit("queued recovery did not converge")
    if leased["run"]["status"] != "completed" or leased["job"]["status"] != "done" or leased["job"]["attempt"] != 2:
        raise SystemExit("leased recovery did not converge at attempt 2")
    if sum(event["type"] == "job.lease_expired" for event in leased["events"]) != 1:
        raise SystemExit("leased recovery evidence is not exactly one expiry")
    if sum(event["type"] == "run.completed" for event in leased["events"]) != 1:
        raise SystemExit("leased recovery has duplicate or missing terminal evidence")
    if sum(event["type"] == "run.completed" for event in queued["events"]) != 1:
        raise SystemExit("queued recovery has duplicate or missing terminal evidence")
    if waiting["run"]["status"] != "completed" or waiting["job"]["status"] != "done":
        raise SystemExit("waiting resume did not complete")
    if (
        one("waiting", "approval_requests")["status"] != "consumed"
        or waiting["decision_count"] != 1
        or waiting["approve_decision_count"] != 1
    ):
        raise SystemExit("waiting approval did not consume exactly one decision")
    if (
        one("waiting", "actions")["status"] != "succeeded"
        or not mock_matches_action(waiting)
        or not exactly_one_succeeded_irreversible(waiting)
    ):
        raise SystemExit("waiting action/Mock did not converge exactly once")
    if not prefix(waiting["events"], prior["waiting"]["events"]):
        raise SystemExit("waiting event history is not append-only")
    if not set((x["checkpoint_ns"], x["checkpoint_id"]) for x in prior["waiting"]["checkpoints"]).issubset(
        set((x["checkpoint_ns"], x["checkpoint_id"]) for x in waiting["checkpoints"])
    ):
        raise SystemExit("waiting PRE checkpoints were lost")
    if waiting["llm_invocations"] != prior["waiting"]["llm_invocations"]:
        raise SystemExit("approval resume duplicated prior LLM work")
    prior_read_only = [x for x in prior["waiting"]["tool_invocations"] if x["effect"] == "read_only"]
    final_read_only = [x for x in waiting["tool_invocations"] if x["effect"] == "read_only"]
    if final_read_only != prior_read_only:
        raise SystemExit("approval resume duplicated prior read-only tool work")
else:
    raise SystemExit("unknown snapshot validation phase")
PY
}

wait_until_lease_expired() {
    local lease_value="$1" wait_seconds
    wait_seconds="$(python3 -c '
import datetime as dt, math, sys
expiry=dt.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00"))
remaining=(expiry-dt.datetime.now(dt.timezone.utc)).total_seconds()+1.0
print(max(0, math.ceil(remaining)))
' "$lease_value")"
    (( wait_seconds <= PF_GATE86_DEADLINE_SECONDS )) \
        || fail "lease expiry exceeds rehearsal deadline"
    if (( wait_seconds > 0 )); then
        sleep "$wait_seconds"
    fi
}

assert_volumes_retained() {
    local volume
    while IFS= read -r volume; do
        [[ -n "$volume" ]] || continue
        docker volume inspect "$volume" >/dev/null \
            || fail "required durable volume is missing after outage: $volume"
    done <<<"$PROJECT_VOLUMES"
}

sse_verify() {
    local workspace_id="$1" run_id="$2"
    compose exec -T api python -c '
import json, sys, urllib.request
workspace_id, run_id = sys.argv[1:]
url=f"http://127.0.0.1:8000/api/v1/workspaces/{workspace_id}/runs/{run_id}/events"
def ids(last=None):
    headers={"Accept":"text/event-stream"}
    if last is not None: headers["Last-Event-ID"]=str(last)
    request=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(request,timeout=15) as response:
        return [int(line[3:].strip()) for line in response.read().decode().splitlines() if line.startswith("id:")]
first=ids()
if not first: raise SystemExit("terminal SSE returned no event IDs")
cursor=first[-1]
replayed=ids(cursor)
if any(value <= cursor for value in replayed):
    raise SystemExit("SSE replay duplicated acknowledged event")
print(json.dumps({"last_event_id":cursor,"replayed_ids":replayed},separators=(",",":")))
' "$workspace_id" "$run_id"
}

write_manifest() {
    local metrics_pre_path="$1" metrics_final_path="$2" final_path="$3" sse_json="$4"
    local manifest_path="$work_directory/manifest.json"
    [[ ! -e "$manifest_path" ]] || fail "manifest already exists"
    python3 - "$manifest_path" "$metrics_pre_path" "$metrics_final_path" "$final_path" "$sse_json" <<'PY'
import json, os, sys
manifest_path, metrics_pre_path, metrics_final_path, final_path, sse_json = sys.argv[1:]
final=json.load(open(final_path, encoding="utf-8"))
fixtures={name:{
    "run_id": value["run"]["id"],
    "run_status": value["run"]["status"],
    "job_status": value["job"]["status"],
    "job_attempt": value["job"]["attempt"],
    "next_event_seq": value["run"]["next_event_seq"],
    "event_count": len(value["events"]),
    "approval_request_ids": [item["id"] for item in value["approval_requests"]],
    "approval_statuses": [item["status"] for item in value["approval_requests"]],
    "decision_count": value["decision_count"],
    "approve_decision_count": value["approve_decision_count"],
    "action_ids": [item["id"] for item in value["actions"]],
    "action_statuses": [item["status"] for item in value["actions"]],
    "mock_count": len(value["mock_submissions"]),
    "mock_idempotency_keys": [
        item["idempotency_key"] for item in value["mock_submissions"]
    ],
    "checkpoint_count": len(value["checkpoints"]),
} for name,value in final["fixtures"].items()}
manifest={
    "result":"PASS",
    "scope":{
        "recovery_path":"project_scoped_full_stack",
        "application_compose_recovery":"VERIFIED",
        "whole_host_recovery":"NOT_VERIFIED",
    },
    "explicit_not_verified":[
        "linux_os_boot","docker_daemon_restart","boot_time_container_start",
        "host_networking_firewall","host_mount_initialization",
        "systemd_service_order","vm_vps_lifecycle","host_crash_power_loss",
    ],
    "project":{
        "compose_project_name":os.environ["COMPOSE_PROJECT_NAME"],
        "application_image_ref":os.environ["APPLICATION_IMAGE_REF"],
        "application_image_id":os.environ["APPLICATION_IMAGE_ID"],
        "postgres_image":os.environ["POSTGRES_IMAGE_REF"],
        "postgres_version":os.environ["POSTGRES_VERSION"],
        "migration_revision":os.environ["MIGRATION_REVISION"],
        "caddy_binding_pre":json.loads(os.environ["PRE_CADDY_BINDING"]),
        "caddy_binding_post":json.loads(os.environ["POST_CADDY_BINDING"]),
        "retained_volumes":os.environ["PROJECT_VOLUMES"].splitlines(),
    },
    "fixtures":fixtures,
    "recovery":{
        "lease_expiry_reclaim":True,"queue_recovery":True,
        "checkpoint_preservation_resume":True,"event_continuity":True,
        "mock_exact_counts":True,"sse":json.loads(sse_json),
    },
    "metrics":{
        "pre":json.load(open(metrics_pre_path,encoding="utf-8")),
        "final":json.load(open(metrics_final_path,encoding="utf-8")),
    },
    "durations":{
        "outage_seconds":int(os.environ["OUTAGE_SECONDS"]),
        "recovery_seconds":int(os.environ["RECOVERY_SECONDS"]),
        "interpretation":"measured observation; not an RTO SLA",
    },
}
with open(manifest_path,"x",encoding="utf-8") as handle:
    json.dump(manifest,handle,sort_keys=True,separators=(",",":"))
    handle.write("\n")
os.chmod(manifest_path,0o600)
PY
}

run_shared_host() {
    [[ "${PF_GATE86_SHARED_HOST_OUTAGE_APPROVED:-}" == "1" ]] \
        || fail "PF_GATE86_SHARED_HOST_OUTAGE_APPROVED=1 is required"
    validate_evidence_directory
    acquire_deployment_lock
    deployment_preflight

    stage="create-private-evidence-directory"
    local utc random_token
    utc="$(date -u +%Y%m%dT%H%M%SZ)"
    random_token="$(python3 -c 'import secrets; print(secrets.token_hex(6))')"
    MARKER="gate86-$random_token"
    LEASED_BY="$MARKER-crash-window"
    work_directory="$PF_GATE86_EVIDENCE_DIR/$MARKER-$utc"
    [[ ! -e "$work_directory" ]] || fail "evidence work directory already exists"
    mkdir --mode 0700 -- "$work_directory"
    export MARKER LEASED_BY

    stage="metrics-pre"
    write_new_file "$work_directory/metrics-pre.json" "$(runtime_metrics_json)"

    stage="fixture-identity-document"
    local me_json workspace_actor document_id action_id claim_json claimed_run
    me_json="$(api_request GET /api/v1/me)"
    read -r ACTOR_ID WORKSPACE_ID <<<"$(python3 -c '
import json,sys
value=json.loads(sys.argv[1]); matches=[w for w in value["workspaces"] if w["kind"]=="personal" and w["role"]=="admin"]
if len(matches)!=1: raise SystemExit("expected exactly one personal/admin workspace")
print(value["user_id"], matches[0]["workspace_id"])
' "$me_json")"
    require_uuid ACTOR_ID "$ACTOR_ID"
    require_uuid WORKSPACE_ID "$WORKSPACE_ID"
    document_id="$(create_document "$WORKSPACE_ID" "$ACTOR_ID" "$MARKER")"
    require_uuid document_id "$document_id"

    stage="fixture-baseline"
    BASELINE_RUN_ID="$(create_run "$WORKSPACE_ID" application "$MARKER baseline" "$document_id")"
    require_uuid BASELINE_RUN_ID "$BASELINE_RUN_ID"
    poll_run_status "$WORKSPACE_ID" "$BASELINE_RUN_ID" waiting_approval >/dev/null
    action_id="$(action_id_for_run "$BASELINE_RUN_ID")"
    require_uuid baseline_action_id "$action_id"
    approve_action "$WORKSPACE_ID" "$action_id" "$MARKER-baseline-approve"
    poll_run_status "$WORKSPACE_ID" "$BASELINE_RUN_ID" completed >/dev/null

    stage="fixture-waiting"
    WAITING_RUN_ID="$(create_run "$WORKSPACE_ID" application "$MARKER waiting" "$document_id")"
    require_uuid WAITING_RUN_ID "$WAITING_RUN_ID"
    poll_run_status "$WORKSPACE_ID" "$WAITING_RUN_ID" waiting_approval >/dev/null

    stage="fixture-stop-worker-for-crash-window"
    compose stop --timeout 30 worker
    wait_worker_stopped

    stage="fixture-leased-crash-window"
    LEASED_RUN_ID="$(create_run "$WORKSPACE_ID" research "$MARKER leased")"
    require_uuid LEASED_RUN_ID "$LEASED_RUN_ID"
    claim_json="$(claim_crash_window "$LEASED_BY")"
    claimed_run="$(json_value "$claim_json" 'value["run_id"]')"
    [[ "$claimed_run" == "$LEASED_RUN_ID" ]] || fail "one-off worker claimed the wrong run"

    stage="fixture-queued"
    QUEUED_RUN_ID="$(create_run "$WORKSPACE_ID" research "$MARKER queued")"
    require_uuid QUEUED_RUN_ID "$QUEUED_RUN_ID"
    export ACTOR_ID WORKSPACE_ID BASELINE_RUN_ID WAITING_RUN_ID LEASED_RUN_ID QUEUED_RUN_ID

    stage="pre-recovery-snapshot"
    write_new_file "$work_directory/pre-recovery.json" "$(recovery_snapshot)"
    validate_snapshot pre "$work_directory/pre-recovery.json"

    stage="controlled-full-stack-outage"
    local outage_start recovery_start recovery_end lease_expiry
    outage_start="$(date +%s)"
    outage_started=1
    compose down --timeout 30
    [[ -z "$(compose ps --all --quiet)" ]] || fail "project containers remain after outage"
    assert_volumes_retained
    lease_expiry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixtures"]["leased"]["job"]["lease_expires_at"])' \
        "$work_directory/pre-recovery.json")"
    wait_until_lease_expired "$lease_expiry"

    stage="recovery-postgres"
    recovery_start="$(date +%s)"
    compose up --detach --wait postgres
    require_recovered_service postgres "$POSTGRES_IMAGE_REF"
    assert_volumes_retained

    stage="post-storage-snapshot"
    write_new_file "$work_directory/post-storage.json" "$(recovery_snapshot)"
    validate_snapshot post-storage "$work_directory/post-storage.json" \
        "$work_directory/pre-recovery.json"

    stage="recovery-api"
    compose up --detach --wait --no-deps api
    require_recovered_service api "$APPLICATION_IMAGE_REF" "$APPLICATION_IMAGE_ID"
    compose exec -T api python -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/readyz",timeout=2).read()'

    stage="recovery-caddy"
    compose up --detach --wait --no-deps caddy
    POST_CADDY_BINDING="$(actual_caddy_binding "$(one_container_id caddy)")"
    export POST_CADDY_BINDING
    [[ "$POST_CADDY_BINDING" == "$PRE_CADDY_BINDING" ]] \
        || fail "Caddy binding drifted during reconstruction"
    validate_safe_binding "$POST_CADDY_BINDING"

    stage="recovery-worker"
    compose up --detach --wait --no-deps worker
    worker_started=1
    require_recovered_service worker "$APPLICATION_IMAGE_REF" "$APPLICATION_IMAGE_ID"

    stage="waiting-before-decision"
    write_new_file "$work_directory/waiting-before-decision.json" "$(recovery_snapshot)"
    validate_snapshot waiting-before-decision \
        "$work_directory/waiting-before-decision.json" "$work_directory/pre-recovery.json"

    stage="queue-and-lease-recovery"
    poll_run_status "$WORKSPACE_ID" "$QUEUED_RUN_ID" completed >/dev/null
    poll_run_status "$WORKSPACE_ID" "$LEASED_RUN_ID" completed >/dev/null

    stage="approval-resume"
    action_id="$(action_id_for_run "$WAITING_RUN_ID")"
    require_uuid waiting_action_id "$action_id"
    approve_action "$WORKSPACE_ID" "$action_id" "$MARKER-waiting-approve"
    poll_run_status "$WORKSPACE_ID" "$WAITING_RUN_ID" completed >/dev/null

    stage="final-snapshot"
    write_new_file "$work_directory/final.json" "$(recovery_snapshot)"
    validate_snapshot final "$work_directory/final.json" "$work_directory/pre-recovery.json"

    stage="terminal-sse"
    local sse_json
    sse_json="$(sse_verify "$WORKSPACE_ID" "$WAITING_RUN_ID")"

    stage="metrics-final"
    write_new_file "$work_directory/metrics-final.json" "$(runtime_metrics_json)"
    recovery_end="$(date +%s)"
    OUTAGE_SECONDS=$(( recovery_start - outage_start ))
    RECOVERY_SECONDS=$(( recovery_end - recovery_start ))
    export OUTAGE_SECONDS RECOVERY_SECONDS
    write_manifest "$work_directory/metrics-pre.json" "$work_directory/metrics-final.json" \
        "$work_directory/final.json" "$sse_json"
    worker_started=0
    stage="complete"
    log "PASS evidence=$work_directory/manifest.json"
    log "application/Compose recovery=VERIFIED"
    log "whole-host recovery=NOT VERIFIED"
}

initialize_common
if [[ "$mode" == "metrics" ]]; then
    run_metrics_mode
else
    run_shared_host
fi
