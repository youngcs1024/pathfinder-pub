#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ACCEPTANCE_SQL="$REPOSITORY_ROOT/scripts/gate87_live_acceptance.sql"

stage="initialization"
mode="${1:-}"
evidence_directory=""
declare -a COMPOSE_COMMAND=()

log() {
    printf 'gate87: %s\n' "$*"
}

fail() {
    printf 'gate87: error: stage=%s: %s\n' "$stage" "$1" >&2
    printf '%s\n' 'gate87: no business mutation or automatic retry was attempted' >&2
    exit 1
}

unexpected_error() {
    local exit_code="$1"
    printf 'gate87: unexpected read-only command failure stage=%s exit_code=%s\n' \
        "$stage" "$exit_code" >&2
    exit "$exit_code"
}

interrupted() {
    printf 'gate87: interrupted during stage=%s; no later stage was started\n' "$stage" >&2
    exit 130
}

trap 'unexpected_error $?' ERR
trap interrupted INT TERM HUP

require_nonempty() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "$name is required"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
}

require_uuid() {
    local name="$1" value="$2"
    python3 -c '
import sys, uuid
try:
    value = uuid.UUID(sys.argv[1])
except (AttributeError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if str(value) == sys.argv[1].lower() else 1)
' "$value" || fail "$name must be a canonical UUID"
}

validate_cost_cap() {
    require_nonempty PF_GATE87_COST_CAP_CNY
    python3 -c '
from decimal import Decimal, InvalidOperation
import sys
try:
    value = Decimal(sys.argv[1])
except InvalidOperation:
    raise SystemExit(1)
raise SystemExit(0 if value.is_finite() and value > 0 else 1)
' "$PF_GATE87_COST_CAP_CNY" || fail "PF_GATE87_COST_CAP_CNY must be a positive decimal"
}

validate_compose_files() {
    require_nonempty PF_GATE87_COMPOSE_FILES
    local item canonical repository has_external=0
    IFS=':' read -r -a compose_files <<<"$PF_GATE87_COMPOSE_FILES"
    [[ "${#compose_files[@]}" -ge 2 ]] \
        || fail "PF_GATE87_COMPOSE_FILES must include base and private overlay files"
    repository="$(realpath -e -- "$REPOSITORY_ROOT")"
    COMPOSE_COMMAND=(docker compose --project-name "$COMPOSE_PROJECT_NAME")
    for item in "${compose_files[@]}"; do
        [[ -n "$item" ]] || fail "PF_GATE87_COMPOSE_FILES contains an empty entry"
        canonical="$(realpath -e -- "$item")" || fail "a configured Compose file is missing"
        [[ -f "$canonical" ]] || fail "every configured Compose path must be a file"
        if [[ "$canonical" != "$repository" && "$canonical" != "$repository/"* ]]; then
            has_external=1
        fi
        COMPOSE_COMMAND+=(--file "$canonical")
    done
    [[ "$has_external" -eq 1 ]] \
        || fail "PF_GATE87_COMPOSE_FILES must include a repository-external private overlay"
}

validate_evidence_directory() {
    require_nonempty PF_GATE87_EVIDENCE_DIR
    local lexical canonical repository owner mode_bits
    lexical="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' \
        "$PF_GATE87_EVIDENCE_DIR")"
    canonical="$(realpath -e -- "$PF_GATE87_EVIDENCE_DIR")" \
        || fail "PF_GATE87_EVIDENCE_DIR must already exist"
    repository="$(realpath -e -- "$REPOSITORY_ROOT")"
    [[ -d "$canonical" ]] || fail "PF_GATE87_EVIDENCE_DIR must be a directory"
    if [[ "$lexical" == "$repository" || "$lexical" == "$repository/"* \
        || "$canonical" == "$repository" || "$canonical" == "$repository/"* ]]; then
        fail "PF_GATE87_EVIDENCE_DIR must be outside the repository"
    fi
    owner="$(stat -c '%u' -- "$canonical")"
    [[ "$owner" == "$(id -u)" ]] || fail "PF_GATE87_EVIDENCE_DIR must be current-user owned"
    mode_bits=$((8#$(stat -c '%a' -- "$canonical")))
    (( (mode_bits & 077) == 0 )) \
        || fail "PF_GATE87_EVIDENCE_DIR must not grant group or other permissions"
    evidence_directory="$canonical"
}

compose() {
    "${COMPOSE_COMMAND[@]}" "$@"
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

effective_metadata() {
    compose config --format json | python3 -c '
import json, sys
config = json.load(sys.stdin)
services = config.get("services", {})
if set(services) != {"postgres", "api", "worker", "caddy"}:
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
print(json.dumps({
    "bindings": sorted(ports, key=lambda value: (value["target"], value["published"])),
    "api_image": services["api"].get("image"),
    "worker_image": services["worker"].get("image"),
}, sort_keys=True, separators=(",", ":")))
'
}

postgres_scalar() {
    local query="$1"
    compose exec -T postgres sh -c \
        'psql -X -qAt --set ON_ERROR_STOP=on --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --command "$1"' \
        gate87-psql "$query"
}

postgres_acceptance() {
    compose exec -T postgres sh -c \
        'psql -X -qAt --set ON_ERROR_STOP=on --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" "$@"' \
        gate87-psql \
        --set "actor_user_id=$PF_GATE87_ACTOR_USER_ID" \
        --set "workspace_id=$PF_GATE87_WORKSPACE_ID" \
        --set "document_id=$PF_GATE87_DOCUMENT_ID" \
        --set "run_id=$PF_GATE87_RUN_ID" \
        --set "cost_cap_cny=$PF_GATE87_COST_CAP_CNY" <"$ACCEPTANCE_SQL"
}

validate_binding() {
    python3 -c '
import json, sys
bindings = json.loads(sys.argv[1])
if not bindings or {item["target"] for item in bindings} - {80, 443}:
    raise SystemExit(1)
if any(item["host_ip"] not in {"127.0.0.1", "::1"} for item in bindings):
    raise SystemExit(1)
' "$1" || fail "Caddy binding is not the approved loopback-only shared-host binding"
}

run_preflight() {
    stage="runtime-preflight"
    compose config --quiet >/dev/null
    local metadata service id state health image_ref image_id
    local api_ref api_id worker_ref worker_id caddy_id current_binding planned_binding
    metadata="$(effective_metadata)"
    for service in postgres api worker caddy; do
        id="$(one_container_id "$service")"
        IFS='|' read -r state health image_ref image_id <<<"$(container_state "$id")"
        [[ "$state" == "running" ]] || fail "$service must be running"
        if [[ "$service" != "caddy" && "$health" != "healthy" ]]; then
            fail "$service must be healthy"
        fi
        case "$service" in
            api) api_ref="$image_ref"; api_id="$image_id" ;;
            worker) worker_ref="$image_ref"; worker_id="$image_id" ;;
            caddy) caddy_id="$id" ;;
        esac
    done
    [[ "$api_ref" == "$worker_ref" && "$api_id" == "$worker_id" ]] \
        || fail "API and worker application image identity differs"
    python3 -c '
import json, sys
value = json.loads(sys.argv[1])
raise SystemExit(0 if value["api_image"] == sys.argv[2]
                 and value["worker_image"] == sys.argv[2] else 1)
' "$metadata" "$api_ref" || fail "running application image differs from effective Compose config"

    current_binding="$(actual_caddy_binding "$caddy_id")"
    planned_binding="$(python3 -c \
        'import json,sys; print(json.dumps(json.loads(sys.argv[1])["bindings"],sort_keys=True,separators=(",",":")))' \
        "$metadata")"
    [[ "$current_binding" == "$planned_binding" ]] \
        || fail "effective/current Caddy binding mismatch"
    validate_binding "$current_binding"

    local worker_modes api_mode current_revision candidate_head unknown_count
    worker_modes="$(compose exec -T worker python -c \
        'from app.config import Settings; import json; s=Settings(); print(json.dumps([s.llm_mode,s.search_mode,s.trace_mode,s.langfuse_sample_rate],separators=(",",":")))')"
    [[ "$worker_modes" == '["qwen","tavily","langfuse",1.0]' ]] \
        || fail "worker must use qwen/tavily/langfuse with sample rate 1.0"
    api_mode="$(compose exec -T api python -c \
        'from app.config import Settings; print(Settings().auth_mode)')"
    [[ "$api_mode" == "supabase" ]] || fail "API auth mode must be supabase"
    current_revision="$(postgres_scalar 'SELECT version_num FROM alembic_version')"
    candidate_head="$(compose exec -T api alembic heads | awk 'NF {print $1; exit}')"
    [[ -n "$current_revision" && "$current_revision" == "$candidate_head" ]] \
        || fail "database revision differs from candidate head"
    unknown_count="$(postgres_scalar \
        "SELECT (SELECT count(*) FROM action_intents WHERE status='outcome_unknown') + (SELECT count(*) FROM tool_invocations WHERE status='outcome_unknown')")"
    [[ "$unknown_count" == "0" ]] || fail "current outcome_unknown facts require manual reconciliation"

    PREFLIGHT_JSON="$(python3 -c '
import json, sys
print(json.dumps({
    "version": 1,
    "project": sys.argv[1],
    "application_image_ref": sys.argv[2],
    "application_image_id": sys.argv[3],
    "caddy_bindings": json.loads(sys.argv[4]),
    "database_revision": sys.argv[5],
    "runtime": {"auth": "supabase", "llm": "qwen", "search": "tavily",
                "trace": "langfuse", "langfuse_sample_rate": 1.0},
    "cost_cap_cny": sys.argv[6],
    "outcome_unknown_count": 0,
    "accepted": True,
}, sort_keys=True, separators=(",", ":")))
' "$COMPOSE_PROJECT_NAME" "$api_ref" "$api_id" "$current_binding" \
        "$current_revision" "$PF_GATE87_COST_CAP_CNY")"
}

initialize() {
    stage="static-preflight"
    case "$mode" in
        preflight|verify-live) ;;
        *) fail "usage: scripts/gate87_acceptance.sh preflight|verify-live" ;;
    esac
    require_nonempty COMPOSE_PROJECT_NAME
    [[ "$COMPOSE_PROJECT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] \
        || fail "COMPOSE_PROJECT_NAME contains unsafe characters"
    for command_name in awk docker id python3 realpath stat; do
        require_command "$command_name"
    done
    [[ -f "$ACCEPTANCE_SQL" ]] || fail "Gate 8.7 acceptance SQL is missing"
    validate_cost_cap
    validate_compose_files
    validate_evidence_directory
}

initialize
run_preflight
if [[ "$mode" == "preflight" ]]; then
    [[ ! -e "$evidence_directory/gate87-preflight.json" ]] \
        || fail "Gate 8.7 preflight evidence already exists"
    printf '%s\n' "$PREFLIGHT_JSON" | tee "$evidence_directory/gate87-preflight.json"
    log "preflight PASS"
    exit 0
fi

stage="live-identifiers"
for name in PF_GATE87_ACTOR_USER_ID PF_GATE87_WORKSPACE_ID PF_GATE87_DOCUMENT_ID PF_GATE87_RUN_ID; do
    require_nonempty "$name"
    require_uuid "$name" "${!name}"
done

stage="live-database-verification"
[[ ! -e "$evidence_directory/gate87-live-$PF_GATE87_RUN_ID.json" ]] \
    || fail "Gate 8.7 live evidence already exists"
live_json="$(postgres_acceptance)" \
    || fail "live acceptance facts did not satisfy the fail-closed SQL contract"
python3 -c 'import json,sys; value=json.loads(sys.argv[1]); raise SystemExit(0 if value.get("accepted") is True else 1)' \
    "$live_json" || fail "live acceptance output is invalid or rejected"
printf '%s\n' "$live_json" | tee "$evidence_directory/gate87-live-$PF_GATE87_RUN_ID.json"
log "verify-live PASS"
