#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly CONSISTENCY_SQL="$REPOSITORY_ROOT/scripts/gate85_consistency.sql"
readonly LOCK_ROOT="${TMPDIR:-/tmp}"

stage="initialization"
lock_directory=""
lock_acquired=0
work_directory=""
remote_staging_directory="none"
remote_dump_path="none"
backup_path="none"
manifest_path="none"
candidate_image_id="unknown"
candidate_head="unknown"
source_revision="unknown"
source_postgres_image="unknown"
source_postgres_version="unknown"
worktree_head="unknown"
worktree_fingerprint="unknown"
fixture_document_id="unknown"
restored_e2e_run_id="unknown"
restore_revision="unknown"

log() {
    printf 'gate85: %s\n' "$*"
}

diagnose() {
    printf 'gate85: stage=%s\n' "$stage" >&2
    printf 'gate85: source_project=%s restore_project=%s\n' \
        "${PF_GATE85_SOURCE_PROJECT:-missing}" \
        "${PF_GATE85_RESTORE_PROJECT:-missing}" >&2
    printf 'gate85: source_revision=%s candidate_head=%s restore_revision=%s\n' \
        "$source_revision" "$candidate_head" "$restore_revision" >&2
    printf 'gate85: remote_staging_directory=%s remote_dump_path=%s\n' \
        "$remote_staging_directory" "$remote_dump_path" >&2
    printf 'gate85: backup_path=%s manifest_path=%s\n' \
        "$backup_path" "$manifest_path" >&2
    printf 'gate85: work_directory=%s\n' "${work_directory:-not-created}" >&2
    printf '%s\n' \
        'gate85: restore resources are retained; no automatic destructive cleanup was performed' >&2
}

fail() {
    printf 'gate85: error: %s\n' "$1" >&2
    diagnose
    exit 1
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
    printf 'gate85: interrupted by %s during stage=%s; subsequent stages were not started\n' \
        "$signal_name" "$stage" >&2
    diagnose
    exit 130
}

unexpected_error() {
    local exit_code="$1"
    printf 'gate85: unexpected command failure exit_code=%s\n' "$exit_code" >&2
    diagnose
    exit "$exit_code"
}

trap cleanup_lock EXIT
trap 'unexpected_error $?' ERR
trap 'interrupted SIGINT' INT
trap 'interrupted SIGTERM' TERM
trap 'interrupted SIGHUP' HUP

require_nonempty() {
    local variable_name="$1"
    if [[ -z "${!variable_name:-}" ]]; then
        fail "$variable_name is required"
    fi
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        fail "$command_name is required"
    fi
}

validate_project_name() {
    local variable_name="$1"
    local value="${!variable_name}"
    if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
        fail "$variable_name must contain only letters, digits, dot, underscore, or hyphen"
    fi
}

validate_uuid() {
    local variable_name="$1"
    local value="${!variable_name}"
    if [[ ! "$value" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
        fail "$variable_name must be a lowercase UUID"
    fi
}

validate_image_tag() {
    if [[ ! "$PATHFINDER_IMAGE" =~ ^[a-z0-9][a-z0-9._/-]{0,200}:[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
        fail "PATHFINDER_IMAGE must be an explicit local image tag"
    fi
    if [[ "$PATHFINDER_IMAGE" == *:latest ]]; then
        fail "PATHFINDER_IMAGE must not use the mutable latest tag"
    fi
}

validate_deadline() {
    PF_GATE85_DEADLINE_SECONDS="${PF_GATE85_DEADLINE_SECONDS:-180}"
    if [[ ! "$PF_GATE85_DEADLINE_SECONDS" =~ ^[0-9]+$ \
        || "$PF_GATE85_DEADLINE_SECONDS" -lt 30 \
        || "$PF_GATE85_DEADLINE_SECONDS" -gt 900 ]]; then
        fail "PF_GATE85_DEADLINE_SECONDS must be an integer between 30 and 900"
    fi
    export PF_GATE85_DEADLINE_SECONDS
}

validate_backup_directory() {
    local lexical_backup canonical_backup canonical_repository mode permission owner
    lexical_backup="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' \
        "$PF_GATE85_BACKUP_DIR")"
    if ! canonical_backup="$(realpath -e -- "$PF_GATE85_BACKUP_DIR")"; then
        fail "PF_GATE85_BACKUP_DIR must already exist and be resolvable"
    fi
    canonical_repository="$(realpath -e -- "$REPOSITORY_ROOT")"
    if [[ ! -d "$canonical_backup" ]]; then
        fail "PF_GATE85_BACKUP_DIR must be a directory"
    fi
    if [[ "$lexical_backup" == "$canonical_repository" \
        || "$lexical_backup" == "$canonical_repository/"* \
        || "$canonical_backup" == "$canonical_repository" \
        || "$canonical_backup" == "$canonical_repository/"* ]]; then
        fail "PF_GATE85_BACKUP_DIR must be outside the repository and not escape through a repository symlink"
    fi
    owner="$(stat -c '%u' -- "$canonical_backup")"
    if [[ "$owner" != "$(id -u)" ]]; then
        fail "PF_GATE85_BACKUP_DIR must be owned by the current user"
    fi
    mode="$(stat -c '%a' -- "$canonical_backup")"
    permission=$((8#$mode))
    if (( (permission & 077) != 0 )); then
        fail "PF_GATE85_BACKUP_DIR must not grant group or other permissions"
    fi
    PF_GATE85_BACKUP_DIR="$canonical_backup"
    export PF_GATE85_BACKUP_DIR
}

static_preflight() {
    stage="static-preflight"
    for variable_name in \
        PF_GATE85_SOURCE_SSH \
        PF_GATE85_SOURCE_PROJECT \
        PF_GATE85_BACKUP_DIR \
        PF_GATE85_RESTORE_PROJECT \
        PF_GATE85_FIXTURE_RUN_ID \
        PATHFINDER_IMAGE; do
        require_nonempty "$variable_name"
    done
    for command_name in docker git id python3 realpath scp sha256sum ssh stat; do
        require_command "$command_name"
    done
    if [[ ! -f "$CONSISTENCY_SQL" ]]; then
        fail "consistency SQL is missing"
    fi
    if [[ ! "$PF_GATE85_SOURCE_SSH" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
        fail "PF_GATE85_SOURCE_SSH contains unsafe characters"
    fi
    validate_project_name PF_GATE85_SOURCE_PROJECT
    validate_project_name PF_GATE85_RESTORE_PROJECT
    validate_uuid PF_GATE85_FIXTURE_RUN_ID
    validate_image_tag
    validate_deadline
    validate_backup_directory
    if [[ "$PF_GATE85_SOURCE_PROJECT" == "$PF_GATE85_RESTORE_PROJECT" ]]; then
        fail "source and restore Compose projects must be different"
    fi
    worktree_head="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"
    if [[ ! "$worktree_head" =~ ^[0-9a-f]{40}$ ]]; then
        fail "current Git HEAD is invalid"
    fi
    work_directory="$(mktemp -d "${TMPDIR:-/tmp}/pathfinder-gate85.XXXXXXXX")"
    log "work_directory=$work_directory (retained for diagnostics)"
}

acquire_lock() {
    stage="rehearsal-lock"
    lock_directory="${LOCK_ROOT%/}/pathfinder-gate85-${PF_GATE85_RESTORE_PROJECT}.lock"
    if ! mkdir -- "$lock_directory" 2>/dev/null; then
        fail "another Gate 8.5 rehearsal holds lock $lock_directory"
    fi
    lock_acquired=1
    printf 'pid=%s\nsource_project=%s\nrestore_project=%s\nstarted_utc=%s\n' \
        "$$" "$PF_GATE85_SOURCE_PROJECT" "$PF_GATE85_RESTORE_PROJECT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$lock_directory/owner"
}

line_count() {
    awk 'NF {count += 1} END {print count + 0}'
}

validate_container_id() {
    local container_id="$1"
    local service="$2"
    if [[ ! "$container_id" =~ ^[0-9a-f]{12,64}$ ]]; then
        fail "remote $service returned an invalid container ID"
    fi
}

remote_deployment_preflight() {
    stage="remote-deployment-preflight"
    local output line_kind service container_id status health value
    if ! output="$(ssh -- "$PF_GATE85_SOURCE_SSH" sh -s -- \
        gate85-remote-preflight "$PF_GATE85_SOURCE_PROJECT" <<'REMOTE_PREFLIGHT'
set -eu
action="$1"
project="$2"
if [ "$action" != "gate85-remote-preflight" ]; then
    exit 91
fi
case "$project" in
    ''|*[!A-Za-z0-9_.-]*|[!A-Za-z0-9]*) exit 92 ;;
esac
for service in postgres api worker caddy; do
    ids="$(docker ps --all --quiet \
        --filter "label=com.docker.compose.project=$project" \
        --filter "label=com.docker.compose.service=$service")"
    count="$(printf '%s\n' "$ids" | awk 'NF {count += 1} END {print count + 0}')"
    if [ "$count" -ne 1 ]; then
        printf 'cardinality\t%s\t%s\n' "$service" "$count"
        exit 93
    fi
    container_id="$(printf '%s\n' "$ids" | awk 'NF {print; exit}')"
    status="$(docker inspect --format '{{.State.Status}}' "$container_id")"
    health="$(docker inspect --format \
        '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container_id")"
    printf 'service\t%s\t%s\t%s\t%s\n' \
        "$service" "$container_id" "$status" "$health"
done
postgres_id="$(docker ps --all --quiet \
    --filter "label=com.docker.compose.project=$project" \
    --filter 'label=com.docker.compose.service=postgres')"
postgres_image="$(docker inspect --format '{{.Config.Image}}@{{.Image}}' "$postgres_id")"
postgres_version="$(docker exec "$postgres_id" postgres --version)"
printf 'metadata\tpostgres_image\t%s\n' "$postgres_image"
printf 'metadata\tpostgres_version\t%s\n' "$postgres_version"
REMOTE_PREFLIGHT
)"; then
        fail "remote source project preflight failed for exact project labels"
    fi
    while IFS=$'\t' read -r line_kind service container_id status health; do
        [[ -n "$line_kind" ]] || continue
        if [[ "$line_kind" == "service" ]]; then
            validate_container_id "$container_id" "$service"
            if [[ "$status" != "running" ]]; then
                fail "remote $service is not running"
            fi
            if [[ "$service" != "caddy" && "$health" != "healthy" ]]; then
                fail "remote $service is not healthy"
            fi
            printf -v "SOURCE_${service^^}_ID" '%s' "$container_id"
            export "SOURCE_${service^^}_ID"
        elif [[ "$line_kind" == "metadata" ]]; then
            value="$container_id"
            case "$service" in
                postgres_image) source_postgres_image="$value" ;;
                postgres_version) source_postgres_version="$value" ;;
                *) fail "remote preflight returned unknown metadata" ;;
            esac
        else
            fail "remote preflight returned malformed evidence"
        fi
    done <<<"$output"
    for service in POSTGRES API WORKER CADDY; do
        local source_variable="SOURCE_${service}_ID"
        if [[ -z "${!source_variable:-}" ]]; then
            fail "remote preflight omitted the $service container identity"
        fi
    done
    if [[ "$source_postgres_image" == "unknown" \
        || "$source_postgres_version" == "unknown" ]]; then
        fail "remote preflight omitted PostgreSQL image/version evidence"
    fi
    log "remote_source=healthy project=$PF_GATE85_SOURCE_PROJECT"
    log "source_postgres_image=$source_postgres_image"
    log "source_postgres_version=$source_postgres_version"
}

local_restore_absence_preflight() {
    stage="local-restore-absence-preflight"
    local containers volumes networks
    if ! containers="$(docker ps --all --quiet \
        --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT")" \
        || ! volumes="$(docker volume ls --quiet \
            --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT")" \
        || ! networks="$(docker network ls --quiet \
            --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT")"; then
        fail "cannot inspect local restore project resources"
    fi
    if [[ -n "$containers" || -n "$volumes" || -n "$networks" ]]; then
        fail "local restore project already has containers, volumes, or networks; inspect it manually"
    fi
    log "restore_project_absent=$PF_GATE85_RESTORE_PROJECT"
}

worktree_fingerprint_value() {
    {
        git -C "$REPOSITORY_ROOT" rev-parse HEAD
        git -C "$REPOSITORY_ROOT" status --porcelain=v1
        git -C "$REPOSITORY_ROOT" diff --binary HEAD
        git -C "$REPOSITORY_ROOT" ls-files --others --exclude-standard -z \
            | while IFS= read -r -d '' path; do
                printf 'untracked:%s\n' "$path"
                sha256sum -- "$REPOSITORY_ROOT/$path"
            done
    } | sha256sum | awk '{print $1}'
}

compose() {
    docker compose \
        --project-name "$PF_GATE85_RESTORE_PROJECT" \
        --file "$REPOSITORY_ROOT/compose.yaml" \
        "$@"
}

configure_restore_environment() {
    local generated_password
    generated_password="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    export PF_POSTGRES_PASSWORD="$generated_password"
    export PF_DATABASE_URL="postgresql+psycopg://pathfinder:${generated_password}@postgres:5432/pathfinder"
    export PF_PUBLIC_DOMAIN="localhost"
    export PF_AUTH_MODE="fake"
    export PF_LLM_MODE="fake"
    export PF_SEARCH_MODE="fake"
    export PF_TRACE_MODE="off"
    export DASHSCOPE_API_KEY=""
    export LANGFUSE_BASE_URL=""
    export LANGFUSE_PUBLIC_KEY=""
    export LANGFUSE_SECRET_KEY=""
    export PF_QWEN_WORKSPACE_ID=""
    export PF_SUPABASE_PROJECT_REF=""
    export PF_SUPABASE_PUBLISHABLE_KEY=""
    export TAVILY_API_KEY=""
}

read_candidate_head() {
    local output heads
    if ! output="$(compose run --rm --no-deps --entrypoint alembic api heads 2>/dev/null)"; then
        fail "candidate Alembic head query failed"
    fi
    heads="$(printf '%s\n' "$output" | awk '$2 == "(head)" {print $1}')"
    if [[ "$(printf '%s\n' "$heads" | line_count)" -ne 1 \
        || ! "$heads" =~ ^[A-Za-z0-9_]+$ ]]; then
        fail "candidate image must contain exactly one valid Alembic head"
    fi
    printf '%s\n' "$heads"
}

build_and_validate_candidate() {
    stage="candidate-image-build"
    configure_restore_environment
    worktree_fingerprint="$(worktree_fingerprint_value)"
    if [[ ! "$worktree_fingerprint" =~ ^[0-9a-f]{64}$ ]]; then
        fail "could not calculate the worktree fingerprint"
    fi
    log "worktree_head=$worktree_head"
    log "worktree_dirty=$([[ -n "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1)" ]] && printf true || printf false)"
    log "worktree_fingerprint=sha256:$worktree_fingerprint"
    if ! docker build --tag "$PATHFINDER_IMAGE" "$REPOSITORY_ROOT" >/dev/null; then
        fail "local Gate 8.5 rehearsal image build failed"
    fi
    if ! candidate_image_id="$(docker image inspect --format '{{.Id}}' "$PATHFINDER_IMAGE")" \
        || [[ ! "$candidate_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        fail "candidate rehearsal image has no valid image ID"
    fi
    if ! compose config --quiet >/dev/null; then
        fail "restore Compose configuration is invalid"
    fi
    if ! compose run --rm --no-deps --entrypoint python api -c \
        'from app.config import Settings; Settings()' >/dev/null \
        || ! compose run --rm --no-deps --entrypoint python worker -c \
        'from app.config import Settings; Settings()' >/dev/null; then
        fail "candidate fake/off settings validation failed"
    fi
    candidate_head="$(read_candidate_head)"
    log "candidate_image=$PATHFINDER_IMAGE candidate_image_id=$candidate_image_id"
    log "candidate_alembic_head=$candidate_head"
}

remote_snapshot() {
    ssh -- "$PF_GATE85_SOURCE_SSH" docker exec -i "$SOURCE_POSTGRES_ID" \
        psql -X --quiet --tuples-only --no-align --set ON_ERROR_STOP=1 \
        --username=pathfinder --dbname=pathfinder \
        --set "fixture_run_id=$PF_GATE85_FIXTURE_RUN_ID" \
        --set "fixture_document_id=$fixture_document_id" <"$CONSISTENCY_SQL"
}

target_snapshot() {
    local run_id="$1"
    local document_id="$2"
    compose exec -T postgres \
        psql -X --quiet --tuples-only --no-align --set ON_ERROR_STOP=1 \
        --username=pathfinder --dbname=pathfinder \
        --set "fixture_run_id=$run_id" \
        --set "fixture_document_id=$document_id" <"$CONSISTENCY_SQL"
}

validate_snapshot_file() {
    local snapshot_file="$1"
    if ! python3 -c '
import json, sys
path = sys.argv[1]
value = json.loads(open(path, encoding="utf-8").read())
checks = value.get("checks")
if not isinstance(checks, dict):
    raise SystemExit("snapshot has no checks object")
failed = sorted(name for name, passed in checks.items() if passed is not True)
if failed:
    raise SystemExit("failed checks: " + ",".join(failed))
' "$snapshot_file"; then
        fail "consistency snapshot failed one or more invariant checks"
    fi
}

json_file_get() {
    local snapshot_file="$1"
    local dotted_path="$2"
    python3 -c '
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
for part in sys.argv[2].split("."):
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
' "$snapshot_file" "$dotted_path"
}

json_text_get() {
    local dotted_path="$1"
    python3 -c '
import json, sys
value = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
' "$dotted_path"
}

discover_fixture_document() {
    stage="source-fixture-discovery"
    local output
    if ! output="$(ssh -- "$PF_GATE85_SOURCE_SSH" sh -s -- \
        gate85-fixture-document "$SOURCE_POSTGRES_ID" "$PF_GATE85_FIXTURE_RUN_ID" <<'REMOTE_FIXTURE_DOCUMENT'
set -eu
action="$1"
postgres_id="$2"
fixture_run_id="$3"
if [ "$action" != "gate85-fixture-document" ]; then
    exit 91
fi
case "$postgres_id" in
    *[!0-9a-f]*|'') exit 92 ;;
esac
case "$fixture_run_id" in
    ????????-????-????-????-????????????) ;;
    *) exit 93 ;;
esac
case "$fixture_run_id" in
    *[!0-9a-f-]*) exit 94 ;;
esac
docker exec "$postgres_id" psql -X --quiet --tuples-only --no-align \
    --set ON_ERROR_STOP=1 --username=pathfinder --dbname=pathfinder \
    --command "SELECT resume_document_id FROM runs WHERE id = '$fixture_run_id'::uuid"
REMOTE_FIXTURE_DOCUMENT
)"; then
        fail "cannot discover the source fixture document"
    fi
    fixture_document_id="${output//$'\r'/}"
    fixture_document_id="${fixture_document_id//$'\n'/}"
    local PF_GATE85_FIXTURE_DOCUMENT_ID="$fixture_document_id"
    validate_uuid PF_GATE85_FIXTURE_DOCUMENT_ID
    log "fixture_run_id=$PF_GATE85_FIXTURE_RUN_ID fixture_document_id=$fixture_document_id"
}

capture_source_snapshot_a() {
    stage="source-snapshot-a"
    local snapshot_file="$work_directory/source-a.json"
    if ! remote_snapshot >"$snapshot_file"; then
        fail "source consistency snapshot A failed"
    fi
    validate_snapshot_file "$snapshot_file"
    source_revision="$(json_file_get "$snapshot_file" schema.alembic_version)"
    if [[ "$source_revision" != "$candidate_head" ]]; then
        fail "source revision does not equal the candidate Alembic head"
    fi
    log "source_snapshot_a=valid revision=$source_revision"
}

remote_create_dump() {
    stage="remote-pg-dump"
    local rehearsal_id timestamp dump_name output remote_sha remote_bytes remote_mode returned_path
    rehearsal_id="$(python3 -c 'import secrets; print(secrets.token_hex(12))')"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    dump_name="${timestamp}-${PF_GATE85_SOURCE_PROJECT}-gate85.dump"
    remote_staging_directory="/tmp/pathfinder-gate85.${rehearsal_id}"
    remote_dump_path="$remote_staging_directory/$dump_name"
    if ! output="$(ssh -- "$PF_GATE85_SOURCE_SSH" sh -s -- \
        gate85-remote-dump "$remote_staging_directory" "$dump_name" "$SOURCE_POSTGRES_ID" <<'REMOTE_DUMP'
set -eu
umask 077
action="$1"
staging_directory="$2"
dump_name="$3"
postgres_id="$4"
if [ "$action" != "gate85-remote-dump" ]; then
    exit 91
fi
case "$staging_directory" in
    /tmp/pathfinder-gate85.[0-9a-f]*) ;;
    *) exit 92 ;;
esac
case "$dump_name" in
    *-gate85.dump) ;;
    *) exit 93 ;;
esac
mkdir -- "$staging_directory"
chmod 700 -- "$staging_directory"
incomplete_path="$staging_directory/$dump_name.incomplete"
final_path="$staging_directory/$dump_name"
if [ -e "$incomplete_path" ] || [ -e "$final_path" ]; then
    exit 94
fi
if ! docker exec "$postgres_id" sh -ec \
    'exec pg_dump --format=custom --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
    >"$incomplete_path"; then
    exit 95
fi
if [ ! -s "$incomplete_path" ]; then
    exit 96
fi
chmod 600 -- "$incomplete_path"
if ! docker exec -i "$postgres_id" sh -ec 'exec pg_restore --list' \
    <"$incomplete_path" >/dev/null 2>&1; then
    exit 97
fi
mv -- "$incomplete_path" "$final_path"
sha="$(sha256sum -- "$final_path" | awk '{print $1}')"
bytes="$(stat -c '%s' -- "$final_path")"
mode="$(stat -c '%a' -- "$final_path")"
printf '%s\t%s\t%s\t%s\n' "$sha" "$bytes" "$mode" "$final_path"
REMOTE_DUMP
)"; then
        fail "remote pg_dump or pg_restore --list validation failed; staging artifact was retained"
    fi
    IFS=$'\t' read -r remote_sha remote_bytes remote_mode returned_path <<<"$output"
    if [[ ! "$remote_sha" =~ ^[0-9a-f]{64}$ \
        || ! "$remote_bytes" =~ ^[1-9][0-9]*$ \
        || "$remote_mode" != "600" \
        || "$returned_path" != "$remote_dump_path" ]]; then
        fail "remote dump evidence was malformed"
    fi
    REMOTE_DUMP_SHA256="$remote_sha"
    REMOTE_DUMP_BYTES="$remote_bytes"
    export REMOTE_DUMP_SHA256 REMOTE_DUMP_BYTES
    log "remote_dump=validated bytes=$REMOTE_DUMP_BYTES sha256=$REMOTE_DUMP_SHA256"
}

capture_and_compare_source_snapshot_b() {
    stage="source-snapshot-b"
    local source_a="$work_directory/source-a.json"
    local source_b="$work_directory/source-b.json"
    if ! remote_snapshot >"$source_b"; then
        fail "source consistency snapshot B failed"
    fi
    validate_snapshot_file "$source_b"
    if ! cmp --silent "$source_a" "$source_b"; then
        fail "source consistency changed during the backup window; backup was not copied or restored"
    fi
    log "source_snapshot_a_equals_b=true"
}

copy_dump_off_host() {
    stage="off-host-copy"
    local dump_name incomplete_path local_sha local_bytes
    dump_name="${remote_dump_path##*/}"
    backup_path="$PF_GATE85_BACKUP_DIR/$dump_name"
    incomplete_path="$backup_path.incomplete"
    manifest_path="$backup_path.manifest.json"
    if [[ -e "$backup_path" || -e "$incomplete_path" || -e "$manifest_path" ]]; then
        fail "refusing to overwrite an existing off-host artifact or manifest"
    fi
    if ! scp -- "$PF_GATE85_SOURCE_SSH:$remote_dump_path" "$incomplete_path"; then
        fail "SCP failed; the local incomplete file was not promoted"
    fi
    chmod 600 -- "$incomplete_path"
    if [[ ! -s "$incomplete_path" ]]; then
        fail "SCP produced an empty local incomplete file"
    fi
    local_sha="$(sha256sum -- "$incomplete_path" | awk '{print $1}')"
    local_bytes="$(stat -c '%s' -- "$incomplete_path")"
    if [[ "$local_sha" != "$REMOTE_DUMP_SHA256" \
        || "$local_bytes" != "$REMOTE_DUMP_BYTES" ]]; then
        fail "off-host checksum or byte count mismatch; restore was not started"
    fi
    mv -- "$incomplete_path" "$backup_path"
    chmod 600 -- "$backup_path"
    log "off_host_copy=verified path=$backup_path bytes=$local_bytes sha256=$local_sha"
}

remove_remote_staging_copy() {
    stage="remote-staging-cleanup"
    if ! ssh -- "$PF_GATE85_SOURCE_SSH" sh -s -- \
        gate85-remote-staging-cleanup "$remote_staging_directory" "$remote_dump_path" <<'REMOTE_CLEANUP'
set -eu
action="$1"
staging_directory="$2"
dump_path="$3"
if [ "$action" != "gate85-remote-staging-cleanup" ]; then
    exit 91
fi
case "$staging_directory" in
    /tmp/pathfinder-gate85.[0-9a-f]*) ;;
    *) exit 92 ;;
esac
case "$dump_path" in
    "$staging_directory"/*-gate85.dump) ;;
    *) exit 93 ;;
esac
if [ ! -f "$dump_path" ]; then
    exit 94
fi
rm -- "$dump_path"
rmdir -- "$staging_directory"
REMOTE_CLEANUP
    then
        fail "verified off-host copy exists, but exact remote staging cleanup failed"
    fi
    log "remote_staging_cleanup=exact-script-created-artifact-only"
    remote_staging_directory="removed-after-verified-copy"
    remote_dump_path="removed-after-verified-copy"
}

start_restore_postgres() {
    stage="restore-postgres-start"
    if ! compose up --detach --no-deps --wait --wait-timeout 60 postgres >/dev/null; then
        fail "isolated restore PostgreSQL did not become healthy"
    fi
    log "restore_postgres=healthy api=not-started worker=not-started caddy=not-started"
}

restore_dump() {
    stage="pg-restore"
    if ! compose exec -T postgres sh -ec \
        'exec pg_restore --exit-on-error --no-owner --no-privileges --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
        <"$backup_path"; then
        fail "pg_restore failed; target database and restore resources were retained without retry"
    fi
    log "pg_restore=passed"
}

query_restore_revision() {
    compose exec -T postgres psql -X --quiet --tuples-only --no-align \
        --set ON_ERROR_STOP=1 --username=pathfinder --dbname=pathfinder \
        --command 'SELECT version_num FROM alembic_version'
}

migrate_restored_database() {
    stage="restored-migration"
    if ! compose run --rm --no-deps api alembic upgrade head >/dev/null; then
        fail "restored migration failed; no downgrade, worker start, retry, or cleanup was performed"
    fi
    restore_revision="$(query_restore_revision)"
    restore_revision="${restore_revision//$'\r'/}"
    restore_revision="${restore_revision//$'\n'/}"
    if [[ "$restore_revision" != "$candidate_head" ]]; then
        fail "restored database revision does not equal the candidate head"
    fi
    log "restore_revision=$restore_revision"
}

compare_restored_snapshot() {
    stage="restored-consistency"
    local restored_snapshot="$work_directory/restored-source-fixture.json"
    if ! target_snapshot "$PF_GATE85_FIXTURE_RUN_ID" "$fixture_document_id" \
        >"$restored_snapshot"; then
        fail "restored consistency snapshot failed"
    fi
    validate_snapshot_file "$restored_snapshot"
    if ! cmp --silent "$work_directory/source-a.json" "$restored_snapshot"; then
        fail "restored database does not exactly match the stable source snapshot; API and worker were not started"
    fi
    log "restored_consistency=exact-match"
}

start_restored_api_and_worker() {
    stage="restored-api-start"
    if ! compose up --detach --no-deps --wait --wait-timeout 60 api >/dev/null; then
        fail "restored API did not become healthy; worker was not started"
    fi
    if ! compose exec -T api python -c \
        'import urllib.request; response=urllib.request.urlopen("http://127.0.0.1:8000/readyz", timeout=2); raise SystemExit(0 if response.status == 200 else 1)' \
        >/dev/null; then
        fail "restored API internal readiness check failed; worker was not started"
    fi
    stage="restored-worker-start"
    if ! compose up --detach --no-deps --wait --wait-timeout 60 worker >/dev/null; then
        fail "restored worker did not become healthy"
    fi
    local worker_ids worker_count
    worker_ids="$(docker ps --all --quiet \
        --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT" \
        --filter 'label=com.docker.compose.service=worker')"
    worker_count="$(printf '%s\n' "$worker_ids" | line_count)"
    if [[ "$worker_count" -ne 1 ]]; then
        fail "restore project must have exactly one worker container"
    fi
    log "restored_api=healthy restored_worker=healthy worker_count=1 caddy=not-started"
}

api_call() {
    local method="$1"
    local path="$2"
    local payload="$3"
    compose exec -T api python -c '
import sys
import urllib.error
import urllib.request

method, path, payload = sys.argv[1:4]
data = payload.encode("utf-8") if payload else None
request = urllib.request.Request(
    "http://127.0.0.1:8000" + path,
    data=data,
    method=method,
    headers={"Content-Type": "application/json"} if data is not None else {},
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        sys.stdout.write(response.read().decode("utf-8"))
except urllib.error.HTTPError as error:
    sys.stderr.write(f"http_status={error.code}\n")
    raise SystemExit(1)
' "$method" "$path" "$payload"
}

poll_run_status() {
    local workspace_id="$1"
    local run_id="$2"
    local expected_status="$3"
    local deadline response status
    deadline=$((SECONDS + PF_GATE85_DEADLINE_SECONDS))
    while (( SECONDS < deadline )); do
        if ! response="$(api_call GET \
            "/api/v1/workspaces/$workspace_id/runs/$run_id" "")"; then
            fail "run status polling failed"
        fi
        status="$(printf '%s' "$response" | json_text_get status)"
        if [[ "$status" == "$expected_status" ]]; then
            printf '%s\n' "$response"
            return 0
        fi
        if [[ "$status" == "failed" || "$status" == "cancelled" ]]; then
            fail "restored E2E run reached unexpected terminal status $status"
        fi
        sleep 0.5
    done
    fail "deadline exceeded waiting for restored E2E run status $expected_status"
}

target_query() {
    local sql="$1"
    local run_id="$2"
    printf '%s\n' "$sql" | compose exec -T postgres psql \
        -X --quiet --tuples-only --no-align \
        --set ON_ERROR_STOP=1 --username=pathfinder --dbname=pathfinder \
        --set "run_id=$run_id"
}

sse_replay_check() {
    local workspace_id="$1"
    local run_id="$2"
    compose exec -T api python -c '
import json
import re
import sys
import urllib.request

workspace_id, run_id = sys.argv[1:3]
url = f"http://127.0.0.1:8000/api/v1/workspaces/{workspace_id}/runs/{run_id}/events"
with urllib.request.urlopen(url, timeout=10) as response:
    first = response.read().decode("utf-8")
ids = [int(value) for value in re.findall(r"(?m)^id: ([0-9]+)$", first)]
if not ids or ids != sorted(ids) or len(ids) != len(set(ids)):
    raise SystemExit("invalid initial SSE event sequence")
last_id = ids[-1]
request = urllib.request.Request(url, headers={"Last-Event-ID": str(last_id)})
with urllib.request.urlopen(request, timeout=10) as response:
    replay = response.read().decode("utf-8")
replayed_ids = [int(value) for value in re.findall(r"(?m)^id: ([0-9]+)$", replay)]
if any(value <= last_id for value in replayed_ids):
    raise SystemExit("SSE replay duplicated an event at or below Last-Event-ID")
print(json.dumps({"last_event_id": last_id, "replayed_ids": replayed_ids}, separators=(",", ":")))
' "$workspace_id" "$run_id"
}

run_restored_fake_e2e() {
    stage="restored-fake-e2e"
    local me actor_id workspace_id marker create_payload create_response waiting_response
    local action_id review request_version decision_payload persisted completed_response
    local job_status source_mock_count e2e_snapshot sse_result
    if ! me="$(api_call GET /api/v1/me "")"; then
        fail "restored fake identity discovery failed"
    fi
    actor_id="$(printf '%s' "$me" | python3 -c '
import json, sys
value = json.load(sys.stdin)
matches = [item for item in value["workspaces"] if item["kind"] == "personal" and item["role"] == "admin"]
if len(matches) != 1:
    raise SystemExit("expected exactly one personal admin workspace")
print(value["user_id"])
')"
    workspace_id="$(printf '%s' "$me" | python3 -c '
import json, sys
value = json.load(sys.stdin)
matches = [item for item in value["workspaces"] if item["kind"] == "personal" and item["role"] == "admin"]
if len(matches) != 1:
    raise SystemExit("expected exactly one personal admin workspace")
print(matches[0]["workspace_id"])
')"
    marker="gate85-restored-$(python3 -c 'import secrets; print(secrets.token_hex(8))')"
    create_payload="$(python3 -c '
import json, sys
print(json.dumps({"mode":"application","query":sys.argv[1],"resume_document_id":sys.argv[2]}, separators=(",", ":")))
' "$marker" "$fixture_document_id")"
    if ! create_response="$(api_call POST \
        "/api/v1/workspaces/$workspace_id/runs" "$create_payload")"; then
        fail "restored E2E run creation failed; mutating request was not retried"
    fi
    restored_e2e_run_id="$(printf '%s' "$create_response" | json_text_get run_id)"
    local PF_GATE85_E2E_RUN_ID="$restored_e2e_run_id"
    validate_uuid PF_GATE85_E2E_RUN_ID
    waiting_response="$(poll_run_status "$workspace_id" "$restored_e2e_run_id" waiting_approval)"
    if [[ "$(printf '%s' "$waiting_response" | json_text_get status)" != "waiting_approval" ]]; then
        fail "restored E2E did not reach waiting approval"
    fi
    action_id="$(target_query \
        "SELECT id FROM action_intents WHERE run_id = :'run_id'::uuid" \
        "$restored_e2e_run_id")"
    action_id="${action_id//$'\r'/}"
    action_id="${action_id//$'\n'/}"
    local PF_GATE85_E2E_ACTION_ID="$action_id"
    validate_uuid PF_GATE85_E2E_ACTION_ID
    if ! review="$(api_call GET \
        "/api/v1/workspaces/$workspace_id/action-intents/$action_id" "")"; then
        fail "restored E2E action review failed"
    fi
    request_version="$(printf '%s' "$review" | json_text_get approval_request.version)"
    if [[ ! "$request_version" =~ ^[1-9][0-9]*$ ]]; then
        fail "restored E2E approval request version is invalid"
    fi
    decision_payload="$(python3 -c '
import json, sys
print(json.dumps({"decision":"approve","expected_version":int(sys.argv[1]),"reason":"synthetic Gate 8.5 restore rehearsal"}, separators=(",", ":")))
' "$request_version")"
    if ! api_call POST \
        "/api/v1/workspaces/$workspace_id/action-intents/$action_id/decision" \
        "$decision_payload" >/dev/null; then
        if ! persisted="$(api_call GET \
            "/api/v1/workspaces/$workspace_id/action-intents/$action_id" "")"; then
            fail "approval response was uncertain and persisted state could not be inspected"
        fi
        if [[ "$(printf '%s' "$persisted" | json_text_get decision.decision)" != "approve" ]]; then
            fail "approval response was uncertain and no persisted approve decision exists; request was not retried"
        fi
        log "approval_response=uncertain persisted_decision=approve no_mutating_retry=true"
    fi
    completed_response="$(poll_run_status "$workspace_id" "$restored_e2e_run_id" completed)"
    if [[ "$(printf '%s' "$completed_response" | json_text_get status)" != "completed" ]]; then
        fail "restored E2E run did not complete"
    fi
    job_status="$(target_query \
        "SELECT status FROM run_jobs WHERE run_id = :'run_id'::uuid" \
        "$restored_e2e_run_id")"
    job_status="${job_status//$'\r'/}"
    job_status="${job_status//$'\n'/}"
    if [[ "$job_status" != "done" ]]; then
        fail "restored E2E job is not done"
    fi
    e2e_snapshot="$work_directory/restored-e2e.json"
    if ! target_snapshot "$restored_e2e_run_id" "$fixture_document_id" >"$e2e_snapshot"; then
        fail "restored E2E consistency snapshot failed"
    fi
    validate_snapshot_file "$e2e_snapshot"
    source_mock_count="$(target_query \
        "SELECT count(*) FROM mock_submissions WHERE run_id = :'run_id'::uuid" \
        "$PF_GATE85_FIXTURE_RUN_ID")"
    source_mock_count="${source_mock_count//$'\r'/}"
    source_mock_count="${source_mock_count//$'\n'/}"
    if [[ "$source_mock_count" != "1" ]]; then
        fail "worker startup changed the restored source fixture Mock submission count"
    fi
    if ! sse_result="$(sse_replay_check "$workspace_id" "$restored_e2e_run_id")"; then
        fail "restored E2E SSE replay check failed"
    fi
    printf '%s\n' "$sse_result" >"$work_directory/restored-e2e-sse.json"
    log "restored_e2e=passed run_id=$restored_e2e_run_id actor_id=$actor_id workspace_id=$workspace_id"
    log "restored_e2e_sse=$sse_result"
}

collect_restore_resources() {
    local kind="$1"
    case "$kind" in
        containers)
            docker ps --all --quiet \
                --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT" \
                | while IFS= read -r id; do
                    [[ -n "$id" ]] || continue
                    docker inspect --format '{{.Id}}|{{.Name}}|{{.State.Status}}' "$id"
                done
            ;;
        volumes)
            docker volume ls --quiet \
                --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT"
            ;;
        networks)
            docker network ls --quiet \
                --filter "label=com.docker.compose.project=$PF_GATE85_RESTORE_PROJECT"
            ;;
        *) return 2 ;;
    esac
}

write_manifest() {
    stage="manifest"
    local completed_utc="$1"
    local backup_seconds="$2"
    local restore_seconds="$3"
    local total_seconds="$4"
    local containers="$5"
    local volumes="$6"
    local networks="$7"
    if ! python3 -c '
import json
import sys
from pathlib import Path

(
    manifest_path,
    source_snapshot_path,
    e2e_snapshot_path,
    sse_path,
    completed_utc,
    source_project,
    source_postgres_image,
    source_postgres_version,
    source_revision,
    dump_path,
    dump_bytes,
    dump_sha256,
    fixture_run_id,
    fixture_document_id,
    restore_project,
    restore_revision,
    e2e_run_id,
    candidate_image,
    candidate_image_id,
    worktree_head,
    worktree_fingerprint,
    backup_seconds,
    restore_seconds,
    total_seconds,
    containers,
    volumes,
    networks,
) = sys.argv[1:]
source = json.loads(Path(source_snapshot_path).read_text(encoding="utf-8"))
e2e = json.loads(Path(e2e_snapshot_path).read_text(encoding="utf-8"))
sse = json.loads(Path(sse_path).read_text(encoding="utf-8"))
manifest = {
    "result": "PASS",
    "completed_utc": completed_utc,
    "source": {
        "project": source_project,
        "postgres_image": source_postgres_image,
        "postgres_version": source_postgres_version,
        "revision": source_revision,
    },
    "backup": {
        "path": dump_path,
        "bytes": int(dump_bytes),
        "sha256": dump_sha256,
        "off_host_checksum_verified": True,
    },
    "source_fixture": {
        "run_id": fixture_run_id,
        "document_id": fixture_document_id,
        "facts": source["fixture"],
        "global_counts": source["global_counts"],
        "checks": source["checks"],
    },
    "restore": {
        "project": restore_project,
        "revision": restore_revision,
        "consistency": "exact-match",
        "candidate_image": candidate_image,
        "candidate_image_id": candidate_image_id,
        "worktree_head": worktree_head,
        "worktree_fingerprint": "sha256:" + worktree_fingerprint,
        "resources_retained": {
            "containers": containers.splitlines(),
            "volumes": volumes.splitlines(),
            "networks": networks.splitlines(),
        },
    },
    "restored_fake_e2e": {
        "run_id": e2e_run_id,
        "facts": e2e["fixture"],
        "checks": e2e["checks"],
        "sse": sse,
    },
    "durations_seconds": {
        "backup_and_off_host_copy": float(backup_seconds),
        "restore_migration_consistency_e2e": float(restore_seconds),
        "total": float(total_seconds),
    },
    "destructive_cleanup": "NOT PERFORMED",
}
Path(manifest_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
' \
        "$manifest_path" \
        "$work_directory/source-a.json" \
        "$work_directory/restored-e2e.json" \
        "$work_directory/restored-e2e-sse.json" \
        "$completed_utc" \
        "$PF_GATE85_SOURCE_PROJECT" \
        "$source_postgres_image" \
        "$source_postgres_version" \
        "$source_revision" \
        "$backup_path" \
        "$REMOTE_DUMP_BYTES" \
        "$REMOTE_DUMP_SHA256" \
        "$PF_GATE85_FIXTURE_RUN_ID" \
        "$fixture_document_id" \
        "$PF_GATE85_RESTORE_PROJECT" \
        "$restore_revision" \
        "$restored_e2e_run_id" \
        "$PATHFINDER_IMAGE" \
        "$candidate_image_id" \
        "$worktree_head" \
        "$worktree_fingerprint" \
        "$backup_seconds" \
        "$restore_seconds" \
        "$total_seconds" \
        "$containers" \
        "$volumes" \
        "$networks"; then
        fail "private recovery manifest could not be written"
    fi
    chmod 600 -- "$manifest_path"
    log "manifest=$manifest_path mode=600"
}

stop_restore_services_without_destroying() {
    stage="restore-services-stop"
    if ! compose stop --timeout 10 worker api postgres >/dev/null; then
        fail "validation passed, but restore services could not be stopped; resources remain retained"
    fi
    log "restore_services=stopped containers_volumes_networks=retained"
}

seconds_between() {
    local start_ns="$1"
    local end_ns="$2"
    python3 -c 'import sys; print(f"{(int(sys.argv[2])-int(sys.argv[1]))/1_000_000_000:.3f}")' \
        "$start_ns" "$end_ns"
}

main() {
    local total_started backup_started backup_finished restore_started restore_finished total_finished
    local backup_seconds restore_seconds total_seconds completed_utc containers volumes networks
    total_started="$(date +%s%N)"
    static_preflight
    acquire_lock
    remote_deployment_preflight
    local_restore_absence_preflight
    build_and_validate_candidate
    discover_fixture_document
    backup_started="$(date +%s%N)"
    capture_source_snapshot_a
    remote_create_dump
    capture_and_compare_source_snapshot_b
    copy_dump_off_host
    remove_remote_staging_copy
    backup_finished="$(date +%s%N)"
    restore_started="$backup_finished"
    start_restore_postgres
    restore_dump
    migrate_restored_database
    compare_restored_snapshot
    start_restored_api_and_worker
    run_restored_fake_e2e
    restore_finished="$(date +%s%N)"
    stop_restore_services_without_destroying
    containers="$(collect_restore_resources containers)"
    volumes="$(collect_restore_resources volumes)"
    networks="$(collect_restore_resources networks)"
    completed_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    total_finished="$(date +%s%N)"
    backup_seconds="$(seconds_between "$backup_started" "$backup_finished")"
    restore_seconds="$(seconds_between "$restore_started" "$restore_finished")"
    total_seconds="$(seconds_between "$total_started" "$total_finished")"
    write_manifest "$completed_utc" "$backup_seconds" "$restore_seconds" "$total_seconds" \
        "$containers" "$volumes" "$networks"
    stage="success"
    log "Gate 8.5 validation: PASS"
    log "restore_project=$PF_GATE85_RESTORE_PROJECT"
    log "restore_containers=$(printf '%s' "$containers" | tr '\n' ',')"
    log "restore_volumes=$(printf '%s' "$volumes" | tr '\n' ',')"
    log "restore_networks=$(printf '%s' "$networks" | tr '\n' ',')"
    log "off_host_dump=$backup_path"
    log "manifest=$manifest_path"
    log "destructive_cleanup=NOT PERFORMED"
    log "TEMPORARY RESTORE ENVIRONMENT RETAINED"
    log "AWAITING EXPLICIT HUMAN DESTRUCTION APPROVAL"
}

main "$@"
