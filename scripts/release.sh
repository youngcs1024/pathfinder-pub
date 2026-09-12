#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly ACTION="${1:-}"
readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly PROJECT_NAME="${COMPOSE_PROJECT_NAME:-pathfinder}"
readonly LOCK_ROOT="${TMPDIR:-/tmp}"

stage="release-lock"
lock_directory=""
lock_acquired=0
deployment_kind="unknown"
candidate_image_id="unavailable"
candidate_head="unknown"
previous_api_image="none"
previous_worker_image="none"
previous_api_image_id="none"
previous_worker_image_id="none"
revision_before="unknown"
revision_after="unknown"
backup_path="none"

log() {
    printf 'release: %s\n' "$*"
}

diagnose() {
    printf 'release: stage=%s\n' "$stage" >&2
    printf 'release: action=%s deployment=%s\n' "${ACTION:-missing}" "$deployment_kind" >&2
    printf 'release: candidate_image=%s candidate_image_id=%s\n' \
        "${PATHFINDER_IMAGE:-missing}" "$candidate_image_id" >&2
    printf 'release: previous_api_image=%s previous_worker_image=%s\n' \
        "$previous_api_image" "$previous_worker_image" >&2
    printf 'release: revision_before=%s candidate_head=%s revision_after=%s\n' \
        "$revision_before" "$candidate_head" "$revision_after" >&2
}

fail() {
    printf 'release: error: %s\n' "$1" >&2
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
    printf 'release: interrupted by %s during stage=%s; automatic processing stopped\n' \
        "$signal_name" "$stage" >&2
    if [[ "$stage" == "migration" ]]; then
        printf '%s\n' \
            'release: interrupted during migration; inspect the current Alembic revision before the next action' >&2
    else
        printf '%s\n' \
            'release: inspect current database and API/worker/Caddy state before rerunning' >&2
    fi
    diagnose
    exit 130
}

trap cleanup_lock EXIT
trap 'interrupted SIGINT' INT
trap 'interrupted SIGTERM' TERM
trap 'interrupted SIGHUP' HUP

require_nonempty() {
    local variable_name="$1"
    if [[ -z "${!variable_name:-}" ]]; then
        fail "$variable_name is required"
    fi
}

validate_project_name() {
    if [[ ! "$PROJECT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        fail "COMPOSE_PROJECT_NAME must contain only letters, digits, dot, underscore, or hyphen"
    fi
}

acquire_lock() {
    validate_project_name
    lock_directory="${LOCK_ROOT%/}/pathfinder-release-${PROJECT_NAME}.lock"
    if ! mkdir -- "$lock_directory" 2>/dev/null; then
        printf 'release: error: another release holds lock %s\n' "$lock_directory" >&2
        printf '%s\n' \
            'release: do not remove it until no release process is running and DB/API/worker/Caddy state has been inspected' >&2
        exit 1
    fi
    lock_acquired=1
    printf 'pid=%s\naction=%s\nstarted_utc=%s\n' \
        "$$" "${ACTION:-missing}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$lock_directory/owner"
}

validate_action() {
    case "$ACTION" in
        release | rollback) ;;
        *) fail "usage: ./scripts/release.sh release|rollback" ;;
    esac
}

validate_image_reference() {
    if [[ "$PATHFINDER_IMAGE" == "pathfinder-ci:local" ]]; then
        fail "PATHFINDER_IMAGE must not use the local default tag"
    fi

    if [[ "${PF_RELEASE_REHEARSAL:-0}" == "1" ]]; then
        if [[ "${PF_LLM_MODE:-fake}" != "fake" \
            || "${PF_SEARCH_MODE:-fake}" != "fake" \
            || "${PF_AUTH_MODE:-fake}" != "fake" \
            || "${PF_TRACE_MODE:-off}" != "off" ]]; then
            fail "PF_RELEASE_REHEARSAL=1 requires fake/fake/fake/off modes"
        fi
        log "image_reference_kind=local-worktree-rehearsal (not production immutable evidence)"
        return
    fi

    if [[ "$PATHFINDER_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
        log "image_reference_kind=registry-digest"
        return
    fi
    if [[ "$PATHFINDER_IMAGE" =~ :(commit-)?[0-9a-f]{7,40}$ ]]; then
        log "image_reference_kind=commit-identified-tag (tag identity is not registry immutability)"
        return
    fi
    fail "PATHFINDER_IMAGE must be a registry digest or commit-SHA-identified tag"
}

validate_backup_directory() {
    local canonical_backup
    if ! canonical_backup="$(realpath -m -- "$PF_RELEASE_BACKUP_DIR")"; then
        fail "PF_RELEASE_BACKUP_DIR cannot be resolved"
    fi
    if [[ "$canonical_backup" == "$REPOSITORY_ROOT" \
        || "$canonical_backup" == "$REPOSITORY_ROOT/"* ]]; then
        fail "PF_RELEASE_BACKUP_DIR must be outside the repository"
    fi
    PF_RELEASE_BACKUP_DIR="$canonical_backup"
    export PF_RELEASE_BACKUP_DIR
}

static_preflight() {
    stage="static-preflight"
    require_nonempty PATHFINDER_IMAGE
    require_nonempty PF_PUBLIC_DOMAIN
    require_nonempty PF_RELEASE_BACKUP_DIR
    require_nonempty PF_DATABASE_URL
    require_nonempty PF_POSTGRES_PASSWORD
    validate_image_reference
    validate_backup_directory

    if ! command -v docker >/dev/null 2>&1; then
        fail "docker is required"
    fi
    if [[ -n "${PF_RELEASE_EDGE_SMOKE_URL:-}" ]]; then
        if ! command -v curl >/dev/null 2>&1; then
            fail "curl is required when PF_RELEASE_EDGE_SMOKE_URL is set"
        fi
        if [[ "${PF_RELEASE_REHEARSAL:-0}" == "1" ]]; then
            if [[ ! "$PF_RELEASE_EDGE_SMOKE_URL" =~ ^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?/$ ]]; then
                fail "rehearsal edge smoke must target a loopback root URL"
            fi
        elif [[ ! "$PF_RELEASE_EDGE_SMOKE_URL" =~ ^https://[^/?#]+/$ ]]; then
            fail "production edge smoke must target an HTTPS root URL without query or fragment"
        fi
    fi

    if ! candidate_image_id="$(docker image inspect --format '{{.Id}}' "$PATHFINDER_IMAGE" 2>/dev/null)"; then
        fail "candidate image is not available to Docker"
    fi
    if [[ -z "$candidate_image_id" ]]; then
        fail "candidate image has no inspected image ID"
    fi
    log "provided_image=$PATHFINDER_IMAGE"
    log "inspected_image_id=$candidate_image_id"

    if ! docker compose config --quiet >/dev/null 2>&1; then
        fail "docker compose config validation failed"
    fi
    validate_candidate_settings api
    validate_candidate_settings worker
    candidate_head="$(read_candidate_head)"
    log "candidate_alembic_head=$candidate_head"
}

validate_candidate_settings() {
    local service="$1"
    if ! docker compose run --rm --no-deps --entrypoint python "$service" -c \
        'from app.config import Settings; Settings()' >/dev/null 2>&1; then
        fail "candidate $service settings validation failed"
    fi
}

read_candidate_head() {
    local output heads
    if ! output="$(docker compose run --rm --no-deps --entrypoint alembic api heads 2>/dev/null)"; then
        fail "candidate Alembic head query failed"
    fi
    heads="$(printf '%s\n' "$output" | awk '$2 == "(head)" {print $1}')"
    if [[ "$(printf '%s\n' "$heads" | awk 'NF {count += 1} END {print count + 0}')" -ne 1 ]]; then
        fail "candidate image must contain exactly one Alembic head"
    fi
    if [[ ! "$heads" =~ ^[A-Za-z0-9_]+$ ]]; then
        fail "candidate Alembic head has an invalid value"
    fi
    printf '%s\n' "$heads"
}

compose_ids() {
    local service="$1"
    docker compose ps --all --quiet "$service" 2>/dev/null
}

line_count() {
    awk 'NF {count += 1} END {print count + 0}'
}

inspect_container() {
    local container_id="$1"
    docker inspect --format \
        '{{.Config.Image}}|{{.Image}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}' \
        "$container_id" 2>/dev/null
}

detect_deployment() {
    stage="deployment-state-preflight"
    local api_ids worker_ids caddy_ids postgres_ids
    local api_count worker_count caddy_count postgres_count
    if ! api_ids="$(compose_ids api)" \
        || ! worker_ids="$(compose_ids worker)" \
        || ! caddy_ids="$(compose_ids caddy)" \
        || ! postgres_ids="$(compose_ids postgres)"; then
        fail "cannot inspect Compose deployment state"
    fi
    api_count="$(printf '%s\n' "$api_ids" | line_count)"
    worker_count="$(printf '%s\n' "$worker_ids" | line_count)"
    caddy_count="$(printf '%s\n' "$caddy_ids" | line_count)"
    postgres_count="$(printf '%s\n' "$postgres_ids" | line_count)"

    if [[ "$postgres_count" -gt 1 ]]; then
        deployment_kind="inconsistent"
        fail "multiple PostgreSQL containers exist for this Compose project"
    fi
    if [[ "$api_count" -eq 0 && "$worker_count" -eq 0 && "$caddy_count" -eq 0 ]]; then
        deployment_kind="fresh"
        log "deployment=fresh"
        return
    fi
    if [[ "$api_count" -ne 1 || "$worker_count" -ne 1 || "$caddy_count" -ne 1 \
        || "$postgres_count" -ne 1 ]]; then
        deployment_kind="inconsistent"
        fail "partial deployment: API, worker, Caddy, and PostgreSQL must each have exactly one container"
    fi

    deployment_kind="existing"
    local api_inspect worker_inspect caddy_inspect postgres_inspect
    if ! api_inspect="$(inspect_container "$api_ids")" \
        || ! worker_inspect="$(inspect_container "$worker_ids")" \
        || ! caddy_inspect="$(inspect_container "$caddy_ids")" \
        || ! postgres_inspect="$(inspect_container "$postgres_ids")"; then
        deployment_kind="inconsistent"
        fail "cannot inspect existing deployment containers"
    fi
    IFS='|' read -r previous_api_image previous_api_image_id api_state api_health \
        <<<"$api_inspect"
    IFS='|' read -r previous_worker_image previous_worker_image_id worker_state worker_health \
        <<<"$worker_inspect"
    local _caddy_image _caddy_id caddy_state caddy_health
    local _postgres_image _postgres_id postgres_state postgres_health
    IFS='|' read -r _caddy_image _caddy_id caddy_state caddy_health <<<"$caddy_inspect"
    IFS='|' read -r _postgres_image _postgres_id postgres_state postgres_health \
        <<<"$postgres_inspect"

    if [[ "$previous_api_image" != "$previous_worker_image" \
        || "$previous_api_image_id" != "$previous_worker_image_id" ]]; then
        deployment_kind="inconsistent"
        fail "existing API and worker image references or IDs do not match"
    fi
    if [[ "$api_state" != "running" || "$api_health" != "healthy" \
        || "$worker_state" != "running" || "$worker_health" != "healthy" \
        || "$caddy_state" != "running" \
        || "$postgres_state" != "running" || "$postgres_health" != "healthy" ]]; then
        deployment_kind="inconsistent"
        fail "existing deployment is not fully running and healthy"
    fi
    log "deployment=existing"
    log "previous_api_image=$previous_api_image previous_api_image_id=$previous_api_image_id"
    log "previous_worker_image=$previous_worker_image previous_worker_image_id=$previous_worker_image_id"
}

bootstrap_postgres_if_fresh() {
    if [[ "$deployment_kind" != "fresh" ]]; then
        return
    fi
    stage="postgres-bootstrap"
    if ! docker compose up --detach --no-deps --wait --wait-timeout 60 postgres \
        >/dev/null 2>&1; then
        fail "fresh PostgreSQL bootstrap did not become healthy"
    fi
    log "fresh PostgreSQL is healthy; API, worker, and Caddy remain stopped"
}

query_database_revision() {
    docker compose exec -T postgres sh -ec '
        exists="$(psql -X --set ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --tuples-only --no-align --command="SELECT to_regclass('"'"'public.alembic_version'"'"') IS NOT NULL")"
        if [ "$exists" = "t" ]; then
            exec psql -X --set ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --tuples-only --no-align --command="SELECT version_num FROM alembic_version"
        fi
        printf "base\n"
    ' 2>/dev/null
}

database_preflight() {
    stage="database-preflight"
    if ! revision_before="$(query_database_revision)"; then
        fail "cannot query current database Alembic revision"
    fi
    revision_before="${revision_before//$'\r'/}"
    if [[ ! "$revision_before" =~ ^(base|[A-Za-z0-9_]+)$ ]]; then
        fail "database returned an invalid or multiple Alembic revision"
    fi
    log "database_revision_before=$revision_before"

    if [[ "$ACTION" == "rollback" ]]; then
        if [[ "$deployment_kind" != "existing" ]]; then
            fail "rollback requires an existing application deployment and an explicit target image"
        fi
        if [[ "$revision_before" != "$candidate_head" ]]; then
            fail "schema has moved; code rollback refused, use forward repair"
        fi
        return
    fi

    if [[ "$deployment_kind" == "existing" \
        && "$revision_before" != "$candidate_head" \
        && "${PF_RELEASE_SCHEMA_COMPATIBLE:-0}" != "1" ]]; then
        fail "existing schema change requires PF_RELEASE_SCHEMA_COMPATIBLE=1 after manual compatibility review"
    fi
}

create_backup() {
    stage="backup"
    if ! mkdir -p -- "$PF_RELEASE_BACKUP_DIR"; then
        fail "cannot create backup directory"
    fi
    local timestamp release_id incomplete_path
    timestamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
    release_id="${candidate_image_id#sha256:}"
    release_id="${release_id:0:12}"
    backup_path="$PF_RELEASE_BACKUP_DIR/${timestamp}-${PROJECT_NAME}-${release_id}-pre-migration.dump"
    incomplete_path="${backup_path}.incomplete"
    if [[ -e "$backup_path" || -e "$incomplete_path" ]]; then
        fail "refusing to overwrite an existing backup artifact"
    fi

    if ! docker compose exec -T postgres sh -ec \
        'exec pg_dump --format=custom --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
        >"$incomplete_path" 2>/dev/null; then
        backup_path="$incomplete_path"
        fail "pg_dump failed; incomplete artifact was not promoted or removed"
    fi
    if [[ ! -s "$incomplete_path" ]]; then
        backup_path="$incomplete_path"
        fail "pg_dump produced an empty incomplete artifact"
    fi

    stage="backup-validation"
    if ! docker compose exec -T postgres sh -ec 'exec pg_restore --list' \
        <"$incomplete_path" >/dev/null 2>&1; then
        backup_path="$incomplete_path"
        fail "pg_restore --list rejected the incomplete artifact"
    fi
    if ! mv -- "$incomplete_path" "$backup_path"; then
        backup_path="$incomplete_path"
        fail "validated backup could not be promoted to its final path"
    fi
    log "backup_path=$backup_path"
    log "backup_bytes=$(wc -c <"$backup_path") validation=pg_restore-list-ok"
}

run_migration() {
    stage="migration"
    if ! docker compose run --rm --no-deps api alembic upgrade head >/dev/null 2>&1; then
        if revision_after="$(query_database_revision)"; then
            revision_after="${revision_after//$'\r'/}"
        else
            revision_after="unavailable"
        fi
        printf '%s\n' \
            'release: migration failed; candidate API/worker/Caddy promotion was not started' >&2
        printf '%s\n' \
            'release: failure does not prove the database is unchanged; inspect the recorded revision and use forward repair' >&2
        fail "candidate Alembic upgrade failed"
    fi

    stage="post-migration-revision"
    if ! revision_after="$(query_database_revision)"; then
        fail "cannot query database revision after migration"
    fi
    revision_after="${revision_after//$'\r'/}"
    if [[ "$revision_after" != "$candidate_head" ]]; then
        fail "database revision does not equal the candidate Alembic head after migration"
    fi
    log "database_revision_after=$revision_after"
}

promote_api_and_smoke() {
    stage="candidate-api"
    if ! docker compose up --detach --no-deps --wait --wait-timeout 60 api \
        >/dev/null 2>&1; then
        fail "candidate API did not become healthy; worker was not promoted"
    fi

    stage="internal-api-smoke"
    if ! docker compose exec -T api python -c \
        'import urllib.request; response = urllib.request.urlopen("http://127.0.0.1:8000/readyz", timeout=2); raise SystemExit(0 if response.status == 200 else 1)' \
        >/dev/null 2>&1; then
        fail "mandatory internal candidate API readiness smoke failed; worker was not promoted"
    fi
    log "candidate API healthy and internal /readyz smoke passed"
}

start_caddy_if_fresh() {
    if [[ "$deployment_kind" != "fresh" ]]; then
        log "existing Caddy left unchanged"
        return
    fi
    stage="caddy-initial-start"
    if ! docker compose up --detach --no-deps --wait --wait-timeout 60 caddy \
        >/dev/null 2>&1; then
        fail "initial Caddy start failed; worker was not promoted"
    fi
    log "Caddy initial start completed after API readiness"
}

run_optional_edge_smoke() {
    stage="edge-smoke"
    if [[ -z "${PF_RELEASE_EDGE_SMOKE_URL:-}" ]]; then
        log "edge_smoke=not-requested; internal API readiness remains the mandatory gate"
        return
    fi
    if ! curl --fail --silent --show-error --max-time 10 \
        --output /dev/null "$PF_RELEASE_EDGE_SMOKE_URL"; then
        fail "optional configured edge smoke failed; worker was not promoted"
    fi
    log "edge_smoke=passed"
}

promote_worker() {
    stage="candidate-worker"
    if ! docker compose up --detach --no-deps --wait --wait-timeout 60 worker \
        >/dev/null 2>&1; then
        fail "candidate worker did not become healthy"
    fi
    log "candidate worker healthy"
}

main() {
    acquire_lock
    validate_action
    static_preflight
    detect_deployment

    if [[ "$ACTION" == "rollback" && "$deployment_kind" != "existing" ]]; then
        fail "rollback refused because no previous application deployment exists"
    fi

    bootstrap_postgres_if_fresh
    database_preflight

    if [[ "$ACTION" == "release" ]]; then
        create_backup
        run_migration
    else
        revision_after="$revision_before"
        log "code rollback schema check passed; no database downgrade or restore will run"
    fi

    promote_api_and_smoke
    start_caddy_if_fresh
    run_optional_edge_smoke
    promote_worker
    stage="success"
    log "success action=$ACTION deployment=$deployment_kind image=$PATHFINDER_IMAGE"
    if [[ "$ACTION" == "release" ]]; then
        log "pre_migration_backup=$backup_path"
    fi
}

main
