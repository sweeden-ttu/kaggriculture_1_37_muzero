#!/usr/bin/env bash
# =============================================================================
# train2.sh - Alias to boost.sh
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/boost.sh" "$@"
