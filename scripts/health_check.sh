#!/bin/bash
# Comprehensive health check for stock-screener on Synology DS220+
# Run from ~/stock-screener:  bash scripts/health_check.sh

FRONTEND="http://localhost:7777"
API="$FRONTEND/api/v1"
PASS=0; FAIL=0; WARN=0

green()  { printf "\033[32m[PASS]\033[0m %s\n" "$*"; PASS=$((PASS+1)); }
red()    { printf "\033[31m[FAIL]\033[0m %s\n" "$*"; FAIL=$((FAIL+1)); }
yellow() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; WARN=$((WARN+1)); }
header() { printf "\n\033[1m=== %s ===\033[0m\n" "$*"; }

jq_or() {
  # Parse JSON field with python3 (available in backend container)
  # Usage: jq_or "$JSON" "d.get('field','fallback')" "fallback"
  local json="$1" expr="$2" fallback="${3:-unknown}"
  local result
  result=$(printf '%s' "$json" | docker exec -i stock-screener-backend-1 \
    python3 -c "import sys,json; d=json.load(sys.stdin); print($expr)" 2>/dev/null) || true
  echo "${result:-$fallback}"
}

api_get() {
  # Curl through the nginx proxy; returns body or "ERROR:HTTP_CODE"
  local path="$1"
  local http_code body
  body=$(curl -s -w "\n%{http_code}" "$API$path" 2>/dev/null) || true
  http_code=$(printf '%s' "$body" | tail -1)
  body=$(printf '%s' "$body" | sed '$d')
  if [ "$http_code" = "200" ]; then
    printf '%s' "$body"
  else
    printf 'ERROR:%s' "${http_code:-0}"
  fi
}

# ── 1. DOCKER CONTAINERS ──────────────────────────────────────────────────────
header "1. Docker containers"
for c in \
  "stock-screener-backend-1" \
  "stock-screener-postgres-1" \
  "stock-screener-redis-1" \
  "stock-screener-frontend-1" \
  "stock-screener-celery-beat-1" \
  "stock-screener-celery-general-1" \
  "stock-screener-celery-datafetch-1" \
  "stock-screener-celery-marketjobs-us-1"
do
  status=$(docker inspect "$c" --format '{{.State.Status}}' 2>/dev/null || echo "missing")
  health=$(docker inspect "$c" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}' 2>/dev/null || echo "n/a")
  if [ "$status" = "running" ]; then
    if [ "$health" = "unhealthy" ]; then
      red "$c — running but UNHEALTHY"
    else
      green "$c — running ($health)"
    fi
  else
    red "$c — $status"
  fi
done

# ── 2. BACKEND HEALTH ─────────────────────────────────────────────────────────
header "2. Backend /readyz (checked inside container)"
readyz=$(docker exec stock-screener-backend-1 curl -sf http://localhost:8000/readyz 2>/dev/null) || readyz=""
if [ -n "$readyz" ]; then
  db=$(jq_or "$readyz" "d.get('database','?')" "?")
  redis=$(jq_or "$readyz" "d.get('redis','?')" "?")
  status=$(jq_or "$readyz" "d.get('status','?')" "?")
  if echo "$readyz" | grep -q '"status"'; then
    green "Backend ready — status=$status  database=$db  redis=$redis"
  else
    yellow "Backend responded but unexpected format: $(printf '%s' "$readyz" | head -c 120)"
  fi
else
  red "Backend /readyz unreachable inside container"
fi

# ── 3. FRONTEND PROXY ─────────────────────────────────────────────────────────
header "3. Frontend nginx proxy (port 7777)"
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND/" 2>/dev/null) || http_code=0
if [ "$http_code" = "200" ]; then
  green "Frontend serving at $FRONTEND (HTTP $http_code)"
else
  red "Frontend not responding at $FRONTEND (HTTP $http_code)"
fi

# Verify API proxy works
proxy_code=$(curl -s -o /dev/null -w "%{http_code}" "$API/features" 2>/dev/null) || proxy_code=0
if [ "$proxy_code" = "200" ]; then
  green "API proxy working — /api/v1/features returned HTTP $proxy_code"
else
  yellow "API proxy check — /api/v1/features returned HTTP $proxy_code (may need auth)"
fi

# ── 4. PRICE CACHE STATUS ─────────────────────────────────────────────────────
header "4. Price cache & data freshness"
cache_resp=$(api_get "/cache/market-status")
if printf '%s' "$cache_resp" | grep -q "^ERROR"; then
  code=$(printf '%s' "$cache_resp" | cut -d: -f2)
  yellow "Cache market-status returned HTTP $code (may require warm cache)"
else
  overall=$(jq_or "$cache_resp" "d.get('overall_status', d.get('status','unknown'))" "unknown")
  last=$(jq_or "$cache_resp" "d.get('last_refreshed_trading_day', d.get('last_refresh','unknown'))" "unknown")
  if printf '%s' "$overall" | grep -qi "ok\|complete\|fresh"; then
    green "Price cache — status=$overall  last_refresh=$last"
  else
    yellow "Price cache — status=$overall  last_refresh=$last"
  fi
fi

# Check staleness via DB directly
stale_count=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT COUNT(*) FROM stock_universe su WHERE su.is_active=true AND su.market='US'
   AND NOT EXISTS (
     SELECT 1 FROM stock_prices sp WHERE sp.symbol=su.symbol
     AND sp.date >= (SELECT MAX(date)-3 FROM stock_prices)
   );" 2>/dev/null | tr -d ' \n') || stale_count=""
if [ -n "$stale_count" ] && [ "$stale_count" != "" ]; then
  if [ "$stale_count" -eq "0" ] 2>/dev/null; then
    green "Stale active symbols (no price in last 3 days): 0"
  else
    yellow "Stale active symbols: $stale_count (will be deactivated on next pipeline run)"
  fi
fi

# ── 5. GROUP RANKINGS ─────────────────────────────────────────────────────────
header "5. Group rankings"
groups_resp=$(api_get "/groups/rankings/current?market=US&limit=5")
if printf '%s' "$groups_resp" | grep -q "^ERROR"; then
  code=$(printf '%s' "$groups_resp" | cut -d: -f2)
  yellow "Group rankings HTTP $code (pipeline not yet run today, or stale data)"
else
  count=$(jq_or "$groups_resp" "len(d.get('rankings', d.get('groups', d.get('data',[]))))" "0")
  asof=$(jq_or "$groups_resp" "d.get('as_of_date', d.get('date','unknown'))" "unknown")
  if [ "$count" -gt "0" ] 2>/dev/null; then
    green "Group rankings — $count groups returned  as_of=$asof"
  else
    yellow "Group rankings — 0 groups (pipeline hasn't run today yet)"
  fi
fi

# ── 6. BREADTH DATA ───────────────────────────────────────────────────────────
header "6. Market breadth"
breadth_resp=$(api_get "/breadth/current?market=US")
if printf '%s' "$breadth_resp" | grep -q "^ERROR"; then
  code=$(printf '%s' "$breadth_resp" | cut -d: -f2)
  yellow "Breadth endpoint HTTP $code"
else
  bdate=$(jq_or "$breadth_resp" "d.get('date', d.get('as_of_date','unknown'))" "unknown")
  adv=$(jq_or "$breadth_resp" "str(d.get('advancing', d.get('adv','?')))" "?")
  dec=$(jq_or "$breadth_resp" "str(d.get('declining', d.get('dec','?')))" "?")
  if [ "$bdate" != "unknown" ] && [ "$bdate" != "null" ] && [ -n "$bdate" ]; then
    green "Breadth data — date=$bdate  adv=$adv  dec=$dec"
  else
    yellow "Breadth — no data yet"
  fi
fi

# ── 7. MARKET SNAPSHOT ────────────────────────────────────────────────────────
header "7. Market scan snapshot"
snap_resp=$(api_get "/market-scan/daily-snapshot?market=US")
if printf '%s' "$snap_resp" | grep -q "^ERROR"; then
  code=$(printf '%s' "$snap_resp" | cut -d: -f2)
  yellow "Market snapshot HTTP $code (pipeline not yet run today)"
else
  scount=$(jq_or "$snap_resp" "len(d.get('rows', d.get('stocks', d.get('data',[]))))" "0")
  sdate=$(jq_or "$snap_resp" "d.get('as_of_date', d.get('meta',{}).get('as_of_date','unknown'))" "unknown")
  if [ "$scount" -gt "0" ] 2>/dev/null; then
    green "Market snapshot — $scount stocks  as_of=$sdate"
  else
    yellow "Market snapshot — 0 stocks (pipeline not yet run today)"
  fi
fi

# ── 8. PIPELINE ACTIVITY ──────────────────────────────────────────────────────
header "8. Pipeline activity"
act_resp=$(api_get "/runtime/activity")
if printf '%s' "$act_resp" | grep -q "^ERROR"; then
  red "Runtime activity endpoint unreachable"
else
  sum_status=$(jq_or "$act_resp" "d.get('summary',{}).get('status','unknown')" "unknown")
  us_status=$(jq_or "$act_resp" \
    "next((m for m in d.get('markets',[]) if m.get('market')=='US'),{}).get('status','no US entry')" \
    "unknown")
  us_stage=$(jq_or "$act_resp" \
    "next((m for m in d.get('markets',[]) if m.get('market')=='US'),{}).get('stage_key','-')" \
    "-")
  us_msg=$(jq_or "$act_resp" \
    "next((m for m in d.get('markets',[]) if m.get('market')=='US'),{}).get('message','')" \
    "")
  if printf '%s' "$us_status" | grep -qi "failed\|stuck"; then
    red "Pipeline US — $us_status  stage=$us_stage  msg=$us_msg"
  elif printf '%s' "$us_status" | grep -qi "running\|queued"; then
    yellow "Pipeline US — ACTIVE ($us_status)  stage=$us_stage"
  else
    green "Pipeline US — $us_status  stage=$us_stage  summary=$sum_status"
  fi
fi

# ── 9. CELERY BEAT ────────────────────────────────────────────────────────────
header "9. Celery-beat scheduler"
beat_line=$(docker logs stock-screener-celery-beat-1 2>&1 | grep -E "beat|Scheduler|daily" | tail -3)
if [ -n "$beat_line" ]; then
  green "celery-beat running — recent log:"
  printf '%s\n' "$beat_line" | sed 's/^/    /'
else
  yellow "celery-beat — no recent log found (may have just started)"
fi

# ── 10. DATABASE ──────────────────────────────────────────────────────────────
header "10. Database key records"

active=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT COUNT(*) FROM stock_universe WHERE is_active=true AND market='US';" 2>/dev/null | tr -d ' \n') || active=0
[ "$active" -gt "0" ] 2>/dev/null && \
  green "Active US symbols: $active" || red "Active US symbols: ${active:-0} (expected > 0)"

latest=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT MAX(date) FROM stock_prices sp
   JOIN stock_universe su ON sp.symbol=su.symbol
   WHERE su.market='US' AND su.is_active=true;" 2>/dev/null | tr -d ' \n') || latest=""
[ -n "$latest" ] && green "Latest price date: $latest" || yellow "Could not read latest price date"

migration=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT string_agg(version_num, ', ') FROM alembic_version;" 2>/dev/null | tr -d ' \n') || migration=""
green "DB migration: ${migration:-unknown}"

refresh=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT value FROM app_settings WHERE key='market.refresh_state.US';" 2>/dev/null | tr -d '\n') || refresh=""
if [ -n "$refresh" ]; then
  r_day=$(jq_or "$refresh" "d.get('last_refreshed_trading_day','?')" "?")
  r_st=$(jq_or "$refresh" "d.get('status','?')" "?")
  green "Last price refresh: trading_day=$r_day  status=$r_st"
else
  yellow "No market refresh state yet (runs after first price refresh)"
fi

sched=$(docker exec stock-screener-postgres-1 psql -U stockscanner stockscanner -t -c \
  "SELECT COUNT(*) FROM app_settings WHERE key LIKE 'runtime.activity.%';" 2>/dev/null | tr -d ' \n') || sched=0
green "Runtime activity records in DB: ${sched:-0}"

# ── SUMMARY ───────────────────────────────────────────────────────────────────
header "SUMMARY"
printf "\033[32mPASS: %d\033[0m  \033[33mWARN: %d\033[0m  \033[31mFAIL: %d\033[0m\n" "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt "0" ]; then
  echo "Action needed: investigate FAIL items above."
elif [ "$WARN" -gt "0" ]; then
  echo "System healthy. WARN items will clear after tonight's pipeline run (00:30 UAE)."
else
  echo "All checks passed. System fully operational."
fi
