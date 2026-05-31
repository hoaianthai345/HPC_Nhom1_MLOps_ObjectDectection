#!/usr/bin/env bash
#
# Verify 4 tính năng Swarm cho báo cáo §3.8:
#   (1) Load balancing — 20 request, kiểm tra phân phối đều giữa replicas
#   (2) Rolling update — đổi env, theo dõi update 1-by-1, không downtime
#   (3) Self-heal — kill 1 replica, đo thời gian swarm re-create
#   (4) Scale — service scale 3 → 5, verify Prometheus targets tự cập nhật
#
# Output evidence vào reports/swarm_evidence/<TIMESTAMP>/:
#   - summary.md       (báo cáo cho phụ lục)
#   - lb_distribution.txt
#   - rolling_timeline.txt
#   - selfheal_timing.txt
#   - scale_targets.txt
#
# Usage: bash scripts/verify_swarm_features.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK="mlops"
SERVICE="${STACK}_api"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$REPO_ROOT/reports/swarm_evidence/$TS"
mkdir -p "$OUT"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
phase(){ echo | tee -a "$OUT/run.log"; log "═══ $* ═══"; }
fail() { log "❌  $*"; exit 1; }

# Preflight
docker service inspect "$SERVICE" >/dev/null 2>&1 \
  || fail "Service $SERVICE chưa deploy. Chạy: bash scripts/start_swarm_stack.sh"

# ───── (1) LOAD BALANCING ─────
phase "1/4  Load balancing — 30 request → check distribution giữa replicas"
LB="$OUT/lb_distribution.txt"
: > "$LB"
for i in $(seq 1 30); do
  resp=$(curl -s -H "X-Probe: $i" http://localhost:8000/health 2>/dev/null)
  echo "$resp" >> "$LB"
done

# Đếm log mỗi replica nhận được bao nhiêu
log "Sample replicas tasks:"
docker service ps "$SERVICE" --filter desired-state=running --format \
  '  {{.Name}}  node={{.Node}}  state={{.CurrentState}}' | tee -a "$OUT/run.log"

# Đếm số log entry chứa "/health" trên mỗi replica trong 60s gần đây
log "Phân phối /health request 60s gần đây (từ docker service logs):"
docker service logs --since 60s "$SERVICE" 2>&1 \
  | grep "/health" \
  | awk '{print $1}' \
  | sort | uniq -c | sort -rn | head -10 | tee -a "$OUT/run.log"

# ───── (2) ROLLING UPDATE ─────
phase "2/4  Rolling update — đổi env YOLO_CONFIDENCE_THRESHOLD 0.25 → 0.30"
ROLL="$OUT/rolling_timeline.txt"

CURRENT_THR=$(docker service inspect "$SERVICE" \
  --format '{{range .Spec.TaskTemplate.ContainerSpec.Env}}{{println .}}{{end}}' \
  | grep YOLO_CONFIDENCE_THRESHOLD | cut -d= -f2)
log "Threshold hiện tại: $CURRENT_THR"

NEW_THR="0.30"; [ "$CURRENT_THR" = "0.30" ] && NEW_THR="0.25"
log "Đổi sang $NEW_THR"

# Khởi động background poller để đo downtime
(
  end=$(($(date +%s) + 60))
  while [ "$(date +%s)" -lt "$end" ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:8000/health)
    echo "$(date +%H:%M:%S) $code" >> "$ROLL"
    sleep 1
  done
) &
POLLER=$!

START_TS=$(date +%s)
docker service update --env-add "YOLO_CONFIDENCE_THRESHOLD=$NEW_THR" \
  --update-parallelism 1 \
  --update-delay 10s \
  --update-order start-first \
  --update-failure-action rollback \
  "$SERVICE" >/dev/null &

# Đợi update converge
while :; do
  state=$(docker service inspect "$SERVICE" --format '{{.UpdateStatus.State}}' 2>/dev/null)
  case "$state" in
    completed) break ;;
    paused|rollback_*) log "Update state: $state"; break ;;
    "") sleep 2 ;;
    *) sleep 2 ;;
  esac
  if [ "$(date +%s)" -gt "$((START_TS + 120))" ]; then
    log "Timeout 120s"
    break
  fi
done
DURATION=$(( $(date +%s) - START_TS ))
log "Rolling update hoàn tất sau ${DURATION}s"

wait "$POLLER" 2>/dev/null || true

# Đếm downtime
TOTAL=$(wc -l < "$ROLL" | tr -d ' ')
OK=$(grep -c "200" "$ROLL" || echo 0)
FAIL=$(( TOTAL - OK ))
log "Trong $TOTAL probe: $OK OK / $FAIL fail → downtime ratio = $(awk "BEGIN{printf \"%.1f\", $FAIL/$TOTAL*100}")%"
echo "Total=$TOTAL OK=$OK Fail=$FAIL Duration=${DURATION}s" >> "$ROLL"

# ───── (3) SELF-HEAL ─────
phase "3/4  Self-heal — kill 1 replica, đo recreate time"
HEAL="$OUT/selfheal_timing.txt"

VICTIM=$(docker ps --filter "name=${SERVICE}." --format '{{.ID}}' | head -1)
[ -n "$VICTIM" ] || fail "Không tìm thấy task của $SERVICE để kill"
VICTIM_NAME=$(docker inspect "$VICTIM" --format '{{.Name}}' | sed 's|^/||')
log "Victim: $VICTIM_NAME ($VICTIM)"

KILL_TS=$(date +%s)
docker rm -f "$VICTIM" >/dev/null
log "Killed at $(date +%H:%M:%S)"

# Đợi service có đủ replicas trở lại
EXPECT=$(docker service inspect "$SERVICE" --format '{{.Spec.Mode.Replicated.Replicas}}')
while :; do
  RUNNING=$(docker service ps "$SERVICE" --filter desired-state=running \
            --filter "current-state=running" -q | wc -l | tr -d ' ')
  if [ "$RUNNING" -ge "$EXPECT" ]; then break; fi
  if [ "$(date +%s)" -gt "$((KILL_TS + 60))" ]; then
    log "Timeout chờ self-heal 60s"
    break
  fi
  sleep 1
done
HEAL_TIME=$(( $(date +%s) - KILL_TS ))
log "Self-heal hoàn tất sau ${HEAL_TIME}s (expect=$EXPECT replicas)"

{
  echo "Victim:    $VICTIM_NAME"
  echo "Killed:    $(date -r "$KILL_TS" +'%F %T')"
  echo "Healed:    $(date +'%F %T')"
  echo "Duration:  ${HEAL_TIME}s"
  echo
  echo "Service ps sau self-heal:"
  docker service ps "$SERVICE" --format \
    '{{.Name}}\t{{.CurrentState}}\t{{.Node}}'
} > "$HEAL"

# ───── (4) SCALE ─────
phase "4/4  Scale 3 → 5 replicas + verify Prometheus tự nhận target mới"
SCALE="$OUT/scale_targets.txt"

# Trước scale
PRE=$(curl -s "http://localhost:9090/api/v1/query?query=count(up{job=%22serving_api%22}%20==%201)" \
      | python3 -c "import sys,json
try:
  r=json.load(sys.stdin)['data']['result']
  print(int(float(r[0]['value'][1])) if r else 0)
except: print(0)")
log "Trước scale: Prometheus thấy $PRE serving_api targets UP"

docker service scale "${SERVICE}=5" >/dev/null
log "Đã scale → 5; chờ converge + Prometheus discovery (45s)"
until [ "$(docker service ps "$SERVICE" --filter desired-state=running --filter current-state=running -q | wc -l | tr -d ' ')" -ge 5 ]; do
  sleep 2
done

# Chờ Prometheus pickup
sleep 30
POST=$(curl -s "http://localhost:9090/api/v1/query?query=count(up{job=%22serving_api%22}%20==%201)" \
       | python3 -c "import sys,json
try:
  r=json.load(sys.stdin)['data']['result']
  print(int(float(r[0]['value'][1])) if r else 0)
except: print(0)")
log "Sau scale: Prometheus thấy $POST serving_api targets UP"

{
  echo "Pre-scale targets:  $PRE"
  echo "Post-scale targets: $POST"
  echo "Expected:           5"
  echo
  echo "Service ps:"
  docker service ps "$SERVICE" --filter desired-state=running --format \
    '{{.Name}}\t{{.CurrentState}}\t{{.Node}}'
} > "$SCALE"

# Reset về 3 cho stable
log "Reset scale về 3 (cleanup)"
docker service scale "${SERVICE}=3" >/dev/null
sleep 5

# ───── SUMMARY ─────
phase "Sinh báo cáo evidence"
SUMMARY="$OUT/summary.md"
{
  echo "# Swarm Verification Evidence — $TS"
  echo
  echo "## (1) Load Balancing"
  echo
  echo "30 request /health phân phối qua \`tasks.api\` (swarm VIP)."
  echo
  echo '```'
  docker service logs --since 60s "$SERVICE" 2>&1 | grep "/health" \
    | awk '{print $1}' | sort | uniq -c | sort -rn | head -10
  echo '```'
  echo
  echo "## (2) Rolling Update — start-first, parallelism 1"
  echo
  echo "Đổi env \`YOLO_CONFIDENCE_THRESHOLD\` $CURRENT_THR → $NEW_THR. Duration **${DURATION}s**, downtime $FAIL/$TOTAL probe."
  echo
  echo "Service update status:"
  echo '```'
  docker service inspect "$SERVICE" --format \
    'state={{.UpdateStatus.State}}  parallelism={{.Spec.UpdateConfig.Parallelism}}  order={{.Spec.UpdateConfig.Order}}'
  echo '```'
  echo
  echo "## (3) Self-Heal"
  echo
  echo "Kill 1 task ngoài backend; swarm tự re-create."
  echo "**Recovery time: ${HEAL_TIME}s** để service quay lại $EXPECT replica."
  echo
  echo "## (4) Scale 3 → 5 + Prometheus auto-discovery"
  echo
  echo "| Trạng thái | targets UP |"
  echo "|---|---|"
  echo "| Trước scale | $PRE |"
  echo "| Sau scale + 30s | $POST |"
  echo "| Expect | 5 |"
  echo
  echo "## File chi tiết"
  echo
  echo "- \`lb_distribution.txt\` — payload 30 request"
  echo "- \`rolling_timeline.txt\` — probe 1s trong lúc rolling update"
  echo "- \`selfheal_timing.txt\` — chi tiết kill + recreate"
  echo "- \`scale_targets.txt\` — service ps + Prometheus count"
  echo "- \`run.log\` — full log"
} > "$SUMMARY"

log "✅ Evidence ở: $OUT"
log "✅ Summary: $SUMMARY"
