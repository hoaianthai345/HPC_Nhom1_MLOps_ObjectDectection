# `infra/swarm/` — Docker Swarm deployment

Stack này swarm-hoá hai thành phần **serving** (FastAPI + Gradio UI) và
**monitor** (Prometheus/Grafana/Alertmanager/Loki/pushgateway/node_exporter).
MLflow và Airflow vẫn được vận hành bằng `docker compose` thường (xem
`infra/mlflow/` và `infra/airflow/`) vì có thành phần stateful (MySQL/Postgres)
gắn chặt vào 1 node.

Mọi service swarm kết nối qua overlay attachable `mlops-overlay`, để các stack
compose ngoài (MLflow/Airflow) vẫn `--network mlops-overlay` join được nếu cần.

## File trong thư mục

| File | Mô tả |
|---|---|
| `docker-stack.serving.yml` | 2 service: `api` (3 replicas, rolling update start-first, resource limit, healthcheck) + `ui` (1 replica) |
| `docker-stack.monitor.yml` | 6 service: prometheus / grafana / alertmanager / loki / pushgateway / node_exporter (mode global) |
| `prometheus-swarm.yml` | Scrape config dùng `dns_sd_configs` với `tasks.<service>` — tự pickup replica mới khi scale |
| `README.md` | File này |

## Quick start

```bash
# Khởi động đầy đủ (init swarm + registry + build + deploy)
bash scripts/start_swarm_stack.sh

# Verify 4 tính năng cho báo cáo (load-balance, rolling update, self-heal, scale)
bash scripts/verify_swarm_features.sh

# Dừng
bash scripts/stop_swarm_stack.sh
```

Sau khi `start_swarm_stack.sh` xong, các URL truy cập giữ nguyên như mode
compose:

| Dịch vụ | URL | Tài khoản |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| Gradio UI | http://localhost:7860 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin/admin` |
| Alertmanager | http://localhost:9093 | — |

## Vận hành — các lệnh thường dùng

```bash
# Liệt kê stack + service
docker stack ls
docker stack services mlops

# Xem task của 1 service (ai chạy ở node nào, state gì)
docker stack ps mlops --filter desired-state=running

# Logs (gộp mọi replica)
docker service logs mlops_api -f --tail 100

# Scale on-demand
docker service scale mlops_api=5

# Rolling update đổi image hoặc env
docker service update --image localhost:5000/serving-api:v2 mlops_api
docker service update --env-add YOLO_CONFIDENCE_THRESHOLD=0.30 mlops_api

# Rollback nhanh khi update fail
docker service rollback mlops_api

# Force re-create 1 replica (test self-heal thủ công)
docker rm -f $(docker ps --filter "name=mlops_api." --format '{{.ID}}' | head -1)
```

## Cấu hình quan trọng

### `deploy.update_config` — rolling update

```yaml
update_config:
  parallelism: 1          # update 1 replica mỗi đợt
  delay: 10s              # nghỉ 10s giữa các đợt
  order: start-first      # tạo replica mới TRƯỚC khi tắt cũ (zero-downtime)
  failure_action: rollback
  monitor: 30s            # 30s healthcheck sau khi start mới
```

### `deploy.restart_policy` — self-heal

```yaml
restart_policy:
  condition: any          # restart cả khi exit code 0
  delay: 5s
  max_attempts: 3
  window: 60s
```

### `deploy.resources` — giới hạn tài nguyên

```yaml
resources:
  limits:    { cpus: "2.0", memory: 3G }
  reservations: { cpus: "0.5", memory: 1G }
```

Limit để chạy đúng cấu hình benchmark trong báo cáo; reservation đảm bảo
Docker không schedule replica lên node thiếu tài nguyên.

## So sánh với compose mode

| Tiêu chí | Compose (`infra/monitor/`, `serving_pipeline/`) | Swarm (`infra/swarm/`) |
|---|---|---|
| Số replicas FastAPI | 1 cố định | 3 mặc định, scale 1-N |
| Update zero-downtime | Không | Có (start-first) |
| Self-heal khi crash | Không (chỉ `restart: unless-stopped`) | Có, swarm scheduler tự re-place |
| Service discovery | container_name | DNS service: `tasks.<service>` |
| Prometheus target | host.docker.internal | `tasks.api` qua dns_sd |
| Image source | local build trực tiếp | Local registry `localhost:5000` |
| Phù hợp | Dev local, debug code | Demo orchestration, benchmark scale |

## Rollback về compose (nếu cần dừng swarm)

```bash
bash scripts/stop_swarm_stack.sh --leave-swarm
# rồi khởi động lại stack compose như cũ
bash scripts/start_full_local.sh
```

Cả hai mode dùng chung `infra/mlflow/` và `infra/airflow/` (vẫn ở compose).

## Lưu ý đã biết

1. **Single-node deployment** — máy local chỉ có 1 node manager. Mọi
   `placement.constraints: node.role == manager` đều thoả; multi-node chỉ cần
   `docker swarm join` thêm worker.
2. **Local registry `localhost:5000`** — không có TLS. Production phải thay
   bằng registry có HTTPS (Harbor, Docker Hub, GitHub Container Registry).
3. **Volume `production_data`** — local volume, nếu scale ra nhiều node phải
   chuyển sang NFS/Longhorn hoặc thiết kế lại serving để không cần state.
4. **MinIO + MySQL của MLflow** vẫn ở compose; serving swarm gọi MLflow qua
   network `mlops-overlay` attachable.
