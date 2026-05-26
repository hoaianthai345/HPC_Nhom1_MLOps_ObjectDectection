# `secrets/` — Credentials cho pipeline

Thư mục này chứa các file credentials nhạy cảm. **Mọi file ở đây (trừ `.gitignore`
và `README.md`) đều bị git ignore** — không bao giờ commit lên GitHub.

## File hỗ trợ

### `kaggle.json` — Kaggle API token

1. Truy cập <https://www.kaggle.com/settings/account>.
2. Mục **API** → bấm **Create New API Token**. Trình duyệt tải về `kaggle.json`.
3. Copy file vừa tải vào đây:

   ```bash
   cp ~/Downloads/kaggle.json hpc_nhom1_code/secrets/kaggle.json
   chmod 600 hpc_nhom1_code/secrets/kaggle.json
   ```

4. Sau bước này, `smoke_demo_pipeline.sh` và các script khác tự dò tới
   `secrets/kaggle.json` thông qua biến `KAGGLE_CONFIG_DIR` — không cần copy
   vào `~/.kaggle/` nữa.

### `.env` các stack (nếu cần override mặc định)

- `infra/mlflow/.env` (đã có template `.env.example`)
- `infra/airflow/.env` (đã có template `.env.example`)
- Các credentials nội bộ này có template riêng trong từng `infra/<stack>/`, không
  cần copy vào `secrets/`.

## Kiểm tra

```bash
ls -la secrets/
# Phải thấy: .gitignore, README.md, kaggle.json (chỉ trên máy bạn)

# Test kaggle hoạt động
KAGGLE_CONFIG_DIR="$(pwd)/secrets" kaggle datasets list | head -3
```

## Quy tắc bảo mật

- ⚠️ KHÔNG share thư mục `secrets/` qua email/Slack/chat.
- ⚠️ KHÔNG copy nội dung file ra ngoài quy trình hợp lý (script hoặc CI).
- Khi rời nhóm hoặc nghi ngờ rò rỉ: vào tài khoản Kaggle/cloud xoá token cũ và sinh mới.
