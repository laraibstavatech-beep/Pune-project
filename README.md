# F1 Conveyor Command Center

A racing-inspired dashboard for YOLO-based conveyor part counting.

## Features
- Live YOLO + ByteTrack counting with line crossing logic.
- F1-style dashboard with analog gauge-like widgets (speed + production).
- Live camera view integrated in UI.
- Production vs Target graph.
- Automatic daily Excel workbook: `YYYY-MM-DD.xlsx` with:
  - `production` sheet
  - `downtime` sheet
- Downtime detection after 2 minutes of no crossing, with remark prompt.
- Start/Stop detection controls.
- Live API:
  - `GET /api/live`
  - `GET /api/files`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables
- `MODEL_PATH` (default: `yolo26n.pt`)
- `CAMERA_SOURCE` (default: `0`)
- `CONF_THRESHOLD` (default: `0.25`)
- `LINE_REL_X` (default: `0.5`)
- `COUNT_DIRECTION` (default: `both`)
- `TARGET_PER_HOUR` (default: `900`)
- `STOP_TIMEOUT_SECONDS` (default: `120`)
- `OUTPUT_DIR` (default: `daily_reports`)
