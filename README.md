# F1 Conveyor Command Center

A racing-inspired dashboard for YOLO-based conveyor part counting.

## What was optimized for faster opening
- **Lazy loading**: YOLO model and camera are loaded only when you click **Start detection**.
- **Model cache**: model uses `st.cache_resource`, so repeated reruns are much faster.
- **Throttled Excel writes**: workbook autosave is periodic (default every 30s) instead of every rerun.
- **API startup only once** in a background thread.

## Features
- Live YOLO + ByteTrack counting with line crossing logic.
- F1-style dashboard with analog gauge-like widgets (speed + production).
- Live camera view in UI.
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

## Important: avoid the ScriptRunContext warning
- Run the dashboard using exactly: `streamlit run app.py`
- Do **not** run `python app.py`.
- Do **not** run `streamlit run` without a target file, otherwise Streamlit looks for `streamlit_app.py` by default.

## Environment Variables
- `MODEL_PATH` (default: `yolo26n.pt`)
- `CAMERA_SOURCE` (default: `0`)
- `CONF_THRESHOLD` (default: `0.25`)
- `LINE_REL_X` (default: `0.5`)
- `COUNT_DIRECTION` (default: `both`)
- `TARGET_PER_HOUR` (default: `900`)
- `STOP_TIMEOUT_SECONDS` (default: `120`)
- `OUTPUT_DIR` (default: `daily_reports`)
- `AUTO_SAVE_SECONDS` (default: `30`)

## Note for first run
If model weights are not available locally, Ultralytics may download them on first use. This can take time depending on network and disk speed.
