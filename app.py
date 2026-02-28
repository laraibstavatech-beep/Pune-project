import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover
    def get_script_run_ctx() -> Any:  # type: ignore
        return None


MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
LINE_REL_X = float(os.getenv("LINE_REL_X", "0.5"))
COUNT_DIRECTION = os.getenv("COUNT_DIRECTION", "both")
TARGET_PER_HOUR = int(os.getenv("TARGET_PER_HOUR", "900"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "daily_reports"))
STOP_TIMEOUT_SECONDS = int(os.getenv("STOP_TIMEOUT_SECONDS", "120"))
SPEED_WINDOW_SECONDS = int(os.getenv("SPEED_WINDOW_SECONDS", "60"))
AUTO_SAVE_SECONDS = int(os.getenv("AUTO_SAVE_SECONDS", "30"))


@dataclass
class DowntimeEvent:
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: int = 0
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "downtime_start": self.start_time,
            "downtime_end": self.end_time or datetime.now(),
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
        }


class ProductionState:
    def __init__(self) -> None:
        self.total_counts: dict[str, int] = defaultdict(int)
        self.current_counts: dict[str, int] = defaultdict(int)
        self.track_last_x: dict[int, int] = {}
        self.track_counted: dict[int, bool] = {}
        self.last_crossing_time = datetime.now()
        self.speed_crossings: deque[datetime] = deque(maxlen=5000)
        self.production_rows: list[dict[str, Any]] = []
        self.downtime_rows: list[dict[str, Any]] = []
        self.active_downtime: DowntimeEvent | None = None
        self.running = False
        self.last_frame: np.ndarray | None = None
        self.today_file = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        self.last_save_time = 0.0
        self.cap: cv2.VideoCapture | None = None

    @property
    def total_parts(self) -> int:
        return sum(self.total_counts.values())

    @property
    def current_speed_ppm(self) -> float:
        now = datetime.now()
        while self.speed_crossings and (now - self.speed_crossings[0]).total_seconds() > SPEED_WINDOW_SECONDS:
            self.speed_crossings.popleft()
        return float(len(self.speed_crossings))

    @property
    def elapsed_hours(self) -> float:
        if not self.production_rows:
            return 0.0
        first = self.production_rows[0]["timestamp"]
        return max((datetime.now() - first).total_seconds() / 3600.0, 1 / 3600)

    @property
    def target_till_now(self) -> float:
        return TARGET_PER_HOUR * self.elapsed_hours

    def save_to_excel(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_save_time) < AUTO_SAVE_SECONDS:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        production_df = pd.DataFrame(self.production_rows)
        downtime_df = pd.DataFrame(self.downtime_rows)
        with pd.ExcelWriter(self.today_file, engine="openpyxl") as writer:
            production_df.to_excel(writer, index=False, sheet_name="production")
            downtime_df.to_excel(writer, index=False, sheet_name="downtime")
        self.last_save_time = now


GLOBAL_STATE = ProductionState()


def in_streamlit_runtime() -> bool:
    return get_script_run_ctx() is not None


def get_state() -> ProductionState:
    if in_streamlit_runtime():
        if "prod_state" not in st.session_state:
            st.session_state.prod_state = ProductionState()
        return st.session_state.prod_state
    return GLOBAL_STATE


class Snapshot(BaseModel):
    timestamp: str
    total_parts: int
    target_till_now: float
    speed_ppm: float
    total_counts: dict[str, int]
    current_counts: dict[str, int]
    running: bool


api = FastAPI(title="Conveyor Live Data API")


@api.get("/api/live")
def live_data() -> JSONResponse:
    state = get_state()
    snap = Snapshot(
        timestamp=datetime.now().isoformat(),
        total_parts=state.total_parts,
        target_till_now=state.target_till_now,
        speed_ppm=state.current_speed_ppm,
        total_counts=dict(state.total_counts),
        current_counts=dict(state.current_counts),
        running=state.running,
    )
    return JSONResponse(content=snap.model_dump())


@api.get("/api/files")
def list_files() -> dict[str, list[str]]:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    return {"files": sorted([p.name for p in OUTPUT_DIR.glob("*.xlsx")])}


def run_api() -> None:
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")


@st.cache_resource(show_spinner=False)
def get_model() -> YOLO:
    return YOLO(MODEL_PATH)


def ensure_api_started() -> None:
    if not st.session_state.get("api_started"):
        threading.Thread(target=run_api, daemon=True).start()
        st.session_state.api_started = True


def get_camera(state: ProductionState) -> cv2.VideoCapture:
    if state.cap is None:
        state.cap = cv2.VideoCapture(CAMERA_SOURCE)
        state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return state.cap


def process_video_frame(frame: np.ndarray, model: YOLO, line_x: int, state: ProductionState) -> np.ndarray:
    state.current_counts = defaultdict(int)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        ids = boxes.id.cpu().numpy()

        for i, box in enumerate(xyxy_arr):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = model.names.get(int(cls_ids[i]), str(int(cls_ids[i])))
            track_id = int(ids[i])
            c_x = int((x1 + x2) / 2.0)
            state.current_counts[class_name] += 1

            last_x = state.track_last_x.get(track_id)
            if last_x is None:
                state.track_last_x[track_id] = c_x
                state.track_counted.setdefault(track_id, False)
            else:
                crossed_right = last_x < line_x and c_x >= line_x
                crossed_left = last_x > line_x and c_x <= line_x
                do_count = (
                    COUNT_DIRECTION == "both" and (crossed_right or crossed_left)
                ) or (COUNT_DIRECTION == "right" and crossed_right) or (COUNT_DIRECTION == "left" and crossed_left)

                if do_count and not state.track_counted.get(track_id, False):
                    state.total_counts[class_name] += 1
                    state.track_counted[track_id] = True
                    cross_time = datetime.now()
                    state.last_crossing_time = cross_time
                    state.speed_crossings.append(cross_time)
                    state.production_rows.append(
                        {
                            "timestamp": cross_time,
                            "class_name": class_name,
                            "running_total": state.total_parts,
                            "speed_ppm": state.current_speed_ppm,
                        }
                    )
                state.track_last_x[track_id] = c_x

            color = (0, 255, 0) if state.track_counted.get(track_id, False) else (255, 80, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} #{track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 3)
    return frame


def create_gauge(value: float, title: str, maximum: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 18, "color": "#EAEAEA"}},
            gauge={
                "axis": {"range": [0, max(1, maximum)]},
                "bar": {"color": color},
                "bgcolor": "#121212",
                "steps": [
                    {"range": [0, maximum * 0.6], "color": "#1B5E20"},
                    {"range": [maximum * 0.6, maximum * 0.85], "color": "#F9A825"},
                    {"range": [maximum * 0.85, maximum], "color": "#B71C1C"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor="#101820", plot_bgcolor="#101820", font_color="#EAEAEA", height=260)
    return fig


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: radial-gradient(circle at top, #1a1d25, #090a0f); color: #F2F2F2; }
          .f1-card { border: 1px solid #ff1744; border-radius: 14px; padding: 10px; background: linear-gradient(135deg,#0f1118,#161b24); box-shadow: 0 0 15px rgba(255,23,68,.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_downtime_state(state: ProductionState) -> None:
    idle_seconds = (datetime.now() - state.last_crossing_time).total_seconds()
    if idle_seconds > STOP_TIMEOUT_SECONDS and state.active_downtime is None:
        state.active_downtime = DowntimeEvent(start_time=state.last_crossing_time + timedelta(seconds=STOP_TIMEOUT_SECONDS))


def main() -> None:
    state = get_state()
    st.set_page_config(page_title="F1 Conveyor Command Center", layout="wide")
    apply_theme()

    st.title("🏁 F1 Conveyor Command Center")
    st.caption("Fast-load mode: model and camera initialize only when you click Start Detection.")

    ensure_api_started()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start detection", use_container_width=True):
            with st.spinner("Loading YOLO model + camera..."):
                get_model()
                get_camera(state)
            state.running = True
    with c2:
        if st.button("⏸ Stop detection", use_container_width=True):
            state.running = False
    with c3:
        if st.button("💾 Save workbook now", use_container_width=True):
            state.save_to_excel(force=True)
            st.success(f"Saved: {state.today_file}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="f1-card">Live Camera Feed</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with right:
        st.plotly_chart(create_gauge(state.current_speed_ppm, "Speed (parts/min)", 200, "#00E676"), use_container_width=True)
        st.plotly_chart(create_gauge(state.total_parts, "Production", max(TARGET_PER_HOUR * 8, 1), "#FF1744"), use_container_width=True)
        remark = "On Pace ✅" if state.total_parts >= state.target_till_now else "Below Pace ⚠️"
        st.markdown(f"**Race Engineer Remark:** {remark}")
        st.markdown(f"Target till now: **{state.target_till_now:.1f}**  ")
        st.markdown(f"Actual: **{state.total_parts}**")

    bar = go.Figure()
    bar.add_trace(go.Bar(x=["Actual", "Target"], y=[state.total_parts, state.target_till_now], marker_color=["#FF1744", "#00E5FF"]))
    bar.update_layout(title="Production vs Target", paper_bgcolor="#101820", plot_bgcolor="#101820", font_color="#EAEAEA")
    st.plotly_chart(bar, use_container_width=True)

    if state.active_downtime is not None and not state.active_downtime.reason:
        st.warning("Conveyor stopped for 2+ minutes. Please enter reason.")
        reason = st.text_input("Downtime reason", key="downtime_reason")
        if st.button("Submit downtime reason") and reason.strip():
            state.active_downtime.reason = reason.strip()
            state.active_downtime.end_time = datetime.now()
            state.active_downtime.duration_seconds = int((state.active_downtime.end_time - state.active_downtime.start_time).total_seconds())
            state.downtime_rows.append(state.active_downtime.as_row())
            state.active_downtime = None
            st.success("Downtime reason recorded.")

    if state.running:
        model = get_model()
        cap = get_camera(state)
        ret, frame = cap.read()
        if ret:
            processed = process_video_frame(frame, model, int(frame.shape[1] * LINE_REL_X), state)
            state.last_frame = processed
            update_downtime_state(state)
            frame_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error("Unable to read camera frame.")
    elif state.last_frame is not None:
        frame_placeholder.image(cv2.cvtColor(state.last_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    state.save_to_excel()


if __name__ == "__main__":
    if in_streamlit_runtime():
        main()
    else:
        print("Please launch the dashboard with: streamlit run app.py")
