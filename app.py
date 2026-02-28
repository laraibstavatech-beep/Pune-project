import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import threading

try:
    from ultralytics import YOLO
except Exception:  # ultralytics may not be installed in all environments
    YOLO = None


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
LINE_REL_X = float(os.getenv("LINE_REL_X", "0.5"))
COUNT_DIRECTION = os.getenv("COUNT_DIRECTION", "both")  # right/left/both
TARGET_PER_HOUR = int(os.getenv("TARGET_PER_HOUR", "900"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "daily_reports"))
STOP_TIMEOUT_SECONDS = int(os.getenv("STOP_TIMEOUT_SECONDS", "120"))
SPEED_WINDOW_SECONDS = int(os.getenv("SPEED_WINDOW_SECONDS", "60"))


@dataclass
class DowntimeEvent:
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: int = 0
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        end = self.end_time or datetime.now()
        return {
            "downtime_start": self.start_time,
            "downtime_end": end,
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
        }


class ProductionState:
    def __init__(self) -> None:
        self.total_counts: dict[str, int] = defaultdict(int)
        self.current_counts: dict[str, int] = defaultdict(int)
        self.track_last_x: dict[int, int] = {}
        self.track_counted: dict[int, bool] = {}
        self.last_crossing_time: datetime = datetime.now()
        self.speed_crossings: Deque[datetime] = deque(maxlen=5000)
        self.production_rows: list[dict[str, Any]] = []
        self.downtime_rows: list[dict[str, Any]] = []
        self.active_downtime: DowntimeEvent | None = None
        self.running = True
        self.last_frame: np.ndarray | None = None
        self.today_file = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.xlsx"

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

    def save_to_excel(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        production_df = pd.DataFrame(self.production_rows)
        downtime_df = pd.DataFrame(self.downtime_rows)
        with pd.ExcelWriter(self.today_file, engine="openpyxl") as writer:
            production_df.to_excel(writer, index=False, sheet_name="production")
            downtime_df.to_excel(writer, index=False, sheet_name="downtime")


state = ProductionState()


class Snapshot(BaseModel):
    timestamp: str
    total_parts: int
    target_till_now: float
    speed_ppm: float
    total_counts: dict[str, int]
    current_counts: dict[str, int]
    running: bool


api = FastAPI(title="Conveyor Live Data API")


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/api/live")
def live_data() -> JSONResponse:
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
    files = sorted([p.name for p in OUTPUT_DIR.glob("*.xlsx")])
    return {"files": files}


def run_api() -> None:
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")


def start_api_if_needed() -> None:
    if not st.session_state.get("api_started"):
        t = threading.Thread(target=run_api, daemon=True)
        t.start()
        st.session_state["api_started"] = True


def process_video_frame(frame: np.ndarray, model: Any, line_x: int) -> np.ndarray:
    state.current_counts = defaultdict(int)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD)

    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        ids = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else np.array(boxes.id)

        for i, box in enumerate(xyxy_arr):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(confs[i])
            cls_idx = int(cls_ids[i])
            class_name = model.names.get(cls_idx, str(cls_idx)) if hasattr(model, "names") else str(cls_idx)
            track_id = int(ids[i])
            c_x = int((x1 + x2) / 2.0)

            state.current_counts[class_name] += 1

            last_x = state.track_last_x.get(track_id)
            if last_x is None:
                state.track_last_x[track_id] = c_x
                state.track_counted.setdefault(track_id, False)
            else:
                if not state.track_counted.get(track_id, False):
                    crossed_right = last_x < line_x and c_x >= line_x
                    crossed_left = last_x > line_x and c_x <= line_x
                    do_count = (
                        (COUNT_DIRECTION == "both" and (crossed_right or crossed_left))
                        or (COUNT_DIRECTION == "right" and crossed_right)
                        or (COUNT_DIRECTION == "left" and crossed_left)
                    )
                    if do_count:
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

                        # conveyor resumes -> close downtime if pending reason exists in UI
                        if state.active_downtime and state.active_downtime.end_time is None:
                            state.active_downtime.end_time = datetime.now()
                            state.active_downtime.duration_seconds = int(
                                (state.active_downtime.end_time - state.active_downtime.start_time).total_seconds()
                            )
                state.track_last_x[track_id] = c_x

            color = (0, 255, 0) if state.track_counted.get(track_id, False) else (255, 80, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"{class_name} #{track_id} {conf:.2f}"
            cv2.putText(frame, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 3)
    return frame


def create_gauge(value: float, title: str, maximum: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 18, "color": "#EAEAEA"}},
            gauge={
                "axis": {"range": [0, max(1, maximum)], "tickcolor": "#A0A0A0"},
                "bar": {"color": color},
                "bgcolor": "#121212",
                "bordercolor": "#2A2A2A",
                "steps": [
                    {"range": [0, maximum * 0.6], "color": "#1B5E20"},
                    {"range": [maximum * 0.6, maximum * 0.85], "color": "#F9A825"},
                    {"range": [maximum * 0.85, maximum], "color": "#B71C1C"},
                ],
            },
            number={"font": {"color": "#FFFFFF"}},
        )
    )
    fig.update_layout(paper_bgcolor="#101820", plot_bgcolor="#101820", margin=dict(l=20, r=20, t=45, b=20), height=280)
    return fig


def ensure_model():
    if YOLO is None:
        st.error("Ultralytics is not installed. Install dependencies from requirements.txt")
        st.stop()
    if "model" not in st.session_state:
        st.session_state.model = YOLO(MODEL_PATH)
    return st.session_state.model


def init_video_if_needed() -> cv2.VideoCapture:
    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(CAMERA_SOURCE)
    return st.session_state.cap


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: radial-gradient(circle at top, #1a1d25, #090a0f); color: #F2F2F2; }
          .f1-card { border: 1px solid #ff1744; border-radius: 14px; padding: 10px; background: linear-gradient(135deg,#0f1118,#161b24); box-shadow: 0 0 15px rgba(255,23,68,.25); }
          .f1-title { font-size: 1.2rem; font-weight: 700; color: #ff5252; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_downtime_state() -> None:
    no_crossing_for = (datetime.now() - state.last_crossing_time).total_seconds()
    if no_crossing_for > STOP_TIMEOUT_SECONDS and state.active_downtime is None:
        state.active_downtime = DowntimeEvent(start_time=state.last_crossing_time + timedelta(seconds=STOP_TIMEOUT_SECONDS))


def render_dashboard() -> None:
    st.set_page_config(page_title="F1 Conveyor Command Center", layout="wide")
    apply_theme()
    start_api_if_needed()

    st.title("🏁 F1 Conveyor Command Center")
    st.caption("YOLOv12 live counting • racing inspired controls • production + downtime intelligence")

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="f1-card"><div class="f1-title">Live Camera Feed</div></div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with right:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(create_gauge(state.current_speed_ppm, "Speed (Parts/min)", 200, "#00E676"), use_container_width=True)
        with g2:
            st.plotly_chart(create_gauge(state.total_parts, "Production", max(TARGET_PER_HOUR * 8, 1), "#FF1744"), use_container_width=True)

        remark = "On Pace ✅" if state.total_parts >= state.target_till_now else "Below Pace ⚠️"
        st.markdown(f"### Race Engineer Remark: **{remark}**")
        st.markdown(f"- Target till now: **{state.target_till_now:.1f}** parts")
        st.markdown(f"- Actual total: **{state.total_parts}** parts")
        st.markdown(f"- Live API: `GET /api/live` on port `8000`")

    prod_df = pd.DataFrame(
        {
            "metric": ["Actual", "Target"],
            "parts": [state.total_parts, state.target_till_now],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=prod_df["metric"], y=prod_df["parts"], marker_color=["#FF1744", "#00E5FF"]))
    fig.update_layout(title="Production vs Target", paper_bgcolor="#101820", plot_bgcolor="#101820", font_color="#EAEAEA")
    st.plotly_chart(fig, use_container_width=True)

    if state.active_downtime is not None and not state.active_downtime.reason:
        with st.container(border=True):
            st.warning("Conveyor appears stopped for 2+ minutes. Please enter downtime reason.")
            reason = st.text_input("Downtime reason", key="downtime_reason")
            if st.button("Submit downtime reason") and reason.strip():
                state.active_downtime.reason = reason.strip()
                if state.active_downtime.end_time is None:
                    state.active_downtime.end_time = datetime.now()
                    state.active_downtime.duration_seconds = int(
                        (state.active_downtime.end_time - state.active_downtime.start_time).total_seconds()
                    )
                state.downtime_rows.append(state.active_downtime.as_row())
                state.active_downtime = None
                st.success("Downtime reason recorded.")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("▶ Start detection", use_container_width=True):
            state.running = True
    with col_btn2:
        if st.button("⏸ Stop detection", use_container_width=True):
            state.running = False
    with col_btn3:
        if st.button("💾 Save workbook now", use_container_width=True):
            state.save_to_excel()
            st.success(f"Saved: {state.today_file}")

    model = ensure_model()
    cap = init_video_if_needed()

    if state.running:
        ret, frame = cap.read()
        if ret:
            line_x = int(frame.shape[1] * LINE_REL_X)
            vis_frame = process_video_frame(frame, model, line_x)
            state.last_frame = vis_frame
            update_downtime_state()
            rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
        else:
            st.error("Cannot read camera frame.")

    if state.last_frame is not None and not state.running:
        rgb = cv2.cvtColor(state.last_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

    state.save_to_excel()


if __name__ == "__main__":
    render_dashboard()
