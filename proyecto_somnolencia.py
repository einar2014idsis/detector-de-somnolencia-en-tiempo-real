from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from pygame import mixer


PROJECT_DIR = Path(__file__).resolve().parent
ALERT_PATH = PROJECT_DIR / "alerta.wav"
EVENT_LOG_PATH = PROJECT_DIR / "registro_eventos.csv"

DEFAULT_EYE_THRESHOLD = 0.25
DEFAULT_CLOSED_SECONDS = 3.0
DEFAULT_MOUTH_THRESHOLD = 0.60
MIN_CAMERA_BRIGHTNESS = 5.0
MIN_CAMERA_CONTRAST = 5.0
MIN_SIGNAL_RATIO = 0.01
CALIBRATION_SECONDS = 3.0
YAWN_MIN_SECONDS = 0.75
EVENT_COOLDOWN_SECONDS = 2.0

LEFT_EYE = (33, 133, 159, 145)
RIGHT_EYE = (362, 263, 386, 374)

running = False
detection_thread: threading.Thread | None = None
closed_eye_start: float | None = None
yawn_start: float | None = None
drowsiness_active = False
yawn_active = False
alarm_playing = False
session_start: float | None = None
last_event_time = 0.0
blink_count = 0
yawn_count = 0
drowsiness_count = 0
calibrated_eye_threshold = DEFAULT_EYE_THRESHOLD
calibration_requested = True
eye_state_history: deque[tuple[float, bool]] = deque()
alert_sound = None
audio_available = False
closed_seconds_setting = DEFAULT_CLOSED_SECONDS
mouth_threshold_setting = DEFAULT_MOUTH_THRESHOLD
volume_setting = 0.75
show_mesh_setting = True


def initialize_audio() -> None:
    global alert_sound, audio_available
    try:
        mixer.init()
        alert_sound = mixer.Sound(str(ALERT_PATH))
        alert_sound.set_volume(0.75)
        audio_available = True
    except (FileNotFoundError, OSError):
        audio_available = False


def eye_aspect_ratio(indices, landmarks, shape) -> float:
    points = [
        np.array([landmarks[index].x * shape[1], landmarks[index].y * shape[0]])
        for index in indices
    ]
    horizontal = np.linalg.norm(points[0] - points[1])
    vertical = np.linalg.norm(points[2] - points[3])
    return float(vertical / horizontal) if horizontal else 0.0


def mouth_aspect_ratio(landmarks, shape) -> float:
    top = np.array([landmarks[13].x * shape[1], landmarks[13].y * shape[0]])
    bottom = np.array([landmarks[14].x * shape[1], landmarks[14].y * shape[0]])
    left = np.array([landmarks[78].x * shape[1], landmarks[78].y * shape[0]])
    right = np.array([landmarks[308].x * shape[1], landmarks[308].y * shape[0]])
    horizontal = np.linalg.norm(left - right)
    vertical = np.linalg.norm(top - bottom)
    return float(vertical / horizontal) if horizontal else 0.0


def camera_has_image(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    signal_ratio = float(np.count_nonzero(gray > 5) / gray.size)
    return (
        brightness >= MIN_CAMERA_BRIGHTNESS
        and contrast >= MIN_CAMERA_CONTRAST
        and signal_ratio >= MIN_SIGNAL_RATIO
    )


def draw_face_mesh(frame, landmarks) -> None:
    height, width, _ = frame.shape
    for start_index, end_index in mp_face_mesh.FACEMESH_TESSELATION:
        start = landmarks[start_index]
        end = landmarks[end_index]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (60, 190, 135), 1)


def draw_dashboard(
    frame: np.ndarray,
    state: str,
    ear: float,
    mar: float,
    fatigue_score: int,
    fps: float,
) -> None:
    alert_state = "alerta" in state or fatigue_score >= 60
    state_color = (30, 40, 230) if alert_state else (40, 190, 110)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (18, 25, 34), -1)
    cv2.putText(
        frame,
        state.upper(),
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        state_color,
        2,
    )
    cv2.putText(
        frame,
        f"EAR {ear:.2f}   MAR {mar:.2f}   FATIGA {fatigue_score}%",
        (14, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        frame,
        f"PARPADEOS {blink_count}   BOSTEZOS {yawn_count}   FPS {fps:.1f}",
        (14, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 220, 230),
        1,
    )


def calculate_fatigue_score(now: float) -> int:
    while eye_state_history and now - eye_state_history[0][0] > 60:
        eye_state_history.popleft()
    if not eye_state_history:
        return 0
    closed_ratio = sum(closed for _, closed in eye_state_history) / len(eye_state_history)
    score = closed_ratio * 65 + min(yawn_count * 6, 18) + min(drowsiness_count * 15, 30)
    return min(100, int(round(score)))


def log_event(event_type: str, ear: float, mar: float, fatigue_score: int) -> None:
    new_file = not EVENT_LOG_PATH.exists()
    with EVENT_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(("fecha", "hora", "evento", "ear", "mar", "fatiga"))
        now = datetime.now()
        writer.writerow(
            (
                now.date().isoformat(),
                now.strftime("%H:%M:%S"),
                event_type,
                f"{ear:.4f}",
                f"{mar:.4f}",
                fatigue_score,
            )
        )


def play_alarm(loop: bool = False) -> None:
    global alarm_playing
    if not audio_available or alert_sound is None:
        return
    alert_sound.set_volume(volume_setting)
    if loop:
        if not alarm_playing:
            alert_sound.play(loops=-1)
            alarm_playing = True
    else:
        alert_sound.play()


def stop_alarm() -> None:
    global alarm_playing
    if audio_available:
        mixer.stop()
    alarm_playing = False


def set_status(message: str, level: str = "info") -> None:
    colors = {
        "success": ("#dff3e7", "#17653a"),
        "warning": ("#fff2cc", "#7a5600"),
        "error": ("#fde7e7", "#a12622"),
        "info": ("#dceaf5", "#244f70"),
    }
    background, foreground = colors.get(level, colors["info"])

    def update() -> None:
        status_label.configure(text=message, bg=background, fg=foreground)

    try:
        window.after(0, update)
    except tk.TclError:
        pass


def update_live_metrics(ear: float, mar: float, fatigue_score: int) -> None:
    def update() -> None:
        ear_var.set(f"ear actual: {ear:.2f}")
        mar_var.set(f"mar actual: {mar:.2f}")
        blink_var.set(f"parpadeos: {blink_count}")
        yawn_var.set(f"bostezos: {yawn_count}")
        alert_var.set(f"alertas de sueño: {drowsiness_count}")
        fatigue_var.set(f"nivel de fatiga: {fatigue_score}%")
        if fatigue_score < 30:
            risk_var.set("riesgo: bajo")
        elif fatigue_score < 60:
            risk_var.set("riesgo: moderado")
        else:
            risk_var.set("riesgo: alto")

    try:
        window.after(0, update)
    except tk.TclError:
        pass


def detect() -> None:
    global running
    global closed_eye_start
    global yawn_start
    global drowsiness_active
    global yawn_active
    global last_event_time
    global blink_count
    global yawn_count
    global drowsiness_count
    global calibrated_eye_threshold
    global calibration_requested

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not capture.isOpened():
        running = False
        set_status("no se pudo abrir la cámara", "error")
        return

    set_status("analizando", "success")
    dark_frames = 0
    calibration_samples: list[float] = []
    calibration_started = time.time()
    previous_frame_time = time.perf_counter()
    smoothed_fps = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while running and capture.isOpened():
            ok, frame = capture.read()
            if not ok or frame is None:
                set_status("la cámara dejó de responder", "error")
                break

            if not camera_has_image(frame):
                dark_frames += 1
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 68), (18, 25, 34), -1)
                cv2.putText(
                    frame,
                    "SIN SENAL DE CAMARA",
                    (16, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                )
                if dark_frames == 5:
                    set_status("sin señal de cámara", "error")
                cv2.imshow("Monitor de somnolencia", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running = False
                continue

            if dark_frames:
                dark_frames = 0
                set_status("cámara activa", "success")

            now = time.time()
            perf_now = time.perf_counter()
            instant_fps = 1.0 / max(perf_now - previous_frame_time, 0.001)
            smoothed_fps = instant_fps if smoothed_fps == 0 else smoothed_fps * 0.9 + instant_fps * 0.1
            previous_frame_time = perf_now

            if calibration_requested:
                calibration_requested = False
                calibration_samples.clear()
                calibration_started = now
                set_status("calibrando mantén los ojos abiertos", "info")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            detected_faces = getattr(results, "multi_face_landmarks", None)
            state = "rostro no detectado"
            ear = 0.0
            mar = 0.0

            if detected_faces:
                landmarks = detected_faces[0].landmark
                left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, frame.shape)
                right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, frame.shape)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(landmarks, frame.shape)

                if show_mesh_setting:
                    draw_face_mesh(frame, landmarks)

                if now - calibration_started < CALIBRATION_SECONDS:
                    state = "calibrando"
                    if ear > 0.12:
                        calibration_samples.append(ear)
                elif calibration_samples:
                    baseline = float(np.median(calibration_samples))
                    calibrated_eye_threshold = min(0.32, max(0.15, baseline * 0.70))
                    calibration_samples.clear()
                    window.after(
                        0,
                        lambda value=calibrated_eye_threshold: threshold_var.set(
                            f"umbral calibrado: {value:.2f}"
                        ),
                    )
                    set_status("calibración completada", "success")

                eyes_closed = ear < calibrated_eye_threshold
                eye_state_history.append((now, eyes_closed))
                state = "atento"

                if eyes_closed:
                    if closed_eye_start is None:
                        closed_eye_start = now
                    closed_duration = now - closed_eye_start
                    state = f"ojos cerrados {closed_duration:.1f}s"
                    if closed_duration >= closed_seconds_setting:
                        state = "alerta de somnolencia"
                        if not drowsiness_active and now - last_event_time >= EVENT_COOLDOWN_SECONDS:
                            drowsiness_active = True
                            last_event_time = now
                            drowsiness_count += 1
                            fatigue_score = calculate_fatigue_score(now)
                            log_event("somnolencia", ear, mar, fatigue_score)
                            play_alarm(loop=True)
                else:
                    if closed_eye_start is not None:
                        closed_duration = now - closed_eye_start
                        if 0.08 <= closed_duration < closed_seconds_setting:
                            blink_count += 1
                    closed_eye_start = None
                    drowsiness_active = False
                    if not yawn_active:
                        stop_alarm()

                if mar > mouth_threshold_setting:
                    if yawn_start is None:
                        yawn_start = now
                    if now - yawn_start >= YAWN_MIN_SECONDS:
                        state = "bostezo detectado"
                        if not yawn_active and now - last_event_time >= EVENT_COOLDOWN_SECONDS:
                            yawn_active = True
                            last_event_time = now
                            yawn_count += 1
                            fatigue_score = calculate_fatigue_score(now)
                            log_event("bostezo", ear, mar, fatigue_score)
                            play_alarm()
                else:
                    yawn_start = None
                    yawn_active = False
            else:
                closed_eye_start = None
                yawn_start = None
                drowsiness_active = False
                yawn_active = False
                stop_alarm()

            fatigue_score = calculate_fatigue_score(now)
            draw_dashboard(frame, state, ear, mar, fatigue_score, smoothed_fps)
            update_live_metrics(ear, mar, fatigue_score)
            cv2.imshow("Monitor de somnolencia", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

    capture.release()
    cv2.destroyAllWindows()
    stop_alarm()
    set_status("detección detenida", "info")


def start_detection() -> None:
    global running, detection_thread, session_start, calibration_requested
    if running:
        set_status("la detección ya está activa", "info")
        return
    running = True
    session_start = time.time()
    calibration_requested = True
    detection_thread = threading.Thread(target=detect, daemon=True)
    detection_thread.start()


def stop_detection() -> None:
    global running
    running = False
    stop_alarm()


def request_calibration() -> None:
    global calibration_requested
    calibration_requested = True
    set_status("calibración solicitada", "info")


def reset_metrics() -> None:
    global blink_count, yawn_count, drowsiness_count, session_start
    blink_count = 0
    yawn_count = 0
    drowsiness_count = 0
    eye_state_history.clear()
    session_start = time.time() if running else None
    update_live_metrics(0.0, 0.0, 0)
    set_status("métricas reiniciadas", "info")


def exit_app() -> None:
    global running
    running = False
    stop_alarm()
    if audio_available:
        mixer.quit()
    window.after(250, window.destroy)


def update_session_clock() -> None:
    elapsed = 0 if session_start is None else int(time.time() - session_start)
    session_var.set(f"tiempo de sesión: {elapsed // 60:02d}:{elapsed % 60:02d}")
    window.after(1000, update_session_clock)


def sync_settings() -> None:
    global closed_seconds_setting
    global mouth_threshold_setting
    global volume_setting
    global show_mesh_setting
    closed_seconds_setting = float(closed_seconds_var.get())
    mouth_threshold_setting = float(mouth_threshold_var.get())
    volume_setting = float(volume_var.get())
    show_mesh_setting = bool(show_mesh_var.get())
    window.after(200, sync_settings)


window = tk.Tk()
window.title("Sistema de monitoreo de somnolencia")
window.geometry("430x640")
window.resizable(False, False)
window.protocol("WM_DELETE_WINDOW", exit_app)

style = ttk.Style()
style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
style.configure("Metric.TLabel", font=("Segoe UI", 10))

main_frame = ttk.Frame(window, padding=18)
main_frame.pack(fill="both", expand=True)

ttk.Label(main_frame, text="detección de somnolencia", style="Title.TLabel").pack(pady=(0, 12))

buttons = ttk.Frame(main_frame)
buttons.pack(fill="x", pady=(0, 12))
ttk.Button(buttons, text="iniciar", command=start_detection).grid(row=0, column=0, sticky="ew", padx=(0, 4))
ttk.Button(buttons, text="detener", command=stop_detection).grid(row=0, column=1, sticky="ew", padx=4)
ttk.Button(buttons, text="finalizar", command=exit_app).grid(row=0, column=2, sticky="ew", padx=(4, 0))
buttons.columnconfigure((0, 1, 2), weight=1)

status_label = tk.Label(
    main_frame,
    text="sistema listo",
    bg="#dceaf5",
    fg="#244f70",
    font=("Segoe UI", 10, "bold"),
    padx=8,
    pady=8,
)
status_label.pack(fill="x", pady=(0, 12))

metrics_frame = ttk.LabelFrame(main_frame, text="métricas en vivo", padding=12)
metrics_frame.pack(fill="x", pady=(0, 12))

ear_var = tk.StringVar(value="ear actual: --")
mar_var = tk.StringVar(value="mar actual: --")
blink_var = tk.StringVar(value="parpadeos: 0")
yawn_var = tk.StringVar(value="bostezos: 0")
alert_var = tk.StringVar(value="alertas de sueño: 0")
fatigue_var = tk.StringVar(value="nivel de fatiga: 0%")
risk_var = tk.StringVar(value="riesgo: bajo")
session_var = tk.StringVar(value="tiempo de sesión: 00:00")
threshold_var = tk.StringVar(value=f"umbral inicial: {DEFAULT_EYE_THRESHOLD:.2f}")

for variable in (
    ear_var,
    mar_var,
    blink_var,
    yawn_var,
    alert_var,
    fatigue_var,
    risk_var,
    session_var,
    threshold_var,
):
    ttk.Label(metrics_frame, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=1)

config_frame = ttk.LabelFrame(main_frame, text="configuración", padding=12)
config_frame.pack(fill="x", pady=(0, 12))

closed_seconds_var = tk.DoubleVar(value=DEFAULT_CLOSED_SECONDS)
mouth_threshold_var = tk.DoubleVar(value=DEFAULT_MOUTH_THRESHOLD)
volume_var = tk.DoubleVar(value=0.75)
show_mesh_var = tk.BooleanVar(value=True)

ttk.Label(config_frame, text="segundos con ojos cerrados").pack(anchor="w")
ttk.Scale(config_frame, from_=1.0, to=5.0, variable=closed_seconds_var).pack(fill="x", pady=(0, 7))
ttk.Label(config_frame, text="sensibilidad de bostezo").pack(anchor="w")
ttk.Scale(config_frame, from_=0.4, to=1.0, variable=mouth_threshold_var).pack(fill="x", pady=(0, 7))
ttk.Label(config_frame, text="volumen de alerta").pack(anchor="w")
ttk.Scale(config_frame, from_=0.0, to=1.0, variable=volume_var).pack(fill="x", pady=(0, 7))
ttk.Checkbutton(config_frame, text="mostrar malla facial", variable=show_mesh_var).pack(anchor="w")

extra_buttons = ttk.Frame(main_frame)
extra_buttons.pack(fill="x")
ttk.Button(extra_buttons, text="recalibrar", command=request_calibration).grid(row=0, column=0, sticky="ew", padx=(0, 4))
ttk.Button(extra_buttons, text="reiniciar métricas", command=reset_metrics).grid(row=0, column=1, sticky="ew", padx=(4, 0))
extra_buttons.columnconfigure((0, 1), weight=1)

initialize_audio()
if not audio_available:
    status_label.configure(text="audio no disponible", bg="#fff2cc", fg="#7a5600")

update_session_clock()
sync_settings()
window.mainloop()
