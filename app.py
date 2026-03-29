from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import jwt
import mediapipe as mp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import split_on_silence
from werkzeug.security import check_password_hash, generate_password_hash


logging.basicConfig(filename="debug.log", level=logging.DEBUG)

app = Flask(__name__, static_folder="dist", static_url_path="/")
CORS(app)
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "app_data.db"))

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE")

db_lock = threading.Lock()
whisper_lock = threading.Lock()
whisper_model: WhisperModel | None = None


def utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_lock:
        conn = get_db()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    token TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    mode TEXT,
                    audience_level TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    feedback TEXT
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS explain_sessions (
                    session_id TEXT PRIMARY KEY,
                    question_index INTEGER NOT NULL DEFAULT 0,
                    pending_questions TEXT NOT NULL DEFAULT '[]',
                    teacher_responses TEXT NOT NULL DEFAULT '[]',
                    original_text TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def parse_json_field(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def generate_chat_title(transcript: str) -> str:
    first_part = (transcript or "").strip().split(".")[0][:50]
    return f"Chat: {first_part or 'New session'}..."


def create_chat_session(transcript: str, mode: str, audience_level: str, session_id: str | None = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    now = utcnow()
    with db_lock:
        conn = get_db()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                (id, title, mode, audience_level, created_at, last_updated, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, generate_chat_title(transcript), mode, audience_level, now, now, None),
            )
            conn.commit()
        finally:
            conn.close()
    return session_id


def get_chat_history(session_id: str) -> dict[str, Any] | None:
    with db_lock:
        conn = get_db()
        try:
            session = row_to_dict(
                conn.execute(
                    """
                    SELECT id, title, mode, last_updated, feedback
                    FROM chat_sessions
                    WHERE id = ?
                    """,
                    (session_id,),
                ).fetchone()
            )
            if not session:
                return None

            messages = [
                {
                    "type": row["type"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                }
                for row in conn.execute(
                    """
                    SELECT type, content, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            ]
        finally:
            conn.close()

    return {
        "id": session["id"],
        "title": session.get("title"),
        "mode": session.get("mode"),
        "messages": messages,
        "last_updated": session.get("last_updated"),
        "feedback": parse_json_field(session.get("feedback"), {}),
    }


def update_chat_session(session_id: str, message: dict[str, Any], feedback: dict[str, Any] | None = None) -> None:
    now = utcnow()
    with db_lock:
        conn = get_db()
        try:
            existing = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO chat_sessions
                    (id, title, mode, audience_level, created_at, last_updated, feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, "Chat: New session...", "", "", now, now, None),
                )

            conn.execute(
                """
                INSERT INTO chat_messages (session_id, type, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    message.get("type", "user"),
                    message.get("content", ""),
                    message.get("timestamp", now),
                ),
            )

            if feedback is not None:
                conn.execute(
                    "UPDATE chat_sessions SET last_updated = ?, feedback = ? WHERE id = ?",
                    (now, json.dumps(feedback), session_id),
                )
            else:
                conn.execute("UPDATE chat_sessions SET last_updated = ? WHERE id = ?", (now, session_id))

            conn.commit()
        finally:
            conn.close()


def create_explain_session(session_id: str) -> None:
    now = utcnow()
    with db_lock:
        conn = get_db()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO explain_sessions
                (session_id, question_index, pending_questions, teacher_responses, original_text, created_at, last_updated)
                VALUES (?, 0, '[]', '[]', NULL, ?, ?)
                """,
                (session_id, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def get_explain_session(session_id: str) -> dict[str, Any] | None:
    with db_lock:
        conn = get_db()
        try:
            row = row_to_dict(
                conn.execute(
                    """
                    SELECT session_id, question_index, pending_questions, teacher_responses, original_text, created_at, last_updated
                    FROM explain_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
            )
        finally:
            conn.close()

    if not row:
        return None

    row["pending_questions"] = parse_json_field(row.get("pending_questions"), [])
    row["teacher_responses"] = parse_json_field(row.get("teacher_responses"), [])
    return row


def update_explain_session(session_id: str, update_data: dict[str, Any]) -> None:
    existing = get_explain_session(session_id)
    now = utcnow()
    payload = {
        "session_id": session_id,
        "question_index": 0,
        "pending_questions": [],
        "teacher_responses": [],
        "original_text": None,
        "created_at": now,
        "last_updated": now,
    }
    if existing:
        payload.update(existing)
    payload.update(update_data)
    payload["last_updated"] = now

    with db_lock:
        conn = get_db()
        try:
            conn.execute(
                """
                INSERT INTO explain_sessions
                (session_id, question_index, pending_questions, teacher_responses, original_text, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    question_index = excluded.question_index,
                    pending_questions = excluded.pending_questions,
                    teacher_responses = excluded.teacher_responses,
                    original_text = excluded.original_text,
                    last_updated = excluded.last_updated
                """,
                (
                    session_id,
                    payload["question_index"],
                    json.dumps(payload["pending_questions"]),
                    json.dumps(payload["teacher_responses"]),
                    payload["original_text"],
                    payload["created_at"],
                    payload["last_updated"],
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_llm_client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    kwargs: dict[str, Any] = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    return json.loads(text)


def chat_completion(messages: list[dict[str, str]], max_tokens: int) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def get_whisper_model() -> WhisperModel:
    global whisper_model
    with whisper_lock:
        if whisper_model is None:
            whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        return whisper_model


def transcribe_audio_file(path: str) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(path, language=WHISPER_LANGUAGE, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = None
frame_lock = threading.Lock()
frame_count = 0
upright_count = 0
nod_count = 0
last_nod_y = None
hand_gesture_ct = 0
last_frame = None


def camera_worker_loop() -> None:
    global cap, frame_count, upright_count, nod_count, last_nod_y, hand_gesture_ct, last_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible.")
        return

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
    )

    while True:
        success, frame = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        annotated = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(annotated, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        with frame_lock:
            frame_count += 1
            last_frame = annotated

            if results.pose_landmarks:
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                if abs(left_shoulder.y - right_shoulder.y) < 0.02:
                    upright_count += 1

                nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y
                if last_nod_y is not None and (last_nod_y - nose_y) > 0.03:
                    nod_count += 1
                last_nod_y = nose_y

            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_gesture_ct += 1

        time.sleep(1 / 30)


camera_thread_started = False
camera_thread_lock = threading.Lock()


def ensure_camera_thread_running() -> None:
    global camera_thread_started
    with camera_thread_lock:
        if not camera_thread_started:
            threading.Thread(target=camera_worker_loop, daemon=True).start()
            camera_thread_started = True


def build_presentation_prompt(audience_level: str, mode: str) -> str:
    return f"""
You are an AI presentation coach analyzing a student's transcript.

Context:
- Audience Level: {audience_level}
- Mode: {mode}

Tasks:
1. Detect filler words and hesitation patterns.
2. Identify [silence] markers as pauses.
3. Analyze structure and propose a clearer outline.
4. Give specific tips to improve pacing and delivery.
5. Generate three comprehension questions for a {audience_level} audience.
6. Suggest 1-3 transcript snippets that could be rephrased more clearly.

Return ONLY valid JSON:
{{
  "summary": "...",
  "clarity": "...",
  "pacing": "...",
  "structureSuggestions": "...",
  "deliveryTips": "...",
  "questions": ["...", "...", "..."],
  "rephrasingSuggestions": [
    {{"original": "...", "suggested": "..."}}
  ]
}}
""".strip()


def summarize_explain_flow(session_id: str, audience_level: str) -> tuple[dict[str, Any], int]:
    session_chat = get_chat_history(session_id)
    if session_chat is None:
        return {"error": "No chat session found for summary."}, 500

    all_messages = session_chat.get("messages", [])
    teacher_explanation = next(
        (
            msg["content"].strip()
            for msg in reversed(all_messages)
            if msg["type"] == "user"
            and not msg["content"].lower().startswith("summarize")
            and len(msg["content"].strip().split()) > 10
        ),
        None,
    )

    if not teacher_explanation:
        return {
            "error": "No explanation found before summarize command.",
            "message": "Please provide an explanation before summarizing.",
        }, 200

    qa_pairs: list[tuple[str, str]] = []
    asked = 0
    for index, msg in enumerate(all_messages):
        if msg["type"] == "assistant" and msg["content"].strip().endswith("?"):
            question = msg["content"].strip()
            answer = next(
                (
                    next_msg["content"].strip()
                    for next_msg in all_messages[index + 1 :]
                    if next_msg["type"] == "user"
                ),
                "",
            )
            qa_pairs.append((question, answer))
            asked += 1
            if asked >= 3:
                break

    combined_history = f"Teacher explained:\n{teacher_explanation}\n\n"
    for idx, (question, answer) in enumerate(qa_pairs, 1):
        combined_history += f"Question {idx}: {question}\nAnswer: {answer}\n\n"

    prompt = f"""
You are a {audience_level.lower()} level student summarizing the teacher's explanation.
Base the summary on both the explanation and the question-answer exchange.
Return ONLY JSON: {{"summary": "...", "keyPoints": ["...", "...", "..."]}}
""".strip()

    raw = chat_completion(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": combined_history},
        ],
        max_tokens=500,
    )
    data = extract_json_object(raw)

    update_chat_session(
        session_id,
        {"type": "assistant", "content": data["summary"], "timestamp": utcnow()},
        feedback={"questions": data.get("keyPoints", [])},
    )

    return {
        "message": data["summary"],
        "feedback": {"questions": data.get("keyPoints", [])},
        "sessionId": session_id,
    }, 200


@app.before_request
def startup_hooks() -> None:
    init_db()
    ensure_camera_thread_running()


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio_only():
    temp_files: list[str] = []
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        audio_file = request.files["audio"]
        if not audio_file.filename:
            return jsonify({"error": "Empty filename."}), 400

        raw_bytes = audio_file.read()
        if not raw_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        audio_seg = AudioSegment.from_file(BytesIO(raw_bytes))
        chunks = split_on_silence(
            audio_seg,
            min_silence_len=500,
            silence_thresh=audio_seg.dBFS - 16,
            keep_silence=250,
        )

        if not chunks:
            chunks = [audio_seg]

        pieces: list[str] = []
        for index, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            temp_files.append(temp_path)
            chunk.export(temp_path, format="wav")
            text = transcribe_audio_file(temp_path)
            if text:
                pieces.append(text)
            if index < len(chunks) - 1:
                pieces.append("[silence]")

        transcript = " ".join(pieces).strip()
        if not transcript:
            return jsonify({"error": "No speech detected. Please speak clearly."}), 400

        return jsonify({"transcript": transcript})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.route("/api/analyze", methods=["POST"])
def analyze_audio():
    try:
        payload = request.get_json() if request.is_json else {}
        session_id = payload.get("sessionId")
        audience_level = payload.get("audienceLevel", "Beginner")
        mode = payload.get("mode", "Presentation")
        final_transcript = payload.get("message", "")

        if not session_id:
            return jsonify({"error": "Missing sessionId"}), 400

        session_chat = get_chat_history(session_id)
        if not session_chat:
            create_chat_session(
                transcript=final_transcript,
                mode=mode,
                audience_level=audience_level,
                session_id=session_id,
            )

        update_chat_session(
            session_id,
            {"type": "user", "content": final_transcript, "timestamp": utcnow()},
        )

        if mode == "Explain":
            text = final_transcript.strip()

            if payload.get("summarize"):
                body, status = summarize_explain_flow(session_id, audience_level)
                return jsonify(body), status

            explain_session = get_explain_session(session_id)
            if not explain_session:
                create_explain_session(session_id)
                explain_session = get_explain_session(session_id)

            explain_session = explain_session or {}
            pending = explain_session.get("pending_questions", [])
            current_q = explain_session.get("question_index", 0)
            teacher_responses = explain_session.get("teacher_responses", [])

            if not pending:
                update_explain_session(session_id, {"original_text": text})
                prompt = f"""
You are a curious student with {audience_level.lower()} level knowledge.
After hearing the teacher's explanation, ask exactly 3 relevant follow-up questions.
Return ONLY JSON: {{"questions": ["q1", "q2", "q3"]}}
""".strip()

                raw = chat_completion(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Teacher says:\n\n{text}"},
                    ],
                    max_tokens=300,
                )
                questions = (extract_json_object(raw).get("questions") or [])[:3]
                update_explain_session(
                    session_id,
                    {
                        "pending_questions": questions,
                        "teacher_responses": [],
                        "question_index": 0,
                    },
                )

                first_question = questions[0] if questions else "Can you explain that in a bit more detail?"
                update_chat_session(
                    session_id,
                    {"type": "assistant", "content": first_question, "timestamp": utcnow()},
                )
                return jsonify(
                    {
                        "message": first_question,
                        "feedback": {"questions": [first_question]},
                        "sessionId": session_id,
                    }
                )

            teacher_responses = teacher_responses + [text]
            update_explain_session(session_id, {"teacher_responses": teacher_responses})

            if current_q + 1 < len(pending):
                next_q = pending[current_q + 1]
                update_explain_session(session_id, {"question_index": current_q + 1})
                update_chat_session(
                    session_id,
                    {"type": "assistant", "content": next_q, "timestamp": utcnow()},
                )
                return jsonify(
                    {
                        "message": next_q,
                        "feedback": {"questions": [next_q]},
                        "sessionId": session_id,
                    }
                )

            thank_you_message = (
                "Thank you for answering all three questions!\n"
                "When you're ready for the final summary, please type summarize."
            )
            update_explain_session(session_id, {"question_index": current_q + 1})
            update_chat_session(
                session_id,
                {"type": "assistant", "content": thank_you_message, "timestamp": utcnow()},
            )
            return jsonify({"message": thank_you_message, "sessionId": session_id})

        if mode == "Presentation":
            raw = chat_completion(
                [
                    {"role": "system", "content": build_presentation_prompt(audience_level, mode)},
                    {"role": "user", "content": f"Transcript:\n\n{final_transcript}"},
                ],
                max_tokens=1000,
            )
            feedback_json = extract_json_object(raw)

            update_chat_session(
                session_id=session_id,
                message={
                    "type": "assistant",
                    "content": feedback_json.get("summary", ""),
                    "timestamp": utcnow(),
                },
                feedback={
                    "clarity": feedback_json.get("clarity", ""),
                    "pacing": feedback_json.get("pacing", ""),
                    "structureSuggestions": feedback_json.get("structureSuggestions", ""),
                    "deliveryTips": feedback_json.get("deliveryTips", ""),
                    "questions": feedback_json.get("questions", []),
                    "rephrasingSuggestions": feedback_json.get("rephrasingSuggestions", []),
                },
            )

            return jsonify(
                {
                    "message": feedback_json.get("summary", ""),
                    "feedback": {
                        "clarity": feedback_json.get("clarity", ""),
                        "pacing": feedback_json.get("pacing", ""),
                        "structureSuggestions": [feedback_json.get("structureSuggestions", "")],
                        "deliveryTips": [feedback_json.get("deliveryTips", "")],
                        "questions": feedback_json.get("questions", []),
                        "rephrasingSuggestions": feedback_json.get("rephrasingSuggestions", []),
                    },
                }
            )

        return jsonify({"error": f"Unknown mode {mode}"}), 400
    except Exception as exc:
        print("=" * 30)
        print("Caught final exception in analyze_audio")
        print("Exception:", exc)
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error in /api/analyze")
        print("=" * 30)
        return jsonify({"error": "Internal Server Error", "details": str(exc)}), 500


@app.route("/api/bodytrack")
def bodytrack():
    def gen_frames():
        global last_frame
        while True:
            with frame_lock:
                frame = last_frame.copy() if last_frame is not None else None
            if frame is None:
                continue
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/bodymetrics")
def bodymetrics():
    global frame_count, upright_count, nod_count, hand_gesture_ct
    with frame_lock:
        fc = frame_count or 1
        up = upright_count
        nd = nod_count
        hg = hand_gesture_ct
        frame_count = 0
        upright_count = 0
        nod_count = 0
        hand_gesture_ct = 0

    posture_score = int((up / fc) * 100)
    gestures_per_min = int((hg / fc) * 30 * 60)
    nods_per_min = int((nd / fc) * 30 * 60)

    return jsonify(
        {
            "postureScore": posture_score,
            "handGestureRate": gestures_per_min,
            "headNodCount": nods_per_min,
            "suggestions": [
                "Keep your shoulders level to appear more confident.",
                "Use deliberate hand gestures and avoid constant motion.",
                "Avoid excessive head nodding so the audience stays focused on your content.",
            ],
        }
    )


@app.route("/api/chats", methods=["GET"])
def list_chat_sessions():
    with db_lock:
        conn = get_db()
        try:
            items = conn.execute(
                """
                SELECT id, title, mode, created_at, last_updated
                FROM chat_sessions
                ORDER BY last_updated DESC
                LIMIT 20
                """
            ).fetchall()
        finally:
            conn.close()

    return jsonify(
        [
            {
                "id": item["id"],
                "sessionId": item["id"],
                "title": item["title"],
                "mode": item["mode"],
                "created": item["created_at"],
                "updated": item["last_updated"],
            }
            for item in items
        ]
    )


@app.route("/api/chats/<session_id>", methods=["GET"])
def get_chat_session(session_id: str):
    session = get_chat_history(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)


@app.route("/api/chats/<session_id>", methods=["DELETE"])
def delete_chat_session(session_id: str):
    with db_lock:
        conn = get_db()
        try:
            deleted = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM explain_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    if deleted.rowcount == 0:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"success": True})


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    user_id = str(uuid.uuid4())
    now = utcnow()
    token = jwt.encode({"sub": user_id, "email": email}, JWT_SECRET, algorithm=JWT_ALGORITHM)

    with db_lock:
        conn = get_db()
        try:
            existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                return jsonify({"message": "User already exists"}), 400

            conn.execute(
                """
                INSERT INTO users (id, email, password, token, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, generate_password_hash(password), token, now),
            )
            conn.commit()
        finally:
            conn.close()

    return jsonify({"message": "Signup successful!", "token": token}), 200


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    with db_lock:
        conn = get_db()
        try:
            user = row_to_dict(
                conn.execute(
                    "SELECT id, email, password, token FROM users WHERE email = ?",
                    (email,),
                ).fetchone()
            )
            if not user or not check_password_hash(user["password"], password):
                return jsonify({"message": "Invalid email or password"}), 401

            token = user.get("token")
            if not token:
                token = jwt.encode(
                    {"sub": user["id"], "email": user["email"]},
                    JWT_SECRET,
                    algorithm=JWT_ALGORITHM,
                )
                conn.execute("UPDATE users SET token = ? WHERE id = ?", (token, user["id"]))
                conn.commit()
        finally:
            conn.close()

    return jsonify({"message": "Login successful!", "token": token}), 200


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    target = Path(app.static_folder) / path
    if path and target.exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    init_db()
    ensure_camera_thread_running()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=False)
