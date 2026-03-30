# 🚀 AI Reverse Learning

AI Reverse Learning helps you practice explanations and presentations with an AI audience that adapts to **beginner, intermediate, or expert listeners**. It simulates real audience behavior and provides intelligent feedback on your delivery, clarity, and engagement.

---

# 🧠 Core Features

- 💬 **ChatGPT-style interface** — clean and intuitive UI
- 🎯 **Audience Level Selection** — Beginner / Intermediate / Expert
- 🧑‍🏫 **Two Modes**
  - **Explain Mode** → teach concepts and get feedback
  - **Presentation Mode** → practice delivery and communication
- 🎤 **Voice Input** — speak naturally, get transcription + analysis
- 📊 **Detailed Feedback** — clarity, pacing, confidence insights
- ❓ **AI-generated Questions** — audience-specific questioning
- 🧍 **Body Language Tracking** — posture & gesture analysis using MediaPipe

---

# 🏗️ Tech Stack

## Frontend

- Vite
- React
- TypeScript
- Tailwind CSS

## Backend

- Flask

## AI & Processing

- OpenAI-compatible chat completions
- faster-whisper (speech-to-text)
- OpenCV + MediaPipe (body language tracking)

## Storage

- SQLite (local persistence)

---

# ⚙️ Prerequisites

- Node.js 18+
- Python 3.10+
- ffmpeg installed and available in PATH  
  👉 https://www.ffmpeg.org/download.html
- OpenAI API key (or compatible provider)

---

## Quick Start On Windows

Use the bootstrap script:

```powershell
.\setup_windows.ps1
```

That script:

- creates `.env` from `.env.example` if needed
- installs Python dependencies
- installs frontend dependencies

Then open two terminals.

Backend:

```powershell
python app.py
```

Frontend:

```powershell
npm run dev
```

The frontend runs on `http://localhost:5173` and talks to the backend at `http://localhost:5000`.

Or launch both with:

```powershell
.\run_local.ps1
```

## Manual Setup

1. Copy the env template:

```powershell
Copy-Item .env.example .env
```

2. Set at least:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
JWT_SECRET=replace-this-in-production
```

3. Install backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Install frontend dependencies:

```powershell
npm install
```

5. Start the app:

```powershell
python app.py
```

```powershell
npm run dev
```

## Environment Variables

Important backend settings live in [.env.example].

- `OPENAI_API_KEY`: required for AI feedback and question generation
- `OPENAI_MODEL`: chat model name, default `gpt-4o-mini`
- `OPENAI_BASE_URL`: optional for OpenRouter, local gateways, or other compatible providers
- `DATABASE_PATH`: SQLite file path, default `app_data.db`
- `JWT_SECRET`: used for signup/login tokens
- `WHISPER_MODEL_SIZE`: `tiny`, `base`, `small`, etc.
- `WHISPER_DEVICE`: `cpu` by default
- `WHISPER_COMPUTE_TYPE`: `int8` by default
- `WHISPER_LANGUAGE`: optional fixed language code

## Current API Surface

The frontend still uses the same backend routes:

- `/api/signup`
- `/api/login`
- `/api/transcribe`
- `/api/analyze`
- `/api/chats`
- `/api/chats/:sessionId`
- `/api/bodytrack`
- `/api/bodymetrics`

## Notes

- The first Whisper transcription may take longer because the model may need to download locally.
- If `ffmpeg` is missing, audio transcription can fail or behave inconsistently.
  Install it from https://www.ffmpeg.org/download.html and make sure it is available on your `PATH`.
- SQLite is fine for local development and demos. For multi-user deployment, move persistence to Postgres or another managed database.

## Future Enhancements

- Audio clip upload and analysis
- Session history and progress tracking
- More detailed analytics on teaching performance
- Custom audience personas
