# UAV Log Viewer & Chatbot Interface

This project provides an interactive system for uploading and analyzing UAV flight log files using a FastAPI backend and a web-based chatbot frontend. The chatbot understands UAV telemetry and can help answer questions about flight data using `.bin` log files.

---

## 🚀 Getting Started

### 1. Navigate to the Backend Directory

```bash
cd backend
```

### 2. Run the Application

Add your OPENAI_API_KEY to `run.sh`

Execute the `run.sh` script to launch the FastAPI server:

```bash
bash run.sh
```

This will start the backend server required to process UAV logs and handle chatbot queries.

---

## 💬 Using the Chatbot UI

Once the backend is running, open your browser and go to:

```
http://localhost:8080
```

Here, you'll find the **UAV Log Viewer UI**, where you can:

* Upload `.bin` files from UAV systems (e.g., ArduPilot)
* Chat with the assistant to get summaries, telemetry insights, and analytical responses

---

## 📁 Project Structure Overview

```
backend/
├── run.sh                      # Script to start FastAPI server
├── main.py                     # FastAPI entrypoint
├── process/
│   └── uav_query_processor.py  # Core chatbot and log processing logic
├── uploads/                    # Directory for uploaded UAV log files
```

---
