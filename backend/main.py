# Import necessary modules from FastAPI and other libraries
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
from pymavlink import mavutil
from typing import Dict
import tempfile
import os
import time
import json

# Import custom processing function
from process.process import run_process

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware to allow requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store count of message types per session
session_message: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
# Dictionary to store chat history per session
session_histories = defaultdict(list)

# Endpoint to upload .bin log files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(None)):
    session_id = session_id or "default_session"  # Use default if no session_id provided
    filename = f"{session_id}.bin"
    save_path = os.path.join("uploads", filename)

    # Save the uploaded file to the local filesystem
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Parse the MAVLink log file and update in-memory message type counts
    try:
        mavlog = mavutil.mavlink_connection(save_path, robust_parsing=True)

        while True:
            msg = mavlog.recv_match(blocking=False)
            if msg is None:
                break
            msg_type = msg.get_type()
            session_message[session_id][msg_type] += 1  # Increment count of message type
    except Exception as e:
        return {"error": f"Failed to parse MAVLink log: {str(e)}"}

    # Output the message type counts to the console for debugging
    print(session_message[session_id])
    return {"message": "File uploaded successfully", "filename": filename}

# WebSocket endpoint for real-time chat interaction
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept incoming WebSocket connection
    try:
        while True:
            # Receive message from client and parse JSON
            data = json.loads(await websocket.receive_text())
            session_id = data.get("sessionId")
            user_msg = data.get("message", "")

            # Handle case where session_id is missing
            if not session_id:
                reply = "Please upload .bin file before chatting."
            else:
                # Append user's message to the session's history
                session_histories[session_id].append({"role": "user", "content": user_msg})

                # Generate a reply using the custom process logic
                reply = await run_process(session_id, session_histories[session_id], session_message[session_id])
                print("reply type:", type(reply))  # Debugging info

                # Append bot's reply to the session's history
                session_histories[session_id].append({"role": "bot", "content": reply})

            # Send the bot's reply back to the client
            await websocket.send_text(reply)
            print(session_histories[session_id])  # Log conversation history for debugging

    except WebSocketDisconnect:
        # Log disconnection event
        print("Client disconnected")
