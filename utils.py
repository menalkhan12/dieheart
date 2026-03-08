import json
import os
import re
from datetime import datetime

CALL_RECORD_FILE = "logs/call_records.json"
LEAD_LOG_FILE = "logs/lead_logs.txt"

def init_call_record(session_id):
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(CALL_RECORD_FILE):
        with open(CALL_RECORD_FILE, "w") as f:
            json.dump({}, f)

    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    data[session_id] = {
        "start_time": str(datetime.now()),
        "turns": [],
        "escalated": False,
        "phone": None
    }

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_call_record(session_id, user, agent, escalated=False, phone=None):
    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    data[session_id]["turns"].append({
        "user": user,
        "agent": agent
    })

    if escalated:
        data[session_id]["escalated"] = True

    if phone:
        data[session_id]["phone"] = phone

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def end_call_record(session_id):
    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    data[session_id]["end_time"] = str(datetime.now())

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def append_lead_log(session_id, phone, unanswered_query):
    """Store lead with call_id, phone, the query that wasn't answered, and timestamp."""
    with open(LEAD_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | call_id={session_id} | phone={phone} | unanswered_query={unanswered_query}\n")

def get_last_user_query(session_id):
    """Get the last user query from call record (the question we couldn't answer before they gave phone)."""
    try:
        if not os.path.exists(CALL_RECORD_FILE):
            return None
        with open(CALL_RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if session_id not in data or not data[session_id].get("turns"):
            return None
        return data[session_id]["turns"][-1]["user"]
    except Exception:
        return None


def get_recent_turns(session_id, n=8):
    """Get last n turns (user/agent pairs) for conversation continuity. Returns list of (user, agent) tuples."""
    try:
        if not os.path.exists(CALL_RECORD_FILE):
            return []
        with open(CALL_RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if session_id not in data or not data[session_id].get("turns"):
            return []
        turns = data[session_id]["turns"][-n:]
        return [(t["user"], t["agent"]) for t in turns]
    except Exception:
        return []

def detect_phone_number(text):
    match = re.search(r"(03\d{9})", text)
    return match.group(1) if match else None