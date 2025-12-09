from app.core.memory import get_history
from fastapi import Depends

def get_session_history(session_id: str):
    return get_history(session_id)
