import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), 'sync_state.json')

class SyncStateManager:
    @staticmethod
    def _load_state() -> Dict[str, str]:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("sync_state.json is corrupted. Starting fresh.")
            return {}

    @staticmethod
    def _save_state(state: Dict[str, str]):
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    @staticmethod
    def get_history_id(user_email: str) -> Optional[str]:
        state = SyncStateManager._load_state()
        return state.get(user_email)

    @staticmethod
    def set_history_id(user_email: str, history_id: str):
        state = SyncStateManager._load_state()
        state[user_email] = str(history_id)
        SyncStateManager._save_state(state)
