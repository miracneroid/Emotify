# action_mapping.py

ACTION_TO_EMOTION = {
    "dancing": "happy",
    "crying": "sad",
    "fighting": "angry",
    "fighting / punching stance": "angry",
    "hiding": "fear",
    "defensive / cowering": "fear",
    "jumping with joy": "happy",
    "arms raised / celebrating": "happy",
    "victory / raising hands": "happy",
    "waving / raised hand": "happy",
    "sitting quietly": "neutral",
    "sitting / resting": "neutral",
    "standing / idle": "neutral",
    "surprised reaction": "surprise",
    "hands on head / distressed": "fear",
    "playing an instrument": "neutral",
    "running away": "fear",
    "walking / running": "neutral",
    "shouting": "angry",
    "laughing": "happy",
    "walking slowly": "neutral",
    "playing sports": "happy",
    "working out": "neutral",
}

def get_emotion_for_action(action):
    return ACTION_TO_EMOTION.get(action.lower(), "neutral")
