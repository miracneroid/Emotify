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

EMOTION_TO_ACTIONS = {
    "happy": ["dancing", "jumping with joy", "arms raised / celebrating", "waving / raised hand", "laughing", "playing sports"],
    "sad": ["crying", "hands on head / distressed", "sitting quietly", "walking slowly"],
    "angry": ["fighting", "fighting / punching stance", "shouting"],
    "fear": ["hiding", "defensive / cowering", "hands on head / distressed", "running away"],
    "surprise": ["surprised reaction", "victory / raising hands"],
    "neutral": ["standing / idle", "sitting / resting", "walking / running", "working out"],
    "disgust": ["defensive / cowering", "sitting quietly"]
}

def get_actions_for_emotion(emotion):
    return EMOTION_TO_ACTIONS.get(emotion.lower(), ["standing / idle"])
