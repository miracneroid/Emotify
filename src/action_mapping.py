# action_mapping.py

# Define a dictionary to map actions to emotions
ACTION_TO_EMOTION = {
    "dancing": "happy",
    "crying": "sad",
    "fighting": "angry",
    "hiding": "fear",
    "jumping with joy": "happy",
    "sitting quietly": "neutral",
    "surprised reaction": "surprise",
    "playing an instrument": "neutral",
    "running away": "fear",
    "shouting": "angry",
    "laughing": "happy",
    "walking slowly": "neutral",
    "playing sports": "happy",
    "working out": "neutral",
}

def get_emotion_for_action(action):
    """Returns the emotion mapped to an action. If not found, return 'neutral'."""
    return ACTION_TO_EMOTION.get(action.lower(), "neutral")
