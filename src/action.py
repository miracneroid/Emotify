import action_mapping

def map_actions_to_emotions(actions):
    """Maps a list of actions to their corresponding emotions."""
    action_emotion_pairs = {}

    for action in actions:
        emotion = action_mapping.get_emotion_for_action(action)
        action_emotion_pairs[action] = emotion

    return action_emotion_pairs

# Example Usage
if __name__ == "__main__":
    sample_actions = ["dancing", "crying", "jumping with joy", "fighting", "laughing"]
    mapped_results = map_actions_to_emotions(sample_actions)

    for action, emotion in mapped_results.items():
        print(f"Action: {action} → Emotion: {emotion}")
