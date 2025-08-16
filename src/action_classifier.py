import math
import numpy as np

class PoseActionClassifier:
    """
    PoseActionClassifier extracts joint angles and spatial relationships
    from MediaPipe Pose landmarks to classify human actions in real time.
    """

    def __init__(self):
        # MediaPipe Pose Landmark Indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

        self.prev_landmarks = None
