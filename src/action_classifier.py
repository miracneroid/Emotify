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

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates the angle (in degrees) between three points (a, b, c) where b is the vertex.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return float(angle)

    def classify(self, landmarks, frame_shape=None):
        if not landmarks:
            return {"action": "No Pose Detected", "confidence": 0.0, "details": {}}

        def get_pt(idx):
            lm = landmarks[idx]
            return (lm.x, lm.y)

        l_shoulder = get_pt(self.LEFT_SHOULDER)
        r_shoulder = get_pt(self.RIGHT_SHOULDER)
        l_elbow = get_pt(self.LEFT_ELBOW)
        r_elbow = get_pt(self.RIGHT_ELBOW)
        l_wrist = get_pt(self.LEFT_WRIST)
        r_wrist = get_pt(self.RIGHT_WRIST)
        l_hip = get_pt(self.LEFT_HIP)
        r_hip = get_pt(self.RIGHT_HIP)
        l_knee = get_pt(self.LEFT_KNEE)
        r_knee = get_pt(self.RIGHT_KNEE)
        l_ankle = get_pt(self.LEFT_ANKLE)
        r_ankle = get_pt(self.RIGHT_ANKLE)
        nose = get_pt(self.NOSE)

        l_arm_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_arm_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        l_hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = self.calculate_angle(r_shoulder, r_hip, r_knee)

        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
        knee_mid_y = (l_knee[1] + r_knee[1]) / 2.0
        wrist_mid_y = (l_wrist[1] + r_wrist[1]) / 2.0

        movement_velocity = 0.0
        if self.prev_landmarks is not None:
            prev_nose = (self.prev_landmarks[self.NOSE].x, self.prev_landmarks[self.NOSE].y)
            prev_l_wrist = (self.prev_landmarks[self.LEFT_WRIST].x, self.prev_landmarks[self.LEFT_WRIST].y)
            movement_velocity = math.hypot(nose[0] - prev_nose[0], nose[1] - prev_nose[1]) + \
                                math.hypot(l_wrist[0] - prev_l_wrist[0], l_wrist[1] - prev_l_wrist[1])

        self.prev_landmarks = landmarks

        action = "Standing / Idle"
        confidence = 0.70

        if l_wrist[1] < shoulder_mid_y and r_wrist[1] < shoulder_mid_y:
            if l_wrist[1] < nose[1] or r_wrist[1] < nose[1]:
                action = "Arms Raised / Celebrating"
                confidence = 0.92
            else:
                action = "Victory / Raising Hands"
                confidence = 0.85

        if movement_velocity > 0.05 or abs(l_knee_angle - r_knee_angle) > 25:
            action = "Walking / Running"
            confidence = 0.80

        details = {
            "l_arm_angle": round(l_arm_angle, 1),
            "r_arm_angle": round(r_arm_angle, 1),
            "l_knee_angle": round(l_knee_angle, 1),
            "r_knee_angle": round(r_knee_angle, 1),
            "l_hip_angle": round(l_hip_angle, 1),
            "r_hip_angle": round(r_hip_angle, 1),
            "velocity": round(movement_velocity, 4)
        }

        return {"action": action, "confidence": confidence, "details": details}
