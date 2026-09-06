import unittest
from action_classifier import PoseActionClassifier

class DummyLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class TestPoseActionClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = PoseActionClassifier()

    def test_empty_landmarks(self):
        result = self.classifier.classify([])
        self.assertEqual(result["action"], "No Pose Detected")

    def test_standing_pose(self):
        landmarks = [DummyLandmark(0.5, 0.5) for _ in range(33)]
        # Nose
        landmarks[0] = DummyLandmark(0.5, 0.2)
        # Shoulders
        landmarks[11] = DummyLandmark(0.4, 0.4)
        landmarks[12] = DummyLandmark(0.6, 0.4)
        # Wrists below shoulders
        landmarks[15] = DummyLandmark(0.4, 0.6)
        landmarks[16] = DummyLandmark(0.6, 0.6)
        # Hips & Knees & Ankles
        landmarks[23] = DummyLandmark(0.45, 0.7)
        landmarks[24] = DummyLandmark(0.55, 0.7)
        landmarks[25] = DummyLandmark(0.45, 0.85)
        landmarks[26] = DummyLandmark(0.55, 0.85)
        landmarks[27] = DummyLandmark(0.45, 0.98)
        landmarks[28] = DummyLandmark(0.55, 0.98)

        result = self.classifier.classify(landmarks)
        self.assertIn(result["action"], ["Standing / Idle", "Walking / Running"])

if __name__ == "__main__":
    unittest.main()
