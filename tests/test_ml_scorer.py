import sys
import os
import unittest
# Use mock or try-except for sklearn in test environment if needed
try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hallucination.ml_scorer import EnsembleHallucinationScorer

class TestMLScorer(unittest.TestCase):
    def test_initialization(self):
        if LogisticRegression is None:
            self.skipTest("scikit-learn not installed")
        scorer = EnsembleHallucinationScorer()
        self.assertTrue(scorer.is_fitted)
        
    def test_prediction_high_hallucination(self):
        if LogisticRegression is None:
            self.skipTest("scikit-learn not installed")
        scorer = EnsembleHallucinationScorer()
        # High uncertainty, RAG failure, Fact failure, Low Consistency
        score = scorer.predict_score(0.9, 1.0, 1.0, 0.1)
        print(f"High Hallucination Score: {score}")
        self.assertGreater(score, 0.8)

    def test_prediction_low_hallucination(self):
        if LogisticRegression is None:
            self.skipTest("scikit-learn not installed")
        scorer = EnsembleHallucinationScorer()
        # Low uncertainty, No RAG failure, No Fact failure, High Consistency
        score = scorer.predict_score(0.1, 0.0, 0.0, 0.95)
        print(f"Low Hallucination Score: {score}")
        self.assertLess(score, 0.3)

if __name__ == "__main__":
    unittest.main()
