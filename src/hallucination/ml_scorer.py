import numpy as np
from typing import Dict, List, Optional
import pickle
import os

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import NotFittedError
except ImportError:
    # Fallback to prevent crash if sklearn not installed yet
    LogisticRegression = None

class EnsembleHallucinationScorer:
    """
    Advanced ML Scorer that combines multiple weak signals (features)
    into a calibrated probability of hallucination.
    
    Features:
    1. Uncertainty Score (Semantic Entropy)
    2. RAG Verification Score (NLI Entailment)
    3. Fact Check Score (Agentic Verification)
    4. CoVe Consistency Score
    """
    def __init__(self, model_path: Optional[str] = None):
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for EnsembleHallucinationScorer")
            
        self.model = LogisticRegression(class_weight='balanced')
        self.is_fitted = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with heuristic weights if no model provided
            # Logic: RAG failure (feature 1) and Fact failure (feature 2) are high signal
            # Uncertainty (feature 0) is medium signal
            # CoVe (feature 3) is medium signal
            
            # Using a simplified mock "fitting" for MVP
            # X = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]] ...
            self._mock_train()

    def _mock_train(self):
        """Train on synthetic data to establish baseline weights."""
        # Feature vector: [uncertainty, rag_fail, fact_fail, cove_score]
        # Note: cove_score is "consistency", so 1.0 is good. Others are "error rates"
        
        # Synthetic True Positives (Hallucinations)
        X_pos = [
            [0.8, 0.9, 1.0, 0.2], # High all errors
            [0.7, 0.1, 1.0, 0.5], # Fact check failed
            [0.2, 0.8, 0.0, 0.3], # RAG failed + CoVe low
        ] * 10
        
        # Synthetic Negatives (Faithful)
        X_neg = [
            [0.1, 0.0, 0.0, 0.9], # Perfect
            [0.3, 0.1, 0.0, 0.8], # Slight uncertainty
            [0.2, 0.0, 0.1, 0.9], # Slight fact error noise
        ] * 10
        
        X = np.array(X_pos + X_neg)
        y = np.array([1]*len(X_pos) + [0]*len(X_neg))
        
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_score(self, 
                     uncertainty: float, 
                     rag_fail_rate: float, 
                     fact_fail_rate: float,
                     cove_score: float) -> float:
        """
        Returns probability of hallucination (0.0 to 1.0).
        """
        if not self.is_fitted:
            return 0.5
            
        features = np.array([[uncertainty, rag_fail_rate, fact_fail_rate, cove_score]])
        prob = self.model.predict_proba(features)[0][1]
        return float(prob)

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
