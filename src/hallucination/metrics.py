from typing import Dict, List
import numpy as np

class HallucinationMetrics:
    def __init__(self):
        self.history = []

    def log_response(self, response_id: str, uncertainty_score: float, rag_score: float):
        self.history.append({
            "id": response_id,
            "uncertainty": uncertainty_score,
            "rag_verification": rag_score
        })

    def get_hallucination_rate(self) -> float:
        """
        Calculate the aggregate hallucination rate.
        We can define a threshold, e.g., if any score > 0.5, it counts as a hallucination.
        """
        if not self.history:
            return 0.0
        
        count_hallucinated = 0
        for record in self.history:
            # Composite score or logic
            is_hallucinated = (record["uncertainty"] > 0.5) or (record["rag_verification"] > 0.3)
            if is_hallucinated:
                count_hallucinated += 1
                
        return count_hallucinated / len(self.history)
