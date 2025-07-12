import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hallucination.text_utils import extract_claims, split_into_sentences
from hallucination.detector import HallucinationDetector
from hallucination.metrics import HallucinationMetrics

def test_text_utils():
    text = "The sky is blue. The grass is green!"
    claims = extract_claims(text)
    assert len(claims) == 2
    assert "The sky is blue" in claims[0]
    print("Text utils test passed.")

def test_metrics():
    metrics = HallucinationMetrics()
    metrics.log_response("1", 0.8, 0.0) # Uncertain
    metrics.log_response("2", 0.0, 0.9) # Hallucinated (RAG)
    metrics.log_response("3", 0.0, 0.0) # Clean
    rate = metrics.get_hallucination_rate()
    assert abs(rate - 0.66) < 0.1
    print("Metrics test passed.")

def test_detector_mock():
    detector = HallucinationDetector()
    prompt = "Who are you?"
    resp = "I am AI."
    
    # Mock sampling function
    def mock_sample(p, n):
        return ["I am AI.", "I am a robot.", "I am code."]
        
    score = detector.detect_uncertainty(prompt, resp, mock_sample)
    print(f"Detector Score: {score}")
    assert score >= 0.0 and score <= 1.0
    print("Detector test passed.")

if __name__ == "__main__":
    test_text_utils()
    test_metrics()
    test_detector_mock()
    print("All basic tests passed.")
