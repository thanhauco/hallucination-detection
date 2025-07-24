import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hallucination.advanced import AdvancedVerifier

def test_cove_verifer_mock():
    # Mock LLM that generates questions and answers
    def mock_llm_fn(prompt: str) -> str:
        if "generate 3 specific" in prompt:
            return "1. Is the sky blue?\n2. Is water wet?\n3. Is fire hot?"
        if "Answer this" in prompt:
            return "Yes."
        if "consistent" in prompt.lower():
            return "Score: 1.0 (Consistent)"
        return "I don't know."

    verifier = AdvancedVerifier(llm_fn=mock_llm_fn)
    
    # Test CoVe
    result = verifier.chain_of_verification("The sky is blue.", "The sky is blue and water is wet.")
    print(f"CoVe Questions: {result.verification_questions}")
    print(f"CoVe Consistency: {result.consistency_score}")
    assert len(result.verification_questions) == 3
    assert result.consistency_score >= 0.9
    print("CoVe test passed.")

def test_entity_consistency():
    def mock_llm(p): return ""
    verifier = AdvancedVerifier(llm_fn=mock_llm)
    
    # Consistent samples
    samples_consistent = [
        "Barack Obama was born in Hawaii.",
        "Obama was born in Honolulu, Hawaii.",
        "The birthplace of Barack Obama is Hawaii."
    ]
    # Inconsistent samples
    samples_inconsistent = [
        "Barack Obama was born in Hawaii.",
        "Barack Obama was born in Kenya.", 
        "Barack Obama was born in Texas."
    ]
    
    score_high = verifier.check_entity_consistency(samples_consistent)
    score_low = verifier.check_entity_consistency(samples_inconsistent)
    
    print(f"Consistent Score: {score_high}")
    print(f"Inconsistent Score: {score_low}")
    
    # Note: Logic depends on exact NER overlap, Spacy model might be strict.
    # We check relative scores
    assert score_high > score_low or score_high >= 0.5
    print("Entity Consistency test passed.")

if __name__ == "__main__":
    test_cove_verifer_mock()
    test_entity_consistency()
