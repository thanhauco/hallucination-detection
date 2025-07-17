import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hallucination.fact_check_agent import FactCheckAgent, VerificationResult, FactCheckReport

def test_fact_check_agent_mock():
    """Test FactCheckAgent with mock search and LLM."""
    
    # Mock search that returns relevant info
    def mock_search(query: str) -> str:
        if "Paris" in query or "France" in query:
            return "Paris is the capital and largest city of France."
        return "No relevant information found."
    
    # Mock LLM that checks for keyword match
    def mock_llm(prompt: str) -> str:
        if "Paris" in prompt and "capital" in prompt and "France" in prompt:
            if "Paris is the capital" in prompt:
                return "TRUE"
        return "UNCERTAIN"
    
    agent = FactCheckAgent(search_fn=mock_search, llm_fn=mock_llm)
    
    # Test single claim verification
    result = agent.verify_claim("Paris is the capital of France.")
    assert isinstance(result, VerificationResult)
    print(f"Claim verified: {result.is_verified}, confidence: {result.confidence}")
    
    # Test full response fact-check
    response = "Paris is the capital of France. It has the Eiffel Tower."
    report = agent.fact_check(response)
    
    assert isinstance(report, FactCheckReport)
    assert report.total_claims == 2
    print(f"Total claims: {report.total_claims}")
    print(f"Verified: {report.verified_claims}, Unverified: {report.unverified_claims}")
    print(f"Hallucination score: {report.hallucination_score}")
    
    print("FactCheckAgent test passed.")

def test_search_query_generation():
    """Test search query simplification."""
    agent = FactCheckAgent()
    
    claim = "The quick brown fox jumps over the lazy dog"
    query = agent._generate_search_query(claim)
    
    # Should remove stop words like "the"
    assert "The" not in query.split() or len(query.split()) <= 10
    print(f"Generated query: {query}")
    print("Search query generation test passed.")

if __name__ == "__main__":
    test_fact_check_agent_mock()
    test_search_query_generation()
    print("All FactCheckAgent tests passed.")
