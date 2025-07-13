"""
FactCheckAgent - Agentic Verification using External Tools (Phase 3)

This module implements an agent that verifies claims against external knowledge
sources like web search APIs. It uses a tool-calling pattern to fetch real-world
information and compare it against generated claims.
"""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from .text_utils import extract_claims

@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: str
    is_verified: bool
    confidence: float  # 0.0 to 1.0
    evidence: str
    source: Optional[str] = None

@dataclass
class FactCheckReport:
    """Full report for a response."""
    response: str
    total_claims: int
    verified_claims: int
    unverified_claims: int
    hallucination_score: float  # 0.0 = all verified, 1.0 = all unverified
    results: List[VerificationResult]

class FactCheckAgent:
    """
    An agent that verifies factual claims using external tools.
    
    Supports pluggable search backends:
    - Web search (DuckDuckGo, Google, Brave)
    - Wikipedia API
    - Custom knowledge base
    """
    
    def __init__(self, 
                 search_fn: Optional[Callable[[str], str]] = None,
                 llm_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            search_fn: A function that takes a query and returns search results as text.
            llm_fn: A function that takes a prompt and returns LLM response (for reasoning).
        """
        self.search_fn = search_fn or self._default_search
        self.llm_fn = llm_fn or self._default_llm
    
    def _default_search(self, query: str) -> str:
        """Default mock search for testing. Replace with real API in production."""
        # In production, integrate with:
        # - duckduckgo-search (pip install duckduckgo-search)
        # - google-api-python-client
        # - Wikipedia API
        return f"[Mock Search Result for: {query}] No external search configured."
    
    def _default_llm(self, prompt: str) -> str:
        """Default mock LLM for testing. Replace with OpenAI/Anthropic in production."""
        return "UNCERTAIN"
    
    def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify a single claim against external sources.
        
        Steps:
        1. Generate a search query from the claim
        2. Fetch evidence from external source
        3. Use LLM to compare claim vs evidence
        4. Return verification result
        """
        # Step 1: Generate search query (could be more sophisticated)
        search_query = self._generate_search_query(claim)
        
        # Step 2: Fetch evidence
        evidence = self.search_fn(search_query)
        
        # Step 3: Compare with LLM
        verification_prompt = f"""Given the following evidence, determine if the claim is TRUE, FALSE, or UNCERTAIN.

CLAIM: {claim}

EVIDENCE: {evidence}

Respond with exactly one word: TRUE, FALSE, or UNCERTAIN."""
        
        llm_response = self.llm_fn(verification_prompt).strip().upper()
        
        # Step 4: Parse result
        if "TRUE" in llm_response:
            return VerificationResult(
                claim=claim,
                is_verified=True,
                confidence=0.9,
                evidence=evidence,
                source="web_search"
            )
        elif "FALSE" in llm_response:
            return VerificationResult(
                claim=claim,
                is_verified=False,
                confidence=0.9,
                evidence=evidence,
                source="web_search"
            )
        else:
            return VerificationResult(
                claim=claim,
                is_verified=False,
                confidence=0.3,  # Low confidence = uncertain
                evidence=evidence,
                source="web_search"
            )
    
    def _generate_search_query(self, claim: str) -> str:
        """
        Transform a claim into a search query.
        For MVP, we just use the claim directly.
        In production, extract key entities and dates.
        """
        # Remove common filler words for better search
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "been", "be"}
        words = [w for w in claim.split() if w.lower() not in stop_words]
        return " ".join(words[:10])  # Limit query length
    
    def fact_check(self, response: str) -> FactCheckReport:
        """
        Fact-check an entire LLM response.
        
        Args:
            response: The LLM-generated text to verify.
            
        Returns:
            FactCheckReport with detailed verification results.
        """
        claims = extract_claims(response)
        results = []
        verified_count = 0
        
        for claim in claims:
            result = self.verify_claim(claim)
            results.append(result)
            if result.is_verified:
                verified_count += 1
        
        total = len(claims) if claims else 1
        
        return FactCheckReport(
            response=response,
            total_claims=len(claims),
            verified_claims=verified_count,
            unverified_claims=len(claims) - verified_count,
            hallucination_score=(len(claims) - verified_count) / total,
            results=results
        )


class DuckDuckGoSearchTool:
    """
    Integration with DuckDuckGo for free web search.
    Requires: pip install duckduckgo-search
    """
    
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self._ddg = None
    
    def _ensure_ddg(self):
        if self._ddg is None:
            try:
                from duckduckgo_search import DDGS
                self._ddg = DDGS()
            except ImportError:
                raise ImportError("Install duckduckgo-search: pip install duckduckgo-search")
    
    def search(self, query: str) -> str:
        """Perform search and return concatenated results."""
        self._ensure_ddg()
        try:
            results = list(self._ddg.text(query, max_results=self.max_results))
            if not results:
                return "No search results found."
            
            formatted = []
            for r in results:
                formatted.append(f"- {r.get('title', 'No title')}: {r.get('body', 'No content')}")
            return "\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"


class WikipediaSearchTool:
    """
    Integration with Wikipedia API for fact verification.
    """
    
    def search(self, query: str) -> str:
        """Fetch Wikipedia summary for query."""
        import urllib.request
        import urllib.parse
        import json
        
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                return data.get("extract", "No information found.")
        except Exception as e:
            return f"Wikipedia lookup failed: {str(e)}"
