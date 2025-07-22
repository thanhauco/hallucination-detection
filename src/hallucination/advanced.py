from typing import List, Callable, Dict, Optional
import spacy
from dataclasses import dataclass

@dataclass
class CoVeResult:
    original_response: str
    verification_questions: List[str]
    verification_answers: List[str]
    consistency_score: float # 1.0 = Consistent, 0.0 = Inconsistent
    revised_response: Optional[str] = None

class AdvancedVerifier:
    """
    Implements advanced verification techniques:
    1. Chain of Verification (CoVe)
    2. Entity Consistency (NER)
    """
    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
             # Fallback if model not loaded, though check should happen in init
            from spacy.lang.en import English
            self.nlp = English()

    def chain_of_verification(self, prompt: str, response: str) -> CoVeResult:
        """
        Execute Chain of Verification (CoVe).
        Step 1: Generate verification questions.
        Step 2: Answer them independently.
        Step 3: Check consistency.
        """
        # Step 1: Draft verification questions
        q_prompt = f"""
        Given the following response, generate 3 specific fact-checking questions to verify its accuracy.
        Response: {response}
        
        Output format:
        1. Question 1
        2. Question 2
        3. Question 3
        """
        q_output = self.llm_fn(q_prompt)
        questions = [lines.strip() for lines in q_output.split("\n") if "?" in lines][:3]
        
        # Step 2: Answer independently (using LLM or Agent)
        answers = []
        for q in questions:
            # Note: Ideally this would use a search tool, but here we check internal consistency
            # or ask the model "What is the truth regarding: {q}?"
            ans = self.llm_fn(f"Answer this question concisely and truthfully: {q}")
            answers.append(ans)
            
        # Step 3: Check Consistency
        verification_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
        check_prompt = f"""
        Original Response: {response}
        
        Verification Facts:
        {verification_text}
        
        Is the Original Response consistent with the Verification Facts? 
        Answer with a score from 0.0 to 1.0, where 1.0 is perfectly consistent.
        """
        score_str = self.llm_fn(check_prompt).strip()
        try:
            # Naive parsing of float from string
            import re
            match = re.search(r"0\.\d+|1\.0|0|1", score_str)
            score = float(match.group()) if match else 0.5
        except:
            score = 0.5

        return CoVeResult(
            original_response=response,
            verification_questions=questions,
            verification_answers=answers,
            consistency_score=score
        )

    def check_entity_consistency(self, samples: List[str]) -> float:
        """
        Check if Named Entities (Persons, Dates, Orgs) are consistent across multiple samples.
        Returns a consistency score (0.0 to 1.0).
        """
        if not samples:
            return 0.0
            
        # Extract entities from all samples
        all_entities = []
        for s in samples:
            doc = self.nlp(s)
            # Filter for specific types
            ents = {e.text.lower() for e in doc.ents if e.label_ in ["PERSON", "ORG", "DATE", "GPE"]}
            all_entities.append(ents)
            
        if not all_entities:
            return 1.0 # No entities to contradict

        # Check overlap
        # Simplistic metric: Intersection over Union of entities? 
        # Or: Count how many entities in Sample 0 appear in others?
        
        # Let's verify that the *core* entities in Sample 0 are present in at least 50% of other samples
        reference_ents = all_entities[0]
        if not reference_ents:
            return 1.0
            
        consistency_counts = []
        for ent in reference_ents:
            # Count how many other samples contain this entity
            count = sum(1 for other in all_entities[1:] if ent in other)
            consistency_counts.append(count / max(1, len(samples)-1))
            
        return sum(consistency_counts) / len(consistency_counts)
