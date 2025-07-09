import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from .text_utils import extract_claims

class RAGVerifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        """
        Initialize the NLI model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def verify_context(self, response: str, context: str) -> float:
        """
        Verify if the response is supported by the context.
        Returns a 'hallucination score' (0.0 = Supported, 1.0 = Unsupported).
        """
        claims = extract_claims(response)
        if not claims:
            return 0.0
            
        unsupported_claims_count = 0
        
        for claim in claims:
            # Check if this claim is entailed by ANY part of the context.
            # Ideally, we split context into chunks too, but for MVP we treat context as premise.
            # If context is too long, it might be truncated.
            is_supported = self._check_entailment(premise=context, hypothesis=claim)
            if not is_supported:
                unsupported_claims_count += 1
                
        return unsupported_claims_count / len(claims)

    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        """
        Check if premise entails hypothesis.
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # DeBERTa-v3 outputs: [Entailment, Neutral, Contradiction] (usually, check specific model config)
        # Note: mapping depends on the specific model. 
        # For 'microsoft/deberta-large-mnli': 0=Contradiction, 1=Neutral, 2=Entailment
        # For 'cross-encoder/nli-deberta-v3-large': 0=Contradiction, 1=Entailment, 2=Neutral (Check config!)
        
        # Let's assume standard MNLI mapping for safety or use a simpler probability check.
        # We'll return True if Entailment > Contradiction.
        
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        
        # Heuristic: If Entailment score is higher than Contradiction and Neutral?
        # Or just Entailment > Thresh?
        entailment_score = probs[2] # Assuming index 2 is entailment (common in MNLI)
        contradiction_score = probs[0]
        
        return entailment_score > 0.5 # Strict threshold
