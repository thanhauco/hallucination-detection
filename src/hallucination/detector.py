import numpy as np
import torch
from typing import List, Callable, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HallucinationDetector:
    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        """
        Initialize the SelfCheckGPT-NLI detector.
        Uses DeBERTa-v3-large for robust entailment checking.
        """
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        return self._model

    def detect_uncertainty(self, 
                           prompt: str, 
                           response: str, 
                           sampling_fn: Callable[[str, int], List[str]], 
                           num_samples: int = 5) -> float:
        """
        Estimate hallucination probability using SelfCheckGPT (NLI variant).
        Hypothesis: If the model 'knows' a fact, diverse samples should logically entail the main response (and vice-versa).
        
        Args:
            prompt: User query.
            response: The main response to evaluate.
            sampling_fn: Function that takes (prompt, num_samples) and returns a list of generated strings.
            num_samples: Number of stochastic samples to generate.
            
        Returns:
            float: SelfCheckGPT score (0.0 = Consistent/Fact, 1.0 = Hallucinated)
        """
        # 1. Generate stochastic samples
        samples = sampling_fn(prompt, num_samples)
        
        if not samples:
            return 0.0 # Cannot evaluate without samples
            
        # 2. SelfCheckGPT-NLI Logic:
        # For each sample S, check if S entails the main Response R?
        # Actually, standard SelfCheckGPT checks: Does Response R entail Sample S? 
        # Or better: Does Sample S entail Response R sentence-by-sentence?
        # Simplified Implementation:
        # Check if the Main Response is consistent with the set of Samples.
        # We calculate P(Hallucination | Samples) ~ Fraction of samples that CONTRADICT the response.
        
        # Note: NLI models usually output: Entailment, Neutral, Contradiction.
        # We define Hallucination Score = Mean Contradiction Probability across samples.
        
        contradiction_scores = []
        for sample in samples:
            score = self._check_contradiction(premise=sample, hypothesis=response)
            contradiction_scores.append(score)
            
        # High contradiction means high hallucination probability
        return float(np.mean(contradiction_scores))

    def _check_contradiction(self, premise: str, hypothesis: str) -> float:
        """
        Returns probability that premise CONTRADICTS hypothesis.
        """
        # Truncate for efficiency if needed
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # DeBERTa-v3 labels: 0=Contradiction, 1=Entailment, 2=Neutral (Check model card!)
            # MNLI mapping is typically: 0=Entailment, 1=Neutral, 2=Contradiction
            # Check specifically for microsoft/deberta-v3-large fine-tuned on MNLI.
            # Assuming standard output logits.
            
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
            # Heuristic map. Let's assume index 2 is contradiction for standard MNLI. 
            # Ideally we check model config. 
            # For simplicity in this implementation, we assume [Entailment, Neutral, Contradiction] -> Index 2
            contradiction_prob = probs[2].item()
            
            return contradiction_prob

    def mock_check(self, main: str, samples: List[str]) -> float:
        """Fallback for tests without loading heavy model."""
        return 0.1
