import spacy
from typing import List

# Load spacy model (assuming en_core_web_sm is available, otherwise need to download)
# For robustness, we might want to lazy load or handle the error
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback or instruction to download
    # In a real scenario, we'd subprocess run download, but for now we assume it exists or use simple split
    nlp = None

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using Spacy or simple fallback.
    """
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Fallback simple splitting
        return [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

def extract_claims(text: str) -> List[str]:
    """
    Extract atomic claims from text.
    For MVP, we treat each sentence as a potential claim.
    """
    sentences = split_into_sentences(text)
    # Future: Use dependency parsing to split compound sentences
    return sentences
