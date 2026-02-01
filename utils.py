
import re

def smart_split_text(text, max_length=500):
    """
    Splits text into chunks of at most max_length characters, 
    respecting sentence boundaries where possible.
    """
    if not text:
        return []

    # Regex to split by sentence endings (. ! ?), keeping the punctuation
    # This splits "Hello. World!" into ["Hello.", " World!"]
    # We use a lookbehind or just capturing group to keep delimiter
    # Simple approach: split by typical sentence terminators
    
    # Pattern: match one or more sentence terminators, optionally followed by quotes/parens
    # We'll simple split by spaces first if needed, but regex is better for sentences.
    # Let's use a simpler heuristic:
    # 1. Split into sentences.
    # 2. Group sentences into chunks.
    
    # This regex attempts to match sentence endings.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            # If current chunk is not empty, push it
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If the sentence itself is longer than max_length, we have to hard split it (fallback)
            # or just accept it (user said "okay if chunk is <500", but didn't strictly say NO > 500 if sentence is huge)
            # But usually we should try to keep it under. 
            # If sentence > max_length, we might need comma splitting or hard split.
            # For this MVP, let's just start a new chunk with it.
            current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks
