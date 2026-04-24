def recall_at_k(retrieved_chunks, relevant_phrases, k):
    """
    Phrase-level Recall@k
    """
    retrieved_k = retrieved_chunks[:k]

    hits = 0
    for phrase in relevant_phrases:
        for chunk in retrieved_k:
            if phrase.lower() in chunk.lower():
                hits += 1
                break

    if not relevant_phrases:
        return 0.0

    return hits / len(relevant_phrases)


def reciprocal_rank(retrieved_chunks, relevant_phrases):
    """
    Phrase-based reciprocal rank.
    Finds rank of first relevant hit.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        for phrase in relevant_phrases:
            if phrase.lower() in chunk.lower():
                return 1 / rank

    return 0.0