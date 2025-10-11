"""Simple retrieval augmented generation tool."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(slots=True)
class RetrievedDocument:
    doc_id: str
    text: str
    score: float


class RAGTool:
    """In-memory lexical retrieval with BM25-like scoring."""

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self._term_freqs: Dict[str, Counter[str]] = {}
        self._doc_freqs: Counter[str] = Counter()

    def index(self, documents: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in documents:
            tokens = self._tokenize(text)
            counter = Counter(tokens)
            self._docs[doc_id] = text
            self._term_freqs[doc_id] = counter
            for token in counter.keys():
                self._doc_freqs[token] += 1

    def query(self, text: str, top_k: int = 3) -> List[RetrievedDocument]:
        if not self._docs:
            return []
        tokens = self._tokenize(text)
        scores: Dict[str, float] = defaultdict(float)
        avg_doc_len = sum(sum(counter.values()) for counter in self._term_freqs.values()) / len(self._term_freqs)
        k1 = 1.6
        b = 0.75
        for token in tokens:
            df = self._doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self._docs) - df + 0.5) / (df + 0.5))
            for doc_id, counter in self._term_freqs.items():
                freq = counter.get(token, 0)
                if freq == 0:
                    continue
                doc_len = sum(counter.values())
                norm = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len / avg_doc_len))
                scores[doc_id] += idf * norm
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [RetrievedDocument(doc_id=doc_id, text=self._docs[doc_id], score=score) for doc_id, score in ranked]

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        word: List[str] = []
        for char in text.lower():
            if char.isspace():
                if word:
                    tokens.append("".join(word))
                    word = []
                continue
            if "\u4e00" <= char <= "\u9fff":
                if word:
                    tokens.append("".join(word))
                    word = []
                tokens.append(char)
                continue
            if char.isalnum():
                word.append(char)
            else:
                if word:
                    tokens.append("".join(word))
                    word = []
        if word:
            tokens.append("".join(word))
        return tokens
