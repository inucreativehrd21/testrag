from dataclasses import dataclass
from typing import Literal


Difficulty = Literal["easy", "medium", "hard"]
Route = Literal["single", "multistep", "graph"]


@dataclass
class RouteDecision:
    difficulty: Difficulty
    strategy: Route
    reason: str


class QueryRouter:
    def __init__(self):
        self.graph_keywords = {"dependency", "관계", "workflow", "흐름"}
        self.multi_keywords = {"단계", "절차", "troubleshoot", "원인"}

    def classify(self, question: str) -> RouteDecision:
        lower = question.lower()
        if any(word in question for word in self.graph_keywords):
            return RouteDecision("hard", "graph", "연관 관계 분석 필요")
        if any(word in question for word in self.multi_keywords):
            return RouteDecision("medium", "multistep", "절차형 질문")
        difficulty = self._estimate_difficulty(lower)
        strategy: Route = "single" if difficulty == "easy" else "multistep"
        return RouteDecision(difficulty, strategy, "기본 규칙")

    def _estimate_difficulty(self, text: str) -> Difficulty:
        if len(text) < 40:
            return "easy"
        if "고가용" in text or "대규모" in text:
            return "hard"
        return "medium"
