"""
Evaluation framework — metrics, dataset, and runner.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Evaluation Dataset Models
# ------------------------------------------------------------------ #

class EvaluationQuestion(BaseModel):
    question_id: str
    question: str
    category: str
    document_title: str | None = None
    section_number: str | None = None
    ground_truth_answer: str = ""
    expected_citation_keywords: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    name: str
    version: str = "1.0"
    questions: list[EvaluationQuestion] = Field(default_factory=list)


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Proportion of relevant chunks in top-k retrieved."""
    if not relevant_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_retrieved & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Proportion of top-k results that are relevant."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / min(k, len(top_k))


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """MRR: reciprocal of rank of first relevant result."""
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG@K: measures ranking quality of retrieval."""
    relevant_set = set(relevant_ids)

    def dcg(ids: list[str]) -> float:
        score = 0.0
        for rank, cid in enumerate(ids[:k], start=1):
            if cid in relevant_set:
                score += 1.0 / math.log2(rank + 1)
        return score

    idcg = dcg(list(relevant_set)[:k])
    if idcg == 0:
        return 0.0
    return dcg(retrieved_ids) / idcg


def citation_accuracy(
    citations: list[Any],
    expected_keywords: list[str],
) -> float:
    """What fraction of expected citation keywords appear in the answer's citations."""
    if not expected_keywords:
        return 1.0
    if not citations:
        return 0.0

    all_citation_text = " ".join(
        f"{getattr(c, 'document_title', '')} {getattr(c, 'section', '')} "
        f"{getattr(c, 'excerpt', '')}"
        for c in citations
    ).lower()

    matched = sum(1 for kw in expected_keywords if kw.lower() in all_citation_text)
    return matched / len(expected_keywords)


def groundedness_score(answer: str, evidence_excerpts: list[str]) -> float:
    """
    Simple groundedness: fraction of answer sentences that contain
    at least one word from the evidence (proxy metric).
    True groundedness requires an NLI model.
    """
    if not answer or not evidence_excerpts:
        return 0.0

    evidence_words = set()
    for excerpt in evidence_excerpts:
        evidence_words.update(w.lower() for w in excerpt.split() if len(w) > 4)

    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 0.0

    grounded = sum(
        1 for sent in sentences
        if any(w.lower() in evidence_words for w in sent.split() if len(w) > 4)
    )
    return grounded / len(sentences)


# ------------------------------------------------------------------ #
# Evaluation Runner
# ------------------------------------------------------------------ #

class EvaluationRunner:
    def __init__(self, engine) -> None:
        self.engine = engine

    def run(self, dataset: EvaluationDataset) -> dict[str, Any]:
        """Run evaluation over the full dataset and return aggregate metrics."""
        results = []
        recalls_5, recalls_10, mrrs, ndcgs, precisions_5 = [], [], [], [], []
        cit_accs, groundednesses = [], []

        for q in dataset.questions:
            logger.info("Evaluating Q%s: %s", q.question_id, q.question[:60])
            try:
                response = self.engine.query(q.question)
            except Exception as e:
                logger.error("Error on Q%s: %s", q.question_id, e)
                continue

            retrieved_ids = [r.chunk_id for r in response.supporting_chunks]

            # Retrieval metrics
            if q.relevant_chunk_ids:
                r5 = recall_at_k(retrieved_ids, q.relevant_chunk_ids, 5)
                r10 = recall_at_k(retrieved_ids, q.relevant_chunk_ids, 10)
                mrr = mean_reciprocal_rank(retrieved_ids, q.relevant_chunk_ids)
                ndcg = ndcg_at_k(retrieved_ids, q.relevant_chunk_ids, 5)
                p5 = precision_at_k(retrieved_ids, q.relevant_chunk_ids, 5)
                recalls_5.append(r5)
                recalls_10.append(r10)
                mrrs.append(mrr)
                ndcgs.append(ndcg)
                precisions_5.append(p5)

            # Generation metrics
            cit_acc = citation_accuracy(response.citations, q.expected_citation_keywords)
            evidence_excerpts = [c.excerpt for c in response.citations if c.excerpt]
            ground = groundedness_score(response.answer, evidence_excerpts)
            cit_accs.append(cit_acc)
            groundednesses.append(ground)

            results.append({
                "question_id": q.question_id,
                "question": q.question,
                "answer": response.answer,
                "confidence": response.confidence.value,
                "evidence_status": response.evidence_status.value,
                "citations_count": len(response.citations),
                "citation_accuracy": round(cit_acc, 3),
                "groundedness": round(ground, 3),
            })

        def avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        aggregate = {
            "dataset": dataset.name,
            "total_questions": len(dataset.questions),
            "evaluated": len(results),
            "retrieval_metrics": {
                "recall@5": avg(recalls_5),
                "recall@10": avg(recalls_10),
                "mrr": avg(mrrs),
                "ndcg@5": avg(ndcgs),
                "precision@5": avg(precisions_5),
            },
            "generation_metrics": {
                "citation_accuracy": avg(cit_accs),
                "groundedness": avg(groundednesses),
            },
            "per_question": results,
        }

        return aggregate


# ------------------------------------------------------------------ #
# Built-in evaluation dataset
# ------------------------------------------------------------------ #

EVALUATION_DATASET = EvaluationDataset(
    name="legal_rag_eval_v1",
    questions=[
        EvaluationQuestion(
            question_id="Q01",
            question="What is the notice period required for termination under the Tamil Nadu Shops and Establishments Act?",
            category="employment_acts",
            expected_citation_keywords=["Tamil Nadu", "Shops", "termination", "notice"],
        ),
        EvaluationQuestion(
            question_id="Q02",
            question="What are the mandatory clauses in an NDA agreement?",
            category="contract_rules/nda",
            expected_citation_keywords=["mandatory", "confidential", "disclosure"],
        ),
        EvaluationQuestion(
            question_id="Q03",
            question="What is the limitation period for filing a civil suit under the Code of Civil Procedure?",
            category="dispute_acts",
            expected_citation_keywords=["limitation", "civil", "suit", "period"],
        ),
        EvaluationQuestion(
            question_id="Q04",
            question="What are the gratuity payment provisions under the Payment of Gratuity Act 1972?",
            category="employment_acts",
            expected_citation_keywords=["gratuity", "payment", "employee", "years"],
        ),
        EvaluationQuestion(
            question_id="Q05",
            question="What constitutes cheque dishonour under the Negotiable Instruments Act?",
            category="Finance",
            expected_citation_keywords=["cheque", "dishonour", "Section 138", "negotiable"],
        ),
        EvaluationQuestion(
            question_id="Q06",
            question="What are the mandatory clauses in an employment contract?",
            category="contract_rules/employment",
            expected_citation_keywords=["mandatory", "employment", "clause"],
        ),
        EvaluationQuestion(
            question_id="Q07",
            question="What are the lease agreement mandatory clauses?",
            category="contract_rules/lease",
            expected_citation_keywords=["lease", "mandatory", "clause", "rent"],
        ),
        EvaluationQuestion(
            question_id="Q08",
            question="What is the arbitration procedure under the Arbitration and Conciliation Act 1996?",
            category="dispute_acts",
            expected_citation_keywords=["arbitration", "procedure", "conciliation"],
        ),
        EvaluationQuestion(
            question_id="Q09",
            question="What are the EPF contribution rates under the Employees Provident Fund Act 1952?",
            category="employment_acts",
            expected_citation_keywords=["EPF", "provident fund", "contribution", "rate"],
        ),
        EvaluationQuestion(
            question_id="Q10",
            question="What is the maternity benefit duration under the Maternity Benefit Act 1961?",
            category="employment_acts",
            expected_citation_keywords=["maternity", "benefit", "weeks", "leave"],
        ),
        EvaluationQuestion(
            question_id="Q11",
            question="What are the vendor contract mandatory clauses?",
            category="contract_rules/vendor",
            expected_citation_keywords=["vendor", "mandatory", "clause", "supply"],
        ),
        EvaluationQuestion(
            question_id="Q12",
            question="What is the trademark registration process under the Trade Marks Act 1999?",
            category="ip_acts",
            expected_citation_keywords=["trademark", "registration", "trade marks"],
        ),
        EvaluationQuestion(
            question_id="Q13",
            question="What remedies are available to a landlord under the Tamil Nadu Buildings Lease and Rent Control Act?",
            category="lease_acts",
            expected_citation_keywords=["landlord", "tenant", "Tamil Nadu", "buildings"],
        ),
        EvaluationQuestion(
            question_id="Q14",
            question="What does the Transfer of Property Act say about lease termination?",
            category="lease_acts",
            expected_citation_keywords=["Transfer of Property", "lease", "termination"],
        ),
        EvaluationQuestion(
            question_id="Q15",
            question="What is the minimum wage provision under the Minimum Wages Act?",
            category="employment_acts",
            expected_citation_keywords=["minimum wage", "wages", "scheduled employment"],
        ),
        EvaluationQuestion(
            question_id="Q16",
            question="Summarise the key findings in the Satyam judgment.",
            category="case_law",
            expected_citation_keywords=["Satyam", "fraud", "corporate", "court"],
        ),
        EvaluationQuestion(
            question_id="Q17",
            question="What are the SLA mandatory clauses?",
            category="contract_rules/sla",
            expected_citation_keywords=["SLA", "service level", "mandatory"],
        ),
        EvaluationQuestion(
            question_id="Q18",
            question="Under the Payment of Wages Act 1936, what deductions are permissible from wages?",
            category="employment_acts",
            expected_citation_keywords=["wages", "deduction", "payment", "permissible"],
        ),
        EvaluationQuestion(
            question_id="Q19",
            question="What are the rules about stamp duty for lease agreements under the Indian Stamp Act?",
            category="lease_acts",
            expected_citation_keywords=["stamp duty", "lease", "Indian Stamp Act"],
        ),
        EvaluationQuestion(
            question_id="Q20",
            question="What penalties does the Information Technology Act 2000 prescribe for cyber crimes?",
            category="ip_acts",
            expected_citation_keywords=["IT Act", "cyber", "penalty", "offence"],
        ),
    ],
)
