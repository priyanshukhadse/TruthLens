import math


class DecisionEngine:

    def __init__(
        self,
        support_similarity_threshold=0.90,
        strong_similarity_threshold=0.95,
        contradiction_confidence_threshold=0.80,
        strong_reranker_threshold=0.70,
        entailment_confidence_threshold=0.70
    ):

        self.support_similarity_threshold = (
            support_similarity_threshold
        )

        self.strong_similarity_threshold = (
            strong_similarity_threshold
        )

        self.contradiction_confidence_threshold = (
            contradiction_confidence_threshold
        )

        self.strong_reranker_threshold = (
            strong_reranker_threshold
        )

        self.entailment_confidence_threshold = (
            entailment_confidence_threshold
        )


    # ========================================================
    # NORMALIZE RERANKER SCORE
    # ========================================================

    def normalize_reranker_score(self, score):

        try:

            score = float(score)

            # Prevent overflow in sigmoid
            score = max(
                min(score, 50.0),
                -50.0
            )

            return (
                1.0
                /
                (
                    1.0
                    +
                    math.exp(-score)
                )
            )

        except Exception:

            return 0.0


    # ========================================================
    # MAKE FINAL DECISION
    #
    # This decision logic is fully domain-agnostic.
    # It relies only on model outputs (NLI label and
    # confidence, semantic similarity, reranker score) and
    # does not depend on any hardcoded entities, keywords,
    # or subject-matter rules — so it works the same way
    # regardless of what documents you index.
    # ========================================================

    def decide(
        self,
        nli_result,
        similarity_score,
        reranker_score=None,
        claim=None,
        evidence=None
    ):

        # ----------------------------------------------------
        # Extract NLI information
        # ----------------------------------------------------

        if not nli_result:

            return {

                "verdict":
                    "UNVERIFIED",

                "confidence":
                    0.0,

                "nli_label":
                    None,

                "nli_confidence":
                    0.0,

                "similarity_score":
                    similarity_score,

                "reranker_score":
                    reranker_score,

                "normalized_reranker_score":
                    None,

                "reason":
                    "No NLI result was available."

            }


        nli_label = (

            str(

                nli_result.get(
                    "label",
                    "NEUTRAL"
                )

            )

            .upper()

        )


        nli_confidence = float(

            nli_result.get(
                "confidence",
                0.0
            )

        )


        # ----------------------------------------------------
        # Normalize reranker
        # ----------------------------------------------------

        if reranker_score is not None:

            normalized_reranker_score = (

                self.normalize_reranker_score(

                    reranker_score

                )

            )

        else:

            normalized_reranker_score = None


        # ====================================================
        # RULE 1 — STRONG NLI CONTRADICTION
        # ====================================================

        if (

            nli_label
            ==
            "CONTRADICTION"

            and

            nli_confidence
            >=
            self.contradiction_confidence_threshold

        ):

            return {

                "verdict":
                    "CONTRADICTED",

                "confidence":
                    nli_confidence,

                "nli_label":
                    nli_label,

                "nli_confidence":
                    nli_confidence,

                "similarity_score":
                    similarity_score,

                "reranker_score":
                    reranker_score,

                "normalized_reranker_score":
                    normalized_reranker_score,

                "reason":
                    (
                        "The NLI model strongly "
                        "contradicts the claim."
                    )

            }


        # ====================================================
        # RULE 2 — STRONG NLI ENTAILMENT
        # ====================================================

        if (

            nli_label
            ==
            "ENTAILMENT"

            and

            nli_confidence
            >=
            self.entailment_confidence_threshold

        ):

            return {

                "verdict":
                    "SUPPORTED",

                "confidence":
                    nli_confidence,

                "nli_label":
                    nli_label,

                "nli_confidence":
                    nli_confidence,

                "similarity_score":
                    similarity_score,

                "reranker_score":
                    reranker_score,

                "normalized_reranker_score":
                    normalized_reranker_score,

                "reason":
                    (
                        "The NLI model directly "
                        "supports the claim."
                    )

            }


        # ====================================================
        # RULE 3 — HIGH SIMILARITY + HIGH RELEVANCE
        # ====================================================

        if (

            nli_label
            ==
            "NEUTRAL"

            and

            similarity_score
            >=
            self.strong_similarity_threshold

            and

            (

                normalized_reranker_score
                is None

                or

                normalized_reranker_score
                >=
                self.strong_reranker_threshold

            )

        ):

            return {

                "verdict":
                    "SUPPORTED",

                "confidence":
                    similarity_score,

                "nli_label":
                    nli_label,

                "nli_confidence":
                    nli_confidence,

                "similarity_score":
                    similarity_score,

                "reranker_score":
                    reranker_score,

                "normalized_reranker_score":
                    normalized_reranker_score,

                "reason":
                    (
                        "The evidence has very "
                        "high semantic similarity "
                        "and strong relevance to "
                        "the claim."
                    )

            }


        # ====================================================
        # RULE 4 — MODERATE RELEVANCE
        # ====================================================

        if (

            nli_label
            ==
            "NEUTRAL"

            and

            similarity_score
            >=
            self.support_similarity_threshold

        ):

            return {

                "verdict":
                    "UNVERIFIED",

                "confidence":
                    similarity_score,

                "nli_label":
                    nli_label,

                "nli_confidence":
                    nli_confidence,

                "similarity_score":
                    similarity_score,

                "reranker_score":
                    reranker_score,

                "normalized_reranker_score":
                    normalized_reranker_score,

                "reason":
                    (
                        "The evidence is related "
                        "to the claim, but there "
                        "is insufficient evidence "
                        "for a supported verdict."
                    )

            }


        # ====================================================
        # RULE 5 — DEFAULT
        # ====================================================

        return {

            "verdict":
                "UNVERIFIED",

            "confidence":
                nli_confidence,

            "nli_label":
                nli_label,

            "nli_confidence":
                nli_confidence,

            "similarity_score":
                similarity_score,

            "reranker_score":
                reranker_score,

            "normalized_reranker_score":
                normalized_reranker_score,

            "reason":
                (
                    "The available evidence does "
                    "not sufficiently support or "
                    "contradict the claim."
                )

        }
