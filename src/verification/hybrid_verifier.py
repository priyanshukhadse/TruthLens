from src.retrieval.retriever import Retriever
from src.reranking.reranker import Reranker
from src.verification.evidence_selector import EvidenceSelector
from src.verification.nli_verifier import NLIVerifier
from src.verification.similarity_scorer import SimilarityScorer
from src.verification.decision_engine import DecisionEngine


class HybridVerifier:

    def __init__(
        self,
        retriever=None,
        reranker=None,
        evidence_selector=None,
        nli_verifier=None,
        similarity_scorer=None,
        decision_engine=None
    ):

        print(
            "\nInitializing Hybrid Verifier..."
        )


        # ====================================================
        # RETRIEVER
        # ====================================================

        print(
            "\nInitializing Retriever..."
        )

        if retriever is None:

            self.retriever = Retriever()

        else:

            self.retriever = retriever


        # ====================================================
        # RERANKER
        # ====================================================

        print(
            "\nInitializing Reranker..."
        )

        if reranker is None:

            self.reranker = Reranker()

        else:

            self.reranker = reranker


        # ====================================================
        # EVIDENCE SELECTOR
        # ====================================================

        print(
            "\nInitializing Evidence Selector..."
        )

        if evidence_selector is None:

            self.evidence_selector = (

                EvidenceSelector(
                    self.reranker
                )

            )

        else:

            self.evidence_selector = (
                evidence_selector
            )


        # ====================================================
        # NLI VERIFIER
        # ====================================================

        print(
            "\nInitializing NLI Verifier..."
        )

        if nli_verifier is None:

            self.nli_verifier = (
                NLIVerifier()
            )

        else:

            self.nli_verifier = (
                nli_verifier
            )


        # ====================================================
        # SIMILARITY SCORER
        # ====================================================

        print(
            "\nInitializing Similarity Scorer..."
        )

        if similarity_scorer is None:

            self.similarity_scorer = (

                SimilarityScorer(

                    model=self.retriever.model

                )

            )

        else:

            self.similarity_scorer = (
                similarity_scorer
            )


        # ====================================================
        # DECISION ENGINE
        # ====================================================

        print(
            "\nInitializing Decision Engine..."
        )

        if decision_engine is None:

            self.decision_engine = (
                DecisionEngine()
            )

        else:

            self.decision_engine = (
                decision_engine
            )


        print(
            "\nHybrid Verifier "
            "initialized successfully."
        )


    # ========================================================
    # VERIFY CLAIM
    # ========================================================

    def verify(
        self,
        claim,
        top_k=3
    ):

        print("\n")

        print(
            "=" * 70
        )

        print(
            "HYBRID VERIFICATION STARTED"
        )

        print(
            "=" * 70
        )


        # ====================================================
        # CLAIM
        # ====================================================

        print(
            "\nClaim:"
        )

        print(
            claim
        )


        # ====================================================
        # STEP 1 — RETRIEVAL
        # ====================================================

        print(
            "\nStep 1: Retrieving evidence..."
        )


        retrieval_results = (
            self.retriever.search(
                query=claim,
                top_k=top_k
            )
        )


        retrieved_documents = [

            item["text"]

            for item in retrieval_results

        ]


        retrieved_metadatas = [

            {

                "source":
                    item.get(
                        "source",
                        "unknown"
                    ),

                "file_path":
                    item.get(
                        "file_path",
                        ""
                    ),

                "document_type":
                    item.get(
                        "document_type",
                        "txt"
                    ),

                "chunk_index":
                    item.get(
                        "chunk_index",
                        -1
                    )

            }

            for item in retrieval_results

        ]

        print(

            f"\nRetrieved "
            f"{len(retrieved_documents)} "
            f"documents."

        )


        # ====================================================
        # NO RETRIEVED DOCUMENTS
        # ====================================================

        if not retrieved_documents:

            return {

                "claim":
                    claim,

                "verdict":
                    "UNVERIFIED",

                "confidence":
                    0.0,

                "reason":
                    "No relevant evidence was retrieved.",

                "evidence":
                    "",

                "nli_result":
                    None,

                "similarity":
                    0.0,

                "reranker_score":
                    0.0,

                "normalized_reranker_score":
                    0.0,

                "retrieved_documents":
                    [],

                "retrieved_metadatas":
                    []

            }


        # ====================================================
        # STEP 2 — RERANKING
        # ====================================================

        print(
            "\nStep 2: Reranking evidence..."
        )


        reranked_results = (

            self.reranker.rerank(

                query=claim,

                documents=retrieval_results,

                top_k=top_k

            )

        )


        if not reranked_results:

            return {

                "claim":
                    claim,

                "verdict":
                    "UNVERIFIED",

                "confidence":
                    0.0,

                "reason":
                    (
                        "No relevant evidence "
                        "remained after reranking."
                    ),

                "evidence":
                    "",

                "nli_result":
                    None,

                "similarity":
                    0.0,

                "reranker_score":
                    0.0,

                "normalized_reranker_score":
                    0.0,

                "retrieved_documents":
                    retrieved_documents,

                "retrieved_metadatas":
                    retrieved_metadatas

            }


        # ====================================================
        # DISPLAY RERANKED RESULTS
        # ====================================================

        print(
            "\nReranked Results:"
        )


        for index, result in enumerate(

            reranked_results,

            start=1

        ):

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Rank {index}"
            )

            print(
                "-" * 70
            )

            print(
                "\nDocument:"
            )

            print(
                result.get(
                    "document",
                    ""
                )

            )

            print(
                "\nReranker Score:"
            )

            print(
                result.get(
                    "score",
                    0.0
                )
            )


        # ====================================================
        # BEST CHUNK
        # ====================================================

        best_chunk = (

            reranked_results[0]

        )

        best_source = best_chunk.get(

            "source",

            "unknown"

        )


        best_file_path = best_chunk.get(

            "file_path",

            ""

        )


        best_document_type = best_chunk.get(

            "document_type",

            "txt"

        )


        best_chunk_index = best_chunk.get(

            "chunk_index",

            -1

        )


        best_document = (

            best_chunk.get(

                "document",

                ""

            )

        )


        best_chunk_score = (

            float(

                best_chunk.get(

                    "score",

                    0.0

                )

            )

        )


        print(
            "\nBest Retrieved Chunk:"
        )

        print(
            best_document
        )


        # ====================================================
        # STEP 2.5 — EVIDENCE SELECTION
        # ====================================================

        print(
            "\nStep 2.5: "
            "Selecting best evidence sentence..."
        )


        selected_evidence = (

            self.evidence_selector.select(

                query=claim,

                document=best_document

            )

        )


        # ====================================================
        # EXTRACT EVIDENCE
        # ====================================================

        evidence_sentence = best_document

        sentence_reranker_score = best_chunk_score


        if isinstance(

            selected_evidence,

            dict

        ):

            evidence_sentence = (

                selected_evidence.get(

                    "sentence",

                    best_document

                )

            )


            sentence_reranker_score = (

                float(

                    selected_evidence.get(

                        "score",

                        best_chunk_score

                    )

                )

            )


        elif isinstance(

            selected_evidence,

            list

        ):

            if selected_evidence:

                first_result = (

                    selected_evidence[0]

                )

                if isinstance(

                    first_result,

                    dict

                ):

                    evidence_sentence = (

                        first_result.get(

                            "sentence",

                            best_document

                        )

                    )

                    sentence_reranker_score = (

                        float(

                            first_result.get(

                                "score",

                                best_chunk_score

                            )

                        )

                    )

                else:

                    evidence_sentence = (

                        str(

                            first_result

                        )

                    )


        elif selected_evidence:

            evidence_sentence = (

                str(

                    selected_evidence

                )

            )


        # ====================================================
        # DISPLAY EVIDENCE
        # ====================================================

        print(
            "\nBest Evidence Sentence:"
        )

        print(
            evidence_sentence
        )


        print(
            "\nSentence Reranker Score:"
        )

        print(
            sentence_reranker_score
        )


        # ====================================================
        # STEP 3 — NLI
        # ====================================================

        print(
            "\nStep 3: Running NLI verification..."
        )


        nli_result = (

            self.nli_verifier.verify(

                claim=claim,

                evidence=evidence_sentence

            )

        )


        print(
            "\nNLI Result:"
        )

        print(
            nli_result
        )


        # ====================================================
        # STEP 4 — SEMANTIC SIMILARITY
        # ====================================================

        print(
            "\nStep 4: Calculating semantic similarity..."
        )


        similarity_score = (

            self.similarity_scorer.calculate_similarity(

                claim=claim,

                evidence=evidence_sentence

            )

        )


        print(
            "\nBGE Similarity:"
        )

        print(
            f"{similarity_score:.4f}"
        )


        # ====================================================
        # STEP 5 — DECISION
        # ====================================================

        print(
            "\nStep 5: Making final decision..."
        )


        decision = (

            self.decision_engine.decide(

                nli_result=nli_result,

                similarity_score=similarity_score,

                reranker_score=sentence_reranker_score,

                claim=claim,

                evidence=evidence_sentence

            )

        )


        # ====================================================
        # EXTRACT DECISION
        # ====================================================

        verdict = (

            decision.get(

                "verdict",

                "UNVERIFIED"

            )

        )


        confidence = float(

            decision.get(

                "confidence",

                0.0

            )

        )


        reason = (

            decision.get(

                "reason",

                "Unable to determine verification result."

            )

        )


        normalized_reranker_score = (

            decision.get(

                "normalized_reranker_score",

                self.decision_engine
                .normalize_reranker_score(

                    sentence_reranker_score

                )

            )

        )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        print("\n")

        print(
            "=" * 70
        )

        print(
            "FINAL VERIFICATION RESULT"
        )

        print(
            "=" * 70
        )


        print(
            "\nVerdict:"
        )

        print(
            verdict
        )


        print(
            "\nConfidence:"
        )

        print(
            f"{confidence:.4f}"
        )


        print(
            "\nReason:"
        )

        print(
            reason
        )


        print(
            "\nEvidence:"
        )

        print(
            evidence_sentence
        )


        print(
            "\n"
            + "=" * 70
        )


        # ====================================================
        # COMPLETE RESULT
        # ====================================================

        result = {

            "claim":
                claim,

            "verdict":
                verdict,

            "confidence":
                confidence,

            "reason":
                reason,

            "evidence":
                evidence_sentence,

            "nli_result":
                nli_result,

            "similarity":
                similarity_score,

            "reranker_score":
                sentence_reranker_score,

            "normalized_reranker_score":
                normalized_reranker_score,

            "retrieved_documents":
                retrieved_documents,

            "retrieved_metadatas":
                retrieved_metadatas

        }


        return result
