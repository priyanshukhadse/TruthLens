from sentence_transformers import SentenceTransformer

import torch


class SimilarityScorer:

    def __init__(

        self,

        model=None,

        model_name="BAAI/bge-small-en-v1.5"

    ):

        # ====================================================
        # LOAD OR REUSE EMBEDDING MODEL
        # ====================================================

        if model is not None:

            print(

                "\nUsing shared embedding model "
                "for similarity scoring."

            )

            self.model = model

        else:

            print(

                f"\nLoading similarity model: "
                f"{model_name}"

            )

            self.model = (

                SentenceTransformer(

                    model_name

                )

            )

            print(

                "Similarity model loaded successfully."

            )


    # ========================================================
    # CALCULATE SEMANTIC SIMILARITY
    # ========================================================

    def calculate_similarity(

        self,

        claim,

        evidence

    ):

        # ----------------------------------------------------
        # Generate Embeddings
        # ----------------------------------------------------

        embeddings = (

            self.model.encode(

                [

                    claim,

                    evidence

                ],

                convert_to_tensor=True

            )

        )


        # ----------------------------------------------------
        # Extract Claim Embedding
        # ----------------------------------------------------

        claim_embedding = (

            embeddings[

                0

            ]

        )


        # ----------------------------------------------------
        # Extract Evidence Embedding
        # ----------------------------------------------------

        evidence_embedding = (

            embeddings[

                1

            ]

        )


        # ----------------------------------------------------
        # Calculate Cosine Similarity
        # ----------------------------------------------------

        similarity = (

            torch.nn.functional
            .cosine_similarity(

                claim_embedding,

                evidence_embedding,

                dim=0

            )

        )


        # ----------------------------------------------------
        # Convert To Float
        # ----------------------------------------------------

        similarity = float(

            similarity.item()

        )


        # ----------------------------------------------------
        # Return Similarity
        # ----------------------------------------------------

        return similarity
