from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print(
            f"Loading reranker model: "
            f"{model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Reranker model loaded successfully."
        )


    # ========================================================
    # RERANK DOCUMENTS
    # ========================================================

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):


        # ====================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # ====================================================

        pairs = []


        for document in documents:


            if isinstance(document, dict):

                text = document.get(

                    "text",

                    ""

                )

            else:

                text = document



            pairs.append(

                [

                    query,

                    text

                ]

            )



        # ====================================================
        # GENERATE SCORES
        # ====================================================

        scores = self.model.predict(

            pairs

        )



        # ====================================================
        # COMBINE DOCUMENTS AND SCORES
        # ====================================================

        ranked_results = []


        for document, score in zip(

            documents,

            scores

        ):



            if isinstance(document, dict):


                result = {


                    "document":

                        document.get(

                            "text",

                            ""

                        ),


                    "score":

                        float(score),


                    "source":

                        document.get(

                            "source",

                            "unknown"

                        ),


                    "file_path":

                        document.get(

                            "file_path",

                            ""

                        ),


                    "document_type":

                        document.get(

                            "document_type",

                            "txt"

                        ),


                    "chunk_index":

                        document.get(

                            "chunk_index",

                            -1

                        )

                }



            else:


                result = {


                    "document":

                        document,


                    "score":

                        float(score)

                }



            ranked_results.append(

                result

            )



        # ====================================================
        # SORT RESULTS
        # ====================================================

        ranked_results.sort(

            key=lambda x:

                x["score"],

            reverse=True

        )



        # ====================================================
        # TOP-K RESULTS
        # ====================================================

        return ranked_results[

            :top_k

        ]
