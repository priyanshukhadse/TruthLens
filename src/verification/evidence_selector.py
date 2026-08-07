import re


class EvidenceSelector:

    def __init__(
        self,
        reranker
    ):

        print(
            "Using existing reranker "
            "for evidence sentence selection."
        )

        # ----------------------------------------------------
        # Reuse Existing Reranker
        # ----------------------------------------------------

        self.reranker = reranker


    # ========================================================
    # SPLIT DOCUMENT INTO SENTENCES
    # ========================================================

    def split_sentences(
        self,
        document
    ):

        sentences = []

        current_sentence = ""

        # ----------------------------------------------------
        # Simple Sentence Splitting
        # ----------------------------------------------------

        for character in document:

            current_sentence += character

            if character in ".!?":

                sentence = (
                    current_sentence
                    .strip()
                )

                if sentence:

                    sentences.append(
                        sentence
                    )

                current_sentence = ""


        # ----------------------------------------------------
        # Handle Remaining Text
        # ----------------------------------------------------

        remaining_text = (
            current_sentence
            .strip()
        )

        if remaining_text:

            sentences.append(
                remaining_text
            )


        return sentences


    # ========================================================
    # CHECK WHETHER SENTENCE NEEDS CONTEXT
    # ========================================================

    def needs_context(
        self,
        sentence,
        query
    ):

        # ----------------------------------------------------
        # Pronouns That Often Require Previous Context
        # ----------------------------------------------------

        context_dependent_words = [

            "it",

            "its",

            "they",

            "their",

            "this",

            "that",

            "these",

            "those",

            "he",

            "she",

            "them"

        ]


        # ----------------------------------------------------
        # Convert Sentence to Lowercase
        # ----------------------------------------------------

        sentence_lower = (

            sentence

            .lower()

        )


        # ----------------------------------------------------
        # Check First Word
        # ----------------------------------------------------

        words = (

            sentence_lower

            .split()

        )


        if words:

            first_word = (

                words[0]

                .strip(
                    ".,!?;:"
                )

            )

            if first_word in context_dependent_words:

                return True


        # ----------------------------------------------------
        # Check Common Context-Dependent Patterns
        # ----------------------------------------------------

        context_patterns = [

            "it is",

            "it was",

            "it has",

            "it can",

            "this is",

            "this was",

            "the object"

        ]


        for pattern in context_patterns:

            pattern_regex = (

                r"\b" + re.escape(pattern) + r"\b"

            )

            if re.search(pattern_regex, sentence_lower):

                # Only use context if query entity
                # is not explicitly present.

                query_words = (

                    query.lower()

                    .split()

                )


                # Check whether any meaningful
                # query word appears.

                meaningful_query_words = [

                    word.strip(
                        ".,!?;:"
                    )

                    for word in query_words

                    if len(
                        word.strip(
                            ".,!?;:"
                        )
                    ) > 2

                ]


                has_query_entity = any(

                    word in sentence_lower

                    for word in meaningful_query_words

                )


                if not has_query_entity:

                    return True


        return False


    # ========================================================
    # SELECT BEST EVIDENCE SENTENCE
    # ========================================================

    def select(
        self,
        query,
        document
    ):

        print(
            "\nSplitting document into sentences..."
        )

        # ----------------------------------------------------
        # Split Document into Sentences
        # ----------------------------------------------------

        sentences = self.split_sentences(
            document
        )


        # ----------------------------------------------------
        # Handle Empty Document
        # ----------------------------------------------------

        if not sentences:

            return {

                "sentence":
                    "",

                "score":
                    0.0

            }


        print(
            f"Total sentences found: "
            f"{len(sentences)}"
        )


        # ----------------------------------------------------
        # Create Query-Sentence Pairs
        # ----------------------------------------------------

        pairs = []

        for sentence in sentences:

            pairs.append(

                [
                    query,
                    sentence
                ]

            )


        # ----------------------------------------------------
        # Calculate Sentence Relevance
        # ----------------------------------------------------

        print(
            "Calculating sentence relevance scores..."
        )

        scores = (

            self.reranker.model.predict(

                pairs

            )

        )


        # ----------------------------------------------------
        # Find Best Sentence
        # ----------------------------------------------------

        best_index = max(

            range(
                len(scores)
            ),

            key=lambda index:
                scores[index]

        )


        best_sentence = (

            sentences[
                best_index
            ]

        )


        best_score = float(

            scores[
                best_index
            ]

        )


        # ====================================================
        # CONTEXT-AWARE EVIDENCE SELECTION
        # ====================================================

        selected_sentences = []


        # ----------------------------------------------------
        # Check Previous Sentence
        # ----------------------------------------------------

        if best_index > 0:

            previous_sentence = (

                sentences[
                    best_index - 1
                ]

            )


            if self.needs_context(

                best_sentence,

                query

            ):

                selected_sentences.append(

                    previous_sentence

                )


        # ----------------------------------------------------
        # Add Best Sentence
        # ----------------------------------------------------

        selected_sentences.append(

            best_sentence

        )


        # ----------------------------------------------------
        # Check Next Sentence
        # ----------------------------------------------------

        if best_index < (

            len(sentences) - 1

        ):

            next_sentence = (

                sentences[
                    best_index + 1
                ]

            )


            # Add next sentence only when
            # the best sentence is short or
            # context-dependent.

            if (

                self.needs_context(

                    best_sentence,

                    query

                )

                and

                len(
                    best_sentence.split()
                )

                <

                30

            ):

                selected_sentences.append(

                    next_sentence

                )


        # ----------------------------------------------------
        # Remove Duplicate Sentences
        # ----------------------------------------------------

        unique_sentences = []


        for sentence in (

            selected_sentences

        ):

            if sentence not in unique_sentences:

                unique_sentences.append(

                    sentence

                )


        # ----------------------------------------------------
        # Combine Evidence
        # ----------------------------------------------------

        final_evidence = (

            " ".join(

                unique_sentences

            )

        )


        # ----------------------------------------------------
        # Debug Output
        # ----------------------------------------------------

        print(

            "\nSelected Evidence:"

        )

        print(

            final_evidence

        )


        print(

            "\nEvidence Score:"

        )

        print(

            best_score

        )


        # ----------------------------------------------------
        # Return Context-Aware Evidence
        # ----------------------------------------------------

        return {

            "sentence":
                final_evidence,

            "score":
                best_score

        }

