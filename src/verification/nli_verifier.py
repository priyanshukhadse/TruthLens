from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch


class NLIVerifier:

    def __init__(
        self,
        model_name=(
            "MoritzLaurer/"
            "DeBERTa-v3-base-mnli-fever-anli"
        )
    ):

        print(
            f"Loading NLI model: "
            f"{model_name}"
        )

        # ----------------------------------------------------
        # Load Tokenizer
        # ----------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # Set Evaluation Mode
        # ----------------------------------------------------

        self.model.eval()

        # ----------------------------------------------------
        # Label Mapping
        # ----------------------------------------------------

        self.id2label = (
            self.model.config.id2label
        )

        print(
            "NLI model loaded successfully."
        )

        print(
            "Label mapping:",
            self.id2label
        )

    # ========================================================
    # VERIFY CLAIM AGAINST EVIDENCE
    # ========================================================

    def verify(
        self,
        claim,
        evidence
    ):

        # ----------------------------------------------------
        # Validate Input
        # ----------------------------------------------------

        if not claim or not evidence:

            return {

                "label":
                    "NEUTRAL",

                "confidence":
                    0.0,

                "probabilities": {

                    "contradiction":
                        0.0,

                    "entailment":
                        0.0,

                    "neutral":
                        0.0

                }

            }

        # ----------------------------------------------------
        # Tokenize Evidence + Claim
        #
        # IMPORTANT:
        #
        # Premise    = Evidence
        # Hypothesis = Claim
        #
        # This is the correct direction for
        # evidence-based fact verification.
        # ----------------------------------------------------

        inputs = self.tokenizer(

            evidence,

            claim,

            return_tensors="pt",

            truncation=True,

            max_length=512

        )

        # ----------------------------------------------------
        # Run NLI Model
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(

                **inputs

            )

        # ----------------------------------------------------
        # Convert Logits to Probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(

            outputs.logits,

            dim=-1

        )[0]

        # ----------------------------------------------------
        # Convert Results to Dictionary
        # ----------------------------------------------------

        label_probabilities = {}

        for index, probability in enumerate(

            probabilities

        ):

            # Get label from model configuration

            label = (

                self.id2label.get(

                    index,

                    str(index)

                )

            ).upper()

            label_probabilities[

                label

            ] = float(

                probability

            )

        # ----------------------------------------------------
        # Get Predicted Label
        # ----------------------------------------------------

        predicted_label = max(

            label_probabilities,

            key=label_probabilities.get

        )

        # ----------------------------------------------------
        # Get Confidence
        # ----------------------------------------------------

        confidence = (

            label_probabilities[

                predicted_label

            ]

        )

        # ----------------------------------------------------
        # Extract Standardized Probabilities
        # ----------------------------------------------------

        contradiction_probability = (

            label_probabilities.get(

                "CONTRADICTION",

                0.0

            )

        )

        entailment_probability = (

            label_probabilities.get(

                "ENTAILMENT",

                0.0

            )

        )

        neutral_probability = (

            label_probabilities.get(

                "NEUTRAL",

                0.0

            )

        )

        # ----------------------------------------------------
        # DEBUG OUTPUT
        # ----------------------------------------------------

        print(

            "\n"
            + "=" * 70

        )

        print(

            "NLI VERIFICATION DEBUG"

        )

        print(

            "=" * 70

        )

        print(

            "\nClaim:"

        )

        print(

            claim

        )

        print(

            "\nEvidence:"

        )

        print(

            evidence

        )

        print(

            "\nNLI Input Direction:"

        )

        print(

            "Premise    = Evidence"

        )

        print(

            "Hypothesis = Claim"

        )

        print(

            "\nNLI Probabilities:"

        )

        print(

            f"CONTRADICTION: "
            f"{contradiction_probability:.4f}"

        )

        print(

            f"ENTAILMENT:    "
            f"{entailment_probability:.4f}"

        )

        print(

            f"NEUTRAL:       "
            f"{neutral_probability:.4f}"

        )

        print(

            "\nPredicted Label:"

        )

        print(

            predicted_label

        )

        print(

            "\nConfidence:"

        )

        print(

            f"{confidence:.4f}"

        )

        print(

            "=" * 70

        )

        # ----------------------------------------------------
        # Return Result
        # ----------------------------------------------------

        return {

            "label":
                predicted_label,

            "confidence":
                confidence,

            "probabilities": {

                "contradiction":
                    contradiction_probability,

                "entailment":
                    entailment_probability,

                "neutral":
                    neutral_probability

            }

        }


