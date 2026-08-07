import streamlit as st

from pathlib import Path

from src.verification.hybrid_verifier import HybridVerifier
from src.ingestion.ingestion_pipeline import IngestionPipeline



# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="AI Truth Checker",

    page_icon="🛰️",

    layout="wide",

    initial_sidebar_state="expanded"

)



# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(

"""
<style>

.main{

    padding-top:20px;

}


.stMetric{

    border-radius:10px;

    padding:15px;

    background-color:#f8f9fa;

}


.block-container{

    padding-top:2rem;

}


footer{

    visibility:hidden;

}

</style>
""",

unsafe_allow_html=True

)



# ==========================================================
# LOAD VERIFIER
# ==========================================================

@st.cache_resource

def load_verifier():

    return HybridVerifier()



verifier = load_verifier()



# ==========================================================
# DELETE EMBEDDINGS FOR A SOURCE FILE
# ==========================================================

def delete_document_embeddings(source_name):

    """
    Remove every chunk belonging to a given
    source file from the ChromaDB collection.
    """

    collection = verifier.retriever.collection

    matching = collection.get(
        where={"source": source_name}
    )

    matching_ids = matching.get("ids", [])

    if matching_ids:

        collection.delete(
            ids=matching_ids
        )

    return len(matching_ids)



# ==========================================================
# MAIN TITLE
# ==========================================================


st.title(
    "🛰️ AI Truth Checker"
)



st.markdown(

"""
Verify factual claims using an explainable AI pipeline:

- Retrieval-Augmented Generation (RAG)
- Cross Encoder Reranking
- Sentence Level Evidence Selection
- Natural Language Inference
- Semantic Similarity
- Decision Engine

"""

)



# ==========================================================
# DOCUMENT MANAGEMENT
# ==========================================================


st.markdown("---")


st.header(
    "📂 Document Management"
)



documents_directory = Path(

    "./data/documents"

)



documents_directory.mkdir(

    exist_ok=True

)




# ==========================================================
# UPLOAD DOCUMENTS
# ==========================================================


uploaded_files = st.file_uploader(

    "Upload PDF or TXT documents",

    type=[

        "pdf",

        "txt"

    ],

    accept_multiple_files=True

)



if uploaded_files:


    if st.button(

        "💾 Save Uploaded Files",

        use_container_width=True

    ):


        for file in uploaded_files:


            file_path = (

                documents_directory

                /

                file.name

            )


            with open(

                file_path,

                "wb"

            ) as f:


                f.write(

                    file.getbuffer()

                )


        st.success(

            "Documents uploaded successfully."

        )


        st.rerun()




# ==========================================================
# SHOW DOCUMENT LIST
# ==========================================================


st.subheader(

    "Uploaded Documents"

)



files = list(

    documents_directory.glob("*")

)



if files:


    for file in files:


        col1,col2 = st.columns(

            [5,1]

        )


        with col1:


            st.write(

                f"📄 {file.name}"

            )


        with col2:


            if st.button(

                "🗑 Delete",

                key=file.name

            ):


                deleted_count = (

                    delete_document_embeddings(
                        file.name
                    )

                )


                file.unlink()


                st.success(

                    f"{file.name} removed "
                    f"({deleted_count} embeddings deleted)"

                )


                st.rerun()



else:


    st.info(

        "No documents uploaded."

    )





# ==========================================================
# INDEX DOCUMENTS
# ==========================================================


if files:


    st.markdown("---")


    if st.button(

        "⚡ Index Documents",

        use_container_width=True

    ):


        with st.spinner(

            "Processing documents and creating embeddings..."

        ):


            pipeline = IngestionPipeline()


            pipeline.run()



        st.success(

            "Documents indexed successfully."

        )






# ==========================================================
# CLAIM VERIFICATION
# ==========================================================


st.markdown("---")


st.header(

    "🔍 Claim Verification"

)



claim = st.text_area(

    "Enter a factual claim",

    placeholder=

    "Example: The company reported a 20% increase in revenue.",

    height=120

)



verify_button = st.button(

    "🔍 Verify Claim",

    use_container_width=True

)




# ==========================================================
# VERIFY
# ==========================================================


if verify_button:


    if not claim.strip():


        st.warning(

            "Please enter a claim."

        )


        st.stop()



    with st.spinner(

        "Analyzing claim..."

    ):


        result = verifier.verify(

            claim

        )




    verdict = result.get(

        "verdict",

        "UNVERIFIED"

    )


    confidence = float(

        result.get(

            "confidence",

            0

        )

    )


    evidence = result.get(

        "evidence",

        ""

    )


    reason = result.get(

        "reason",

        ""

    )


    similarity = float(

        result.get(

            "similarity",

            0

        )

    )


    reranker = float(

        result.get(

            "normalized_reranker_score",

            0

        )

    )


    documents = result.get(

        "retrieved_documents",

        []

    )


    nli = result.get(

        "nli_result",

        {}

    )




    tab1,tab2,tab3 = st.tabs(

        [

            "📋 Result",

            "📄 Evidence",

            "⚙️ Pipeline"

        ]

    )



    # ======================================================
    # RESULT TAB
    # ======================================================


    with tab1:


        # --------------------------------------------------
        # HEADLINE: CONTEXT MATCH SCORE
        #
        # How closely the claim's context matches
        # the content you've inserted into the
        # document store — this is the primary
        # number the app answers.
        # --------------------------------------------------

        st.subheader(

            "🔗 Context Match Score"

        )


        st.metric(

            label=(
                "How closely this claim matches "
                "your inserted documents"
            ),

            value=f"{similarity*100:.1f}%"

        )


        st.progress(

            min(max(similarity, 0.0), 1.0)

        )


        st.caption(

            "This score reflects semantic closeness "
            "between your claim and the best-matching "
            "evidence found in your uploaded documents."

        )


        st.markdown("---")


        if verdict=="SUPPORTED":


            st.success(

                "✅ SUPPORTED"

            )


        elif verdict=="CONTRADICTED":


            st.error(

                "❌ CONTRADICTED"

            )


        else:


            st.warning(

                "⚠️ UNVERIFIED"

            )



        c1,c2,c3,c4 = st.columns(4)


        c1.metric(

            "Context Match",

            f"{similarity*100:.1f}%"

        )


        c2.metric(

            "Verdict",

            verdict

        )


        c3.metric(

            "Verdict Confidence",

            f"{confidence*100:.2f}%"

        )


        c4.metric(

            "Reranker",

            f"{reranker:.3f}"

        )


        st.caption(

            "Verdict Confidence reflects how sure the "
            "NLI model is in its SUPPORTED/CONTRADICTED "
            "judgment, separate from the Context Match "
            "score above."

        )


        st.info(

            reason

        )





    # ======================================================
    # EVIDENCE TAB
    # ======================================================


    with tab2:


        st.subheader(

            "Selected Evidence"

        )


        st.success(

            evidence

        )


        if nli:


            st.subheader(

                "NLI Result"

            )


            rounded_nli = {

                "label":
                    nli.get("label"),

                "confidence":
                    round(
                        float(
                            nli.get("confidence", 0.0)
                        ),
                        3
                    ),

                "probabilities": {

                    key: round(float(value), 3)

                    for key, value in nli.get(
                        "probabilities",
                        {}
                    ).items()

                }

            }


            st.json(

                rounded_nli

            )





    # ======================================================
    # PIPELINE TAB
    # ======================================================


    with tab3:


        st.metric(

            "Retrieved Documents",

            len(documents)

        )


        st.metric(

            "Similarity",

            f"{similarity:.3f}"

        )


        st.metric(

            "Reranker Score",

            f"{reranker:.3f}"

        )


        st.subheader(

            "Retrieved Chunks"

        )


        for i,doc in enumerate(

            documents,

            start=1

        ):


            with st.expander(

                f"Document {i}"

            ):


                st.write(

                    doc

                )






# ==========================================================
# FOOTER
# ==========================================================


st.markdown("---")


st.caption(

"""
🛰️ AI Truth Checker

ChromaDB • BGE Embeddings • Cross Encoder •
Sentence Evidence Selection • DeBERTa-v3 NLI •
Decision Engine

"""

)
