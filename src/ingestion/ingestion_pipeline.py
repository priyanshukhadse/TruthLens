from src.ingestion.document_loader import (
    DocumentLoader
)

from src.ingestion.text_chunker import (
    TextChunker
)

from sentence_transformers import (
    SentenceTransformer
)

import chromadb


class IngestionPipeline:

    def __init__(
        self,
        data_directory="./data/documents",
        db_path="./data/chroma_db",
        collection_name="truth_checker",
        model_name="BAAI/bge-small-en-v1.5",
        chunk_size=3,
        chunk_overlap=1,
        batch_size=16
    ):

        # ====================================================
        # STORE CONFIGURATION
        # ====================================================

        self.data_directory = data_directory

        self.db_path = db_path

        self.collection_name = collection_name

        self.model_name = model_name

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.batch_size = batch_size


        # ====================================================
        # INITIALIZE DOCUMENT LOADER
        # ====================================================

        print(
            "Initializing Document Loader..."
        )

        self.loader = DocumentLoader(
            data_directory=self.data_directory
        )


        # ====================================================
        # INITIALIZE TEXT CHUNKER
        # ====================================================

        print(
            "Initializing Text Chunker..."
        )

        print(
            f"Chunk Size: "
            f"{self.chunk_size} sentences"
        )

        print(
            f"Chunk Overlap: "
            f"{self.chunk_overlap} sentence"
        )


        self.chunker = TextChunker(

            chunk_size=self.chunk_size,

            chunk_overlap=self.chunk_overlap

        )


        # ====================================================
        # LOAD EMBEDDING MODEL
        # ====================================================

        print(
            "\nLoading embedding model:"
        )

        print(
            self.model_name
        )


        self.model = SentenceTransformer(

            self.model_name

        )


        print(
            "Embedding model loaded successfully."
        )


        # ====================================================
        # INITIALIZE CHROMADB
        # ====================================================

        print(
            "\nInitializing ChromaDB..."
        )


        self.client = chromadb.PersistentClient(

            path=self.db_path

        )


        # ====================================================
        # CREATE / LOAD COLLECTION
        # ====================================================

        self.collection = (

            self.client.get_or_create_collection(

                name=self.collection_name

            )

        )


        print(

            "Using ChromaDB collection:",

            self.collection_name

        )


    # ========================================================
    # PREPARE DOCUMENTS
    # ========================================================

    def prepare_documents(

        self

    ):

        # ----------------------------------------------------
        # LOAD DOCUMENTS
        # ----------------------------------------------------

        print(

            "\nLoading documents..."

        )


        documents = (

            self.loader.load_all_documents()

        )


        # ----------------------------------------------------
        # CHECK DOCUMENTS
        # ----------------------------------------------------

        if not documents:

            print(

                "No documents available."

            )

            return []


        # ----------------------------------------------------
        # CREATE CHUNKS
        # ----------------------------------------------------

        print(

            "\nCreating text chunks..."

        )


        chunks = (

            self.chunker.chunk_documents(

                documents

            )

        )


        return chunks


    # ========================================================
    # CREATE UNIQUE CHUNK IDS
    # ========================================================

    def create_chunk_ids(

        self,

        chunks

    ):

        ids = []


        for chunk in chunks:

            source = (

                chunk["source"]

            )


            chunk_index = (

                chunk["chunk_index"]

            )


            # ------------------------------------------------
            # REMOVE FILE EXTENSION
            # ------------------------------------------------

            source_name = (

                source.rsplit(

                    ".",

                    1

                )[0]

            )


            # ------------------------------------------------
            # CREATE UNIQUE ID
            # ------------------------------------------------

            chunk_id = (

                f"{source_name}_"

                f"chunk_"

                f"{chunk_index}"

            )


            ids.append(

                chunk_id

            )


        return ids


    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    def generate_embeddings(

        self,

        chunks

    ):

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        texts = [

            chunk["text"]

            for chunk in chunks

        ]


        # ----------------------------------------------------
        # GENERATE EMBEDDINGS
        # ----------------------------------------------------

        print(

            "\nGenerating embeddings..."

        )


        embeddings = (

            self.model.encode(

                texts,

                batch_size=self.batch_size,

                show_progress_bar=True,

                normalize_embeddings=True

            )

        )


        print(

            "Embedding generation completed."

        )


        return embeddings.tolist()


    # ========================================================
    # CLEAR OLD COLLECTION DATA
    # ========================================================

    def clear_collection(

        self

    ):

        print(

            "\nClearing existing ChromaDB records..."

        )


        existing_data = (

            self.collection.get()

        )


        existing_ids = (

            existing_data.get(

                "ids",

                []

            )

        )


        if existing_ids:

            self.collection.delete(

                ids=existing_ids

            )


            print(

                f"Deleted old records: "

                f"{len(existing_ids)}"

            )

        else:

            print(

                "No existing records found."

            )


    # ========================================================
    # STORE CHUNKS
    # ========================================================

    def store_chunks(

        self,

        chunks,

        embeddings

    ):

        # ----------------------------------------------------
        # CREATE IDS
        # ----------------------------------------------------

        ids = (

            self.create_chunk_ids(

                chunks

            )

        )


        # ----------------------------------------------------
        # CREATE METADATA
        # ----------------------------------------------------

        metadatas = []


        for chunk in chunks:

            metadata = {

                "source":

                    chunk["source"],

                "file_path":

                    chunk["file_path"],

                "chunk_index":

                    chunk["chunk_index"]

            }


            metadatas.append(

                metadata

            )


        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        documents = [

            chunk["text"]

            for chunk in chunks

        ]


        # ----------------------------------------------------
        # STORE CHUNKS
        # ----------------------------------------------------

        print(

            "\nStoring chunks in ChromaDB..."

        )


        self.collection.upsert(

            ids=ids,

            documents=documents,

            embeddings=embeddings,

            metadatas=metadatas

        )


        print(

            "Chunks stored successfully."

        )


        print(

            f"Total chunks stored: "

            f"{len(chunks)}"

        )


    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    def run(

        self

    ):

        print(

            "\n"

            + "=" * 70

        )


        print(

            "DOCUMENT INGESTION PIPELINE STARTED"

        )


        print(

            "=" * 70

        )


        # ----------------------------------------------------
        # PREPARE DOCUMENTS
        # ----------------------------------------------------

        chunks = (

            self.prepare_documents()

        )


        # ----------------------------------------------------
        # CHECK CHUNKS
        # ----------------------------------------------------

        if not chunks:

            print(

                "\nNo chunks to ingest."

            )

            return


        # ----------------------------------------------------
        # CLEAR OLD DATA
        # ----------------------------------------------------

        self.clear_collection()


        # ----------------------------------------------------
        # GENERATE EMBEDDINGS
        # ----------------------------------------------------

        embeddings = (

            self.generate_embeddings(

                chunks

            )

        )


        # ----------------------------------------------------
        # STORE CHUNKS
        # ----------------------------------------------------

        self.store_chunks(

            chunks,

            embeddings

        )


        # ----------------------------------------------------
        # CALCULATE UNIQUE SOURCES
        # ----------------------------------------------------

        unique_sources = set(

            chunk["source"]

            for chunk in chunks

        )


        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print(

            "\n"

            + "=" * 70

        )


        print(

            "DOCUMENT INGESTION COMPLETED"

        )


        print(

            "=" * 70

        )


        print(

            f"\nDocuments processed: "

            f"{len(unique_sources)}"

        )


        print(

            f"Chunks stored: "

            f"{len(chunks)}"

        )


        print(

            f"Embedding model: "

            f"{self.model_name}"

        )


        print(

            f"Embedding dimension: "

            f"{len(embeddings[0])}"

        )


        print(

            f"ChromaDB collection: "

            f"{self.collection_name}"

        )
