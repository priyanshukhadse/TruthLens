import chromadb

from sentence_transformers import SentenceTransformer



class Retriever:


    def __init__(

        self,

        model=None,

        model_name="BAAI/bge-small-en-v1.5",

        db_path="./data/chroma_db",

        collection_name="truth_checker"

    ):


        # ====================================================
        # LOAD EMBEDDING MODEL
        # ====================================================


        if model is not None:


            print(
                "\nUsing shared embedding model."
            )


            self.model = model


        else:


            print(
                f"\nLoading embedding model: {model_name}"
            )


            self.model = SentenceTransformer(

                model_name

            )


            print(
                "Embedding model loaded."
            )



        # ====================================================
        # INITIALIZE CHROMADB
        # ====================================================


        self.client = chromadb.PersistentClient(

            path=db_path

        )


        self.collection = self.client.get_or_create_collection(

            name=collection_name

        )


        print(

            f"Using collection: {collection_name}"

        )



    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(

        self,

        chunks,

        batch_size=16

    ):

        if not chunks:

            print(

                "\nNo chunks provided to add_documents()."

            )

            return

        # ----------------------------------------------------
        # BUILD UNIQUE IDS
        # ----------------------------------------------------

        ids = []

        for chunk in chunks:

            source = chunk.get("source", "unknown")

            source_name = source.rsplit(".", 1)[0]

            chunk_index = chunk.get("chunk_index", 0)

            ids.append(

                f"{source_name}_chunk_{chunk_index}"

            )

        # ----------------------------------------------------
        # BUILD METADATA
        # ----------------------------------------------------

        metadatas = [

            {

                "source": chunk.get("source", "unknown"),

                "file_path": chunk.get("file_path", ""),

                "document_type": chunk.get("document_type", "txt"),

                "chunk_index": chunk.get("chunk_index", -1)

            }

            for chunk in chunks

        ]

        # ----------------------------------------------------
        # EXTRACT TEXT
        # ----------------------------------------------------

        texts = [chunk["text"] for chunk in chunks]

        # ----------------------------------------------------
        # GENERATE EMBEDDINGS
        # ----------------------------------------------------

        print(

            f"\nGenerating embeddings for "
            f"{len(texts)} chunks..."

        )

        embeddings = self.model.encode(

            texts,

            batch_size=batch_size,

            show_progress_bar=True,

            normalize_embeddings=True

        ).tolist()

        # ----------------------------------------------------
        # UPSERT INTO CHROMADB
        # ----------------------------------------------------

        self.collection.upsert(

            ids=ids,

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas

        )

        print(

            f"Added {len(chunks)} chunks to collection."

        )


    # ========================================================
    # SEARCH DOCUMENTS
    # ========================================================

    def search(

        self,

        query,

        top_k=5

    ):


        # ----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = self.model.encode(

            query,

            normalize_embeddings=True

        ).tolist()



        # ----------------------------------------------------
        # SEARCH CHROMADB
        # ----------------------------------------------------

        results = self.collection.query(

            query_embeddings=[

                query_embedding

            ],

            n_results=top_k

        )



        # ----------------------------------------------------
        # FORMAT RESULTS
        # ----------------------------------------------------

        retrieved_documents = []



        documents = results.get(

            "documents",

            [[]]

        )[0]


        metadatas = results.get(

            "metadatas",

            [[]]

        )[0]


        distances = results.get(

            "distances",

            [[]]

        )[0]



        for index, document in enumerate(documents):


            metadata = (

                metadatas[index]

                if index < len(metadatas)

                else {}

            )


            distance = (

                distances[index]

                if index < len(distances)

                else 0.0

            )



            retrieved_documents.append(

                {

                    "text":

                        document,


                    "source":

                        metadata.get(

                            "source",

                            "unknown"

                        ),


                    "file_path":

                        metadata.get(

                            "file_path",

                            ""

                        ),


                    "document_type":

                        metadata.get(

                            "document_type",

                            metadata.get(

                                "type",

                                "txt"

                            )

                        ),


                    "chunk_index":

                        metadata.get(

                            "chunk_index",

                            -1

                        ),


                    "distance":

                        distance

                }

            )



        return retrieved_documents
