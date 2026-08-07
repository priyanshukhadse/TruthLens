import re



class TextChunker:


    def __init__(

        self,

        chunk_size=3,

        chunk_overlap=1

    ):

        if chunk_size <= 0:

            raise ValueError(

                "chunk_size must be greater than 0."

            )

        if chunk_overlap >= chunk_size:

            raise ValueError(

                "chunk_overlap must be smaller than "
                "chunk_size, or chunking will never "
                "make progress."

            )

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap



    # ========================================================
    # SPLIT TEXT INTO SENTENCES
    # ========================================================

    def split_into_sentences(

        self,

        text

    ):


        text = re.sub(

            r"\s+",

            " ",

            text

        ).strip()



        sentences = re.split(

            r"(?<=[.!?])\s+",

            text

        )



        sentences = [

            sentence.strip()

            for sentence in sentences

            if sentence.strip()

        ]



        return sentences



    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    def chunk_text(

        self,

        text

    ):


        sentences = self.split_into_sentences(

            text

        )


        chunks = []



        if not sentences:

            return chunks



        start = 0


        chunk_id = 0



        while start < len(sentences):


            end = min(

                start + self.chunk_size,

                len(sentences)

            )



            chunk = " ".join(

                sentences[start:end]

            )



            chunks.append(

                {

                    "text":

                        chunk,


                    "chunk_index":

                        chunk_id

                }

            )



            chunk_id += 1



            if end >= len(sentences):

                break



            start = (

                end -

                self.chunk_overlap

            )



        return chunks



    # ========================================================
    # CHUNK DOCUMENTS
    # ========================================================

    def chunk_documents(

        self,

        documents

    ):


        all_chunks = []



        for document in documents:



            chunks = self.chunk_text(

                document["text"]

            )



            for chunk in chunks:



                chunk["source"] = (

                    document["source"]

                )


                chunk["file_path"] = (

                    document["file_path"]

                )


                chunk["document_type"] = (

                    document.get(

                        "type",

                        "txt"

                    )

                )


                all_chunks.append(

                    chunk

                )



        print(

            "\nText chunking completed."

        )


        print(

            f"Original documents: {len(documents)}"

        )


        print(

            f"Total chunks created: {len(all_chunks)}"

        )



        return all_chunks
