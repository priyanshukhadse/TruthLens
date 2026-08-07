from pathlib import Path

from pypdf import PdfReader



class DocumentLoader:


    def __init__(

        self,

        data_directory="./data/documents"

    ):


        # ====================================================
        # STORE DATA DIRECTORY
        # ====================================================

        self.data_directory = Path(

            data_directory

        )



    # ========================================================
    # LOAD TEXT FILE
    # ========================================================

    def load_txt(

        self,

        file_path

    ):


        text = (

            file_path.read_text(

                encoding="utf-8"

            )

        )


        return text



    # ========================================================
    # LOAD PDF FILE
    # ========================================================

    def load_pdf(

        self,

        file_path

    ):


        text = ""


        reader = PdfReader(

            str(file_path)

        )


        for page_number, page in enumerate(

            reader.pages,

            start=1

        ):


            page_text = page.extract_text()


            if page_text:

                text += (

                    page_text

                    +

                    "\n"

                )


        return text



    # ========================================================
    # LOAD SINGLE DOCUMENT
    # ========================================================

    def load_file(

        self,

        file_path

    ):


        file_path = Path(

            file_path

        )


        # ----------------------------------------------------
        # Check File
        # ----------------------------------------------------

        if not file_path.exists():

            raise FileNotFoundError(

                f"File not found: {file_path}"

            )



        extension = file_path.suffix.lower()



        # ----------------------------------------------------
        # Select Loader
        # ----------------------------------------------------

        if extension == ".txt":


            text = self.load_txt(

                file_path

            )


        elif extension == ".pdf":


            text = self.load_pdf(

                file_path

            )


        else:


            raise ValueError(

                "Only .txt and .pdf files are supported."

            )



        # ----------------------------------------------------
        # Clean Text
        # ----------------------------------------------------

        text = text.strip()



        if not text:


            raise ValueError(

                f"Document is empty: {file_path}"

            )



        # ----------------------------------------------------
        # Create Document Object
        # ----------------------------------------------------

        document = {


            "text":

                text,


            "source":

                file_path.name,


            "file_path":

                str(file_path),


            "type":

                extension.replace(

                    ".",

                    ""

                )

        }



        return document



    # ========================================================
    # LOAD ALL DOCUMENTS
    # ========================================================

    def load_all_documents(

        self

    ):


        if not self.data_directory.exists():


            raise FileNotFoundError(

                f"Directory not found: "
                f"{self.data_directory}"

            )



        # ----------------------------------------------------
        # Find TXT + PDF
        # ----------------------------------------------------

        files = sorted(

            list(

                self.data_directory.glob(

                    "*.txt"

                )

            )

            +

            list(

                self.data_directory.glob(

                    "*.pdf"

                )

            )

        )



        if not files:


            print(

                "No PDF/TXT documents found."

            )


            return []



        documents = []



        for file_path in files:


            try:


                document = self.load_file(

                    file_path

                )


                documents.append(

                    document

                )


                print(

                    f"Loaded: {file_path.name}"

                )



            except Exception as error:


                print(

                    f"Failed loading "
                    f"{file_path.name}: "
                    f"{error}"

                )



        print(

            "\nDocument loading completed."

        )


        print(

            f"Total documents loaded: "
            f"{len(documents)}"

        )


        return documents
