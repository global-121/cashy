from typing import List

from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.schema import Document

from src.RAG.utils import initialize_logger


class customExcelLoader:
    """
    Custom Excel loading class is used for excel documents:
        1. Loads documents from path
        2. Uses UnstructuredExcelLoader from Langchain to load Excel documents in HTML format
        3. Customized UnstructuredExcelLoader to enable loading from specific sheet and column
        4. Generate metadata; sheet_name, column_name, cell_index, etc.

    Input
    --------

    self_path:
        str
        Location of the document to be loaded
    excel_sheet:
        str
        Only one sheet is allowed to load at one time
        If not specified or does not exist, and error will be raised
    excel_columns:
        List[str]
        The name of excel columns specified are stored in the list
        If not specified, will be an empty List as default, all columns will be loaded
        If the excel columns do not exist, an error will be raised

    Output
    --------
    The output is a list in Document type. Each element has two attributes:
    page_content:
        str
        content of documents. Each page is one page_content in pdf; Each cell is one page_content in excel
    metadata:
        dictionary
        Has information of each page_content
    """

    output: List[Document]

    def __init__(
        self,
        file_path: str,
        excel_sheet: str,
        excel_columns: List[str] = []
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.file_path = file_path
        self.excel_sheet = excel_sheet
        self.excel_columns = excel_columns
        self.unstructured_excel_loader = UnstructuredExcelLoader(
            self.file_path, mode="elements", include_header=True
        )

    def _col_index_to_excel(self, col: int) -> str:
        """
        Convert a zero-based row/column index into corresponding Excel-style cell address, for example, (1, 0) to "A2"

        Input
        --------
        col:
            int
            the number of column index

        Output
        --------
        excel_col:
            str
            the Excel-style cell address
        """
        excel_col = ""
        while col >= 0:
            remainder = col % 26
            excel_col = chr(65 + remainder) + excel_col
            col = col // 26 - 1
        return excel_col

    def _generate_metadata(
        self, doc: Document, column_name: str, col_idx: int, row_idx: int
    ) -> dict:
        """
        Generate information of cell contents, which modifies the metadata created by langchain

        Input
        --------
        doc:
            dictionary
            Metadata attribute created by langchain
        headers:
            list
            Column name of the table, which is created by BeautifulSoup in _process_documents
        col_idx:
            int
            Number of column index
        row_idx:
            int
            Number of row index

        Output
        --------
        new_metadata:
            dictionary
            Metadata of cell contents
        """
        cell_address = self._col_index_to_excel(col_idx) + str(row_idx + 1)

        new_metadata = {
            "source": doc.metadata.get("source", None),
            "page_number": doc.metadata.get("page_number", None),
            "page_name": doc.metadata.get("page_name", None),
            "column_name": column_name,
            "cell_index": cell_address,
        }

        return new_metadata

    def _validate_column_names(self, document: Document):
        """
        Check if the columns specified exist in the loaded documents
        """

        non_existing_columns = []

        html_doc = self._get_html_from_document(document)
        headers = self._get_headers(html_doc)

        for column_name in self.excel_columns:
            if column_name not in headers:
                non_existing_columns.append(column_name)

        if non_existing_columns:
            raise ValueError(
                f"The following column(s) do not exist: {', '.join(non_existing_columns)}"
            )

    def _extract_cells(
        self,
        docs: List[Document],
    ) -> List[Document]:
        """
        Extract and process cell data from the document
        If columns are not specified, all columns will be loaded
        Empty cells will be skipped
        """
        processed_cells = []

        document_soups = [self._get_html_from_document(doc) for doc in docs]
        headers_in_docs = [self._get_headers(soup) for soup in document_soups]

        if not self.excel_columns:
            self.logger.info(
                f"No excel columns specified, loading all columns: {headers_in_doc}"
            )
            self.excel_columns = headers_in_doc

        row_idx = -1
        for document_soup, headers_in_doc, doc in zip(document_soups, headers_in_docs, docs):
            # Finds all table rows ("<tr>") in the parsed HTML ('soup')
            for _, row in enumerate(document_soup.find_all("tr")):

                # Finds all table data ("<td>") elements (cells) within the current row ("row")
                row_idx += 1
                for col_idx, cell in enumerate(row.find_all("td")):
                    # If the col_idx matches the index of the column that should be loaded
                    column_name = headers_in_doc[col_idx]
                    if column_name in self.excel_columns:
                        # If the cell contains text
                        if cell.get_text():
                            # Modify the metadata of the cell through _generate_metadata.
                            metadata = self._generate_metadata(
                                doc, column_name, col_idx, row_idx
                            )

                            # Create the elements of cells containing two attributes: page_content and metadata.
                            cell_doc = Document(
                                page_content=cell.get_text(), metadata=metadata
                            )
                            processed_cells.append(cell_doc)
                    
        return processed_cells

    def _get_headers(self, document_soup: BeautifulSoup) -> List[str]:
        """Get the headers of the excel sheet from the Beautiful Soup read HTML text"""
        return [
            th.get_text(strip=True) for th in document_soup.find("thead").find_all("th")
        ]

    def _get_html_from_document(self, document: Document) -> BeautifulSoup:
        """Get the HTML text from a Langchain Document and read with BeautifulSoup"""
        return BeautifulSoup(document.metadata["text_as_html"], "html.parser")

    def _select_sheet(self, documents: List[Document]) -> List[Document]:
        """
        Select the loaded sheet. Raise an error if the sheet is not specified or the specified sheet does not exist
        """
        sheets = [
            doc
            for doc in documents
            if doc.metadata.get("page_name") == self.excel_sheet
        ]

        if not sheets:
            raise ValueError(f"Sheet {self.excel_sheet} is not found in the documents.")

        return sheets

    def load(self) -> List[Document]:
        """
        Loads the documents using the unstructured excel loader
        Processes the HTML output and converts it to cell based content
        """
        print('excel loading')
        loaded_sheets = self.unstructured_excel_loader.load()
        selected_sheet = self._select_sheet(loaded_sheets)
        for part_sheet in selected_sheet:
            self._validate_column_names(part_sheet)

        processed_docs = self._extract_cells(selected_sheet)

        return processed_docs
