from pathlib import Path
import pandas as pd
from src.RAG.utils import initialize_logger

class ExcelPreprocessor:
    """
    Excel preprocessing class
        1. Reads Excel
        2. Runs preprocessing functionalities
        3. Saves Excel
 
    Input
    --------
    excel_file_path:
        Path
        Location of the Excel file to be preprocessed
    output_file_path:
        Path
        Location of the preprocessed Excel file
    """
   
    def __init__(
        self,
        excel_file_path: Path,
        output_file_path: Path
    ):
        self.logger = initialize_logger(self.__class__.__name__)
        self.excel_file_path = excel_file_path
        self.output_file_path = output_file_path
        self.excel_file = pd.read_excel(self.excel_file_path, sheet_name=None)    
      
       
    def merge_columns(self, column_1: str, column_2: str, sheet_name: str, merged_col_name: str=None):
        """
        Merge columns in Excel sheet
        """
        
        # create new column name
        if not merged_col_name:
            merged_col_name = " ".join([column_1, column_2])

        # get sheet
        df_sheet = self.excel_file[sheet_name]

        # remove empty records
        df_sheet = df_sheet.dropna(subset=[column_1, column_2])

        # merge columns
        df_sheet[merged_col_name] = (
            df_sheet[column_1].astype(str) + " " + df_sheet[column_2].astype(str)
        )
        self.logger.info(f"Merged columns '{column_1}' and '{column_2}'")

        self.excel_file[sheet_name] = df_sheet


    def save(self):
        """
        Save the preprocessed Excel file
        """
        with pd.ExcelWriter(self.output_file_path, engine="openpyxl") as writer:
            for sheet, df in self.excel_file.items():
                df.to_excel(writer, sheet_name=sheet, index=False)


