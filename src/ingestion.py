import pathway as pw

# 1. Define the Schema for the Backstory CSV
# This ensures strict typing and avoids the CsvSettings error.
class QuerySchema(pw.Schema):
    id: str
    book_name: str
    char: str
    caption: str  # Some rows have it, some don't (nullable handled as str)
    content: str

def load_novels(data_dir: str):
    """
    Reads novels as a Pathway Table.
    """
    # Read .txt files. Column 'data' contains the text.
    files = pw.io.fs.read(
        path=data_dir + "*.txt",
        format="plaintext",
        mode="static",
        with_metadata=True
    )
    return files

def load_queries_pathway(csv_path: str):
    """
    Reads the Backstory CSV using the strict schema.
    """
    # FIX: Removed CsvSettings. Used 'schema' instead.
    return pw.io.csv.read(
        csv_path,
        mode="static",
        schema=QuerySchema
    )