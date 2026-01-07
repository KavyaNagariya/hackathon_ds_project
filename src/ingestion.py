import pathway as pw

def load_novels(data_dir: str):
    """
    Reads files from data_dir but filters to keep ONLY .txt files.
    """
    # 1. Read everything in the folder
    files = pw.io.fs.read(
        path=data_dir,
        format="text",
        mode="static",
        with_metadata=True
    )
    
    # 2. Filter: Keep only files ending in .txt
    # This prevents the system from reading the CSV or other junk as a novel.
    novels = files.filter(pw.this.path.endswith(".txt"))
    
    return novels