from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

def semantic_chunks(text):
    chunks = splitter.split_text(text)
    return [{"text": c.strip()} for c in chunks if len(c.strip()) > 100]
