from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
reader = PDFReader()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(file_path: str):
    """Load a PDF file and split it into chunks of text"""
    documents = reader.load_data(file=file_path)
    texts = [d.text for d in documents if getattr(d, 'text', None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks

def embed_texts(texts):
    """Embed a list of texts using OpenAI's embedding API"""
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [e.embedding for e in response.data]