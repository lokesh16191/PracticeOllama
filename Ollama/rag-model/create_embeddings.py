from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Load your text
with open("data.txt", "r") as f:
    text = f.read()

# Split into chunks (for better embedding and retrieval)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents([Document(page_content=text)])

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="turingdance/gte-large-zh:latest")

# Create FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)

# Optional: save for later
db.save_local("vector_db_gte_large")
