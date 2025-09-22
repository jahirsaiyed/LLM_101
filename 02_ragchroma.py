from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Load your notes
docs = [Document(page_content=open("docs/notes.txt").read())]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings + vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Ollama embedding model
store = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_store")

# RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3.2:1b"),
    retriever=store.as_retriever()
)

print(qa.run("Summarize my notes in 3 points."))
