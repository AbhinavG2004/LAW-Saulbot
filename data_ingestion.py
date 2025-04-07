import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load data from file
loader = TextLoader("legal_data.txt")  # Change to your data file
documents = loader.load()

# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.split_documents(documents)

# Create FAISS vector database
db = FAISS.from_documents(texts, embeddings)
db.save_local("vector_db")

print("✅ Vector database created and saved as 'vector_db'.")
