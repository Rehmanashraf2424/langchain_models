from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["HF_HOME"]=r"D:\HuggingFaceCache"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "This is a sample text for generating embeddings."
embedding_vector = embeddings.embed_query(text)
print(embedding_vector)
print(f"Embedding vector length: {len(embedding_vector)}")