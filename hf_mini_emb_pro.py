from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Artificial intelligence is reshaping industries and daily life through advanced algorithms.",
    "The rapid growth of the internet has connected the world in unprecedented ways.",
    "A gentle breeze rustled the autumn leaves in the quiet, sunlit park.",
    "Researchers are developing new AI models that can understand and generate human language.",
    "Digital transformation driven by technology is a key topic for modern businesses.",
    "She followed the recipe carefully to bake a chocolate cake for the celebration.",
    "The impact of social media and online platforms on culture is profound and widely studied.",
    "Renewable energy sources like solar and wind power are crucial for a sustainable future.",
    "Machine learning, a subset of AI, relies on data patterns to make predictions.",
    "Historical analysis often reveals how technological shifts influence societal structures."
]

doc_embeddings = embeddings.embed_documents(sentences)

query = "How is AI changing the world?"
query_embedding = embeddings.embed_query(query)
similarities = cosine_similarity([query_embedding], doc_embeddings)
ranked_indices = np.argsort(similarities[0])[::-1]
print("Query:", query)
print("\nTop 3 most similar sentences:")
for idx in ranked_indices[:3]:
    print(f"Score: {similarities[0][idx]:.4f} - Sentence: {sentences[idx]}")
    