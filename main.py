from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
   "I love pizza",
   "I enjoy eating pizza",
   "The sky is blue"
]

# Convert sentences into vectors
embeddings = model.encode(sentences)

for sentence, vector in zip(sentences, embeddings):
   print("Sentence:", sentence)
   print("Vector length:", len(vector))
   print("First 5 values:", vector[:5])
   print()

