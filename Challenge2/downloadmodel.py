from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./all-MiniLM-L6-v2')

print("✅ Model saved in folder: all-MiniLM-L6-v2")
