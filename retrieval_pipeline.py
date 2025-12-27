from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

"""Load embeddings and Vector store"""
embedding_model = OpenAIEmbeddings(model= "text-embedding-3-small")

db= Chroma(
    persist_directory= persistent_directory,
    embedding_function= embedding_model,
    collection_metadata= { "hnsw:space":"cosine"}
)

"""Search for relevant documents"""
query = "In what year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(search_kwargs= {"k": 5})

# retriever = db.as_retriever(
#     search_type= "similarity_score_threshold",
#     search_kwargs= {
#         "k":5,
#         "score_threshold": 0.3  ## Only return chunks with cosine similarity >= 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

print("---Context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")