from rag import RAG


chunk_size = 256
overlap = 50
text_docs = None
load_model = 'model_file/embeddings'
embedding_model = "intfloat/e5-large-v2"
retrieval_model = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
ranking_model = "BAAI/bge-reranker-large"
retrieval_model = 'model_file/llms/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
rag = RAG(load_model,text_docs, chunk_size, overlap, embedding_model, ranking_model, retrieval_model)

questions = [
    "What is Joe's background?",
    "Where did Joe earn his degree(s)? ",
    "When did Joe get educated?",
    "What are some of Joe's projects?",
    "What is Joe's email?",
    "What is Joe's portfolio link?",
    "What is Joe's location?",
    "What is Joe's specialization?",
    "What are Joe's skills?",
"What is Joe's github?"
]


for q in questions:
    print(rag.ask(q))