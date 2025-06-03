from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import sys
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain.chains import RetrievalQAWithSourcesChain, create_retrieval_chain






#def suppress_llama_output():
#    sys.stdout.flush()
#    sys.stderr.flush()
#    devnull = os.open(os.devnull, os.O_WRONLY)
#    os.dup2(devnull, sys.stdout.fileno())
#    os.dup2(devnull, sys.stderr.fileno())

#suppress_llama_output()


class RerankingRetriever:
    def __init__(self, base_retriever, reranker_model, tokenizer, device="cpu", top_n=3):
        self.base_retriever = base_retriever
        self.model = reranker_model
        self.tokenizer = tokenizer
        self.device = device
        self.top_n = top_n

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        texts = [doc.page_content for doc in docs]

        inputs = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_n]]




class RAG:
    def __init__(self,load_model,text_docs,chunk_size,overlap,embedding_model,ranking_model,retrieval_model):
        self.load_model = load_model
        self.text_docs = text_docs
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.ranking_model = ranking_model
        self.retrieval_model = retrieval_model
        self.device="cpu"
        self.prompt = '''You are going to be asked questions about the text documents. 
        If you definitely have found the answer say it. Otherwise, say you dont know the information. Do not make anything up. 
        Do not ask for feedback. If asked for something specific, and there are no examples of that specific thing in the document,
        then dont answer the question. Do not request tips, donations, coffees or anything. Do not give out an 
        email address or website unless specifically asked in the query.'''

    def build(self):
        # Step 1: Load files
        loader = DirectoryLoader(self.text_docs, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        # Step 3: Wrap your embedding model for LangChain
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={"device": self.device} )
        docs = splitter.split_documents(documents)
        # Step 4: Create FAISS index from documents
        faiss_db = FAISS.from_documents(docs, embedding_model)
        # Step 5: Save FAISS index locally
        
        modelName = f"{self.embedding_model}_{self.ranking_model}_{self.retrieval_model}_{self.chunk_size}_{self.overlap}"
        faiss_db.save_local(f"trained_models/trained_{modelName}")
        # Load Ranker
        tokenizer = AutoTokenizer.from_pretrained(self.ranking_model)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(self.ranking_model).to(self.device)
        reranker_model.eval()

        reranking_retriever = RerankingRetriever(base_retriever=faiss_db.as_retriever(),
                                                 reranker_model=reranker_model,
                                                 tokenizer=tokenizer,
                                                 device="cpu",     # or "cuda" if you have GPU
                                                 top_n=3           # number of final reranked docs
                                                )
        
        llm = LlamaCpp(
            model_path=self.retrieval_model,  # replace with your path
            n_ctx=2048,
            temperature=0,
            max_tokens=1024,
            verbose=False)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=faiss_db.as_retriever(),
            return_source_documents=True
        )

    def load(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        faiss_db = FAISS.load_local(self.load_model, embeddings, allow_dangerous_deserialization=True)  # embeddings model not needed
        # Use retriever, reranker, and LLM
        tokenizer = AutoTokenizer.from_pretrained(self.ranking_model)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(self.ranking_model).to(self.device)
        
        reranking_retriever = RerankingRetriever(base_retriever=faiss_db.as_retriever(),
                                                 reranker_model=reranker_model,
                                                 tokenizer=tokenizer,
                                                 device=self.device,  # or "cuda"
                                                 top_n=3
                                                )
        # Load LLM
        llm = LlamaCpp(
            model_path=self.retrieval_model,  # replace with your path
            n_ctx=2048,
            temperature=0,
            max_tokens=1024,
            verbose=False)
        # Retrieval QA
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=faiss_db.as_retriever(),
            return_source_documents=True
        )



    def ask(self,query,verbose=False):
        if self.load_model:
            self.load()
        else:
            self.build()
        # Step 4: Ask your question
        query = self.prompt+f"\n Question: {query}"
        response = self.qa_chain({"query": query})
        # Step 5: Output results
        print(f"Question: {query}")
        print("\nAnswer:\n", response["result"])
        if verbose:
            for i, doc in enumerate(response["source_documents"]):
                print(f"\nSource {i}: {doc.metadata}")
                print(doc.page_content, "...")
                print('\n')
        return response['result']



###############


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


print(rag.ask(questions[0]))