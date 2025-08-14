import os 
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
docs_dir = os.path.join(os.path.dirname(__file__), "docs_dir")
db_path = os.environ.get('CHROMA_PERSIST_DIR', 'chroma_db')

if groq_api_key is None:
    raise ValueError("GROQ_API_KEY is not set")

class BinarizedEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.embed_model = HuggingFaceEmbeddings(model_name=model_name)

    def _binarize(self, embeddings: List[float]) -> List[int]:
        embeddings_array = np.array(embeddings)
        return (embeddings_array > 0).astype(int).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[int]]:
        float_embeddings = self.embed_model.embed_documents(texts)
        return [self._binarize(embedding) for embedding in float_embeddings]

    def embed_query(self, text: str) -> List[int]:
        float_embedding = self.embed_model.embed_query(text)
        return self._binarize(float_embedding)

class RAG:
    def __init__(self):
        self.llm = self._init_llm()
        self.vector_db = self._init_vector_db()
        self.chain = self._init_chain()

    def _init_llm(self):
        return ChatGroq(model="moonshotai/kimi-k2-instruct", api_key=groq_api_key)

    def _create_vector_db(self):
        loader = DirectoryLoader(docs_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        print("Creating binarized embeddings...")
        embeddings = BinarizedEmbeddings()
        
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)

        print("Vector database created successfully with binarized embeddings.")
        return vector_db
        
    def _init_vector_db(self):
        if not os.path.exists(db_path):
            print("Vector database not found. Creating new one...")
            return self._create_vector_db()
        else:
            print("Loading existing vector database...")
            embeddings = BinarizedEmbeddings()
            return Chroma(persist_directory=db_path, embedding_function=embeddings)

    def _init_chain(self):
        retriever = self.vector_db.as_retriever()
        
        prompt_template = """You are a specialized AI assistant for Lakshya Paliwal's portfolio. Your purpose is to answer questions about Lakshya based on his resume and personal documents.
        Use only the provided context to answer questions.

        If a question is not about Lakshya Paliwal, his skills, experience or his projects, you must politely decline to answer and state that you can only answer questions about him.
        If the context does not contain the answer to a question about Lakshya, you must state that you don't have that information. Do not make up answers.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def chat(self, query: str, chat_history: List[Tuple[str, str]]):
        return self.chain.invoke({"question": query, "chat_history": chat_history})

if __name__ == '__main__':
    rag = RAG()
    chat_history = []
    while True:
        query = input("\nHuman: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break
        
        result = rag.chat(query, chat_history)
        chat_history.append((query, result["answer"]))
        print("\nAssistant: ", result["answer"])
