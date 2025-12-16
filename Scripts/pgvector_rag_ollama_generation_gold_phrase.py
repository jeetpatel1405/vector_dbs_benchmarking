import os
import time
import textwrap
import numpy as np
import sys
from typing import List, Optional, Union, Iterator, Generator, Literal
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity # Required for benchmarking accuracy

 
try:
     
    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.messages import HumanMessage
    
     
    from langchain_ollama import OllamaLLM  
    from langchain_community.vectorstores import PGVector  
    
     
    try:
        from blueprints.function_calling_blueprint import Pipeline as Blueprint
    except ImportError:
        print("Warning: 'blueprints.function_calling_blueprint' not found. Using simple class inheritance.")
        class Blueprint(object): pass
    
    print("Open WebUI Pipeline: All base libraries imported successfully")

except ImportError as e:
    print(f"FATAL ERROR: Open WebUI Pipeline failed to load due to a missing library: {e}")
    Blueprint = object
    PGVector = None
    OllamaLLM = None
    sys.exit(1)
    
 
OLLAMA_MODEL = "mistral"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
DOCS_FOLDER = 'docs'
ACCURACY_THRESHOLD = 0.8  

 
PG_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING", "postgresql+psycopg2://postgres:root@localhost:5432/rag_db")
PG_COLLECTION_NAME = "openwebui_rag_vectors" 

 
class TestCase(BaseModel):
    query: str
    gold_answer: str

 
def _format_context_prompt(query, retrieved_docs):
    """Formats the context and question into a RAG prompt."""
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return textwrap.dedent(f"""\
        Context:
        {context}

        Question:
        {query}

        ---

        Based only on the provided context above, answer the question. If the context does not contain the answer, state that you cannot answer based on the provided information.
    """)

def check_similarity_accuracy(
    response: str, 
    gold_answer: str, 
    embeddings_model: HuggingFaceEmbeddings, 
    threshold: float = ACCURACY_THRESHOLD
) -> (bool, float):
    """
    Checks if the cosine similarity between the response and gold answer 
    embeddings meets the specified threshold.
    """
    if not response or not gold_answer:
        return False, 0.0

    
    embeddings = embeddings_model.embed_documents([response, gold_answer])
    
    
    embed_a = np.array(embeddings[0]).reshape(1, -1)
    embed_b = np.array(embeddings[1]).reshape(1, -1)

     
    similarity = cosine_similarity(embed_a, embed_b)[0][0]
    
     
    return similarity >= threshold, similarity




class PGVectorRAG(object):
    """Handles RAG with PGVector for retrieval and uses a provided LLM for generation."""

    def __init__(self, embeddings, text_splitter, generation_llm: OllamaLLM):
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        self.generation_llm = generation_llm  
        self.vectorstore = None
        self.retriever = None
        self.indexing_time = 0.0  
        self.total_docs_size_bytes = 0  

    def _initialize_pgvector_store(self):
        """Connects to PGVector and initializes the LangChain vector store."""
        
        print(f"Attempting to connect to PostgreSQL at: {PG_CONNECTION_STRING.split('@')[-1]}...") 
        
        if PGVector is None:
            raise RuntimeError("PGVector client not available. Check the ImportError above.")

        try:
            self.vectorstore = PGVector(
                connection_string=PG_CONNECTION_STRING,
                embedding_function=self.embeddings,
                collection_name=PG_COLLECTION_NAME,
            )
            
            print(f"PGVector VectorStore for collection '{PG_COLLECTION_NAME}' initialized.")
            
             
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10}) 
            
        except Exception as e:
            raise ConnectionError(f"PGVector Connection/Initialization Failed. Check PostgreSQL status and connection string: {e}")

    def index_document_file(self, file_path: str) -> float:
        """Loads, splits, embeds, and indexes a single document file, returning the indexing time."""
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
         
        time_ingest = time.time() 
        loader = TextLoader(file_path)
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        ingest_time = time.time() - time_ingest
        self.total_docs_size_bytes += os.path.getsize(file_path)
        
        print(f"Split into {len(texts)} chunks. Ingestion Time: {ingest_time:.6f}s")
        
         
        start_time = time.time()
        self.vectorstore.add_documents(texts)
        indexing_time = time.time() - start_time
        print(f"Indexing time for {os.path.basename(file_path)}: {indexing_time:.6f} seconds")
        return indexing_time

    def index_documents_individually(self):
        """Iterates over files in the docs folder and indexes them one by one."""
        if not self.vectorstore: return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, DOCS_FOLDER)
        print(f"Indexing documents found in: {docs_path}")
        
        if not os.path.exists(docs_path) or not os.listdir(docs_path):
            print(f"Docs directory is empty or does not exist. Skipping indexing.")
            return

        total_indexing_time = 0.0
        file_times = {}

        for file in os.listdir(docs_path):
            if file.endswith(".txt"):
                file_path = os.path.join(docs_path, file)
                try:
                    time_taken = self.index_document_file(file_path)
                    file_times[file] = time_taken
                    total_indexing_time += time_taken
                except Exception as e:
                    print(f"Error indexing file {file}: {e}")
        
        self.indexing_time = total_indexing_time  
        print("\n--- Summary of Individual File Indexing Times ---")
        for file, t in file_times.items():
            print(f"  {file}: {t:.6f} seconds")
        print(f"Total Indexing Time (All files): {self.indexing_time:.6f} seconds")
        
    def timed_rag_generation(self, query: str, top_k: int) -> (str, float, float):
        """
        Performs timed RAG generation using the internal Ollama LLM.
        
        :returns: (response_content, retrieval_latency, generation_latency)
        """
        if not self.retriever:
            return "Internal Error: PGVector retriever not set up.", 0.0, 0.0

       
        original_k = self.retriever.search_kwargs.get("k", 10)
        self.retriever.search_kwargs["k"] = top_k
        
        
        start_time_retrieval = time.time()
        retrieved_docs = self.retriever.get_relevant_documents(query)
        end_time_retrieval = time.time()
        retrieval_latency = end_time_retrieval - start_time_retrieval
        
        
        prompt = _format_context_prompt(query, retrieved_docs)
        
        
        start_time_generation = time.time()
        try:
            response_content = self.generation_llm.invoke(prompt) 
        except Exception as e:
            response_content = f"Ollama Generation Error: {e}"
        end_time_generation = time.time()
        generation_latency = end_time_generation - start_time_generation
        
        
        self.retriever.search_kwargs["k"] = original_k 

         
        return response_content, retrieval_latency, generation_latency

    def generate_completion(self, user_query):
        """Standard pipeline method using the default k=10."""
        
        if not self.retriever:
            return "Internal Error: PGVector retriever not set up. Check connection logs."
            
        try:
             
            final_answer, retrieval_latency, generation_latency = self.timed_rag_generation(
                query=user_query,
                top_k=self.retriever.search_kwargs.get("k", 10)
            )
            print(f"Pipeline Total Latency: {retrieval_latency + generation_latency:.4f}s")
            return final_answer
        except Exception as e:
            print(f"Error during Ollama generation in pipeline: {e}")
            return f"An error occurred: {e}"


 

class Pipeline(Blueprint):
    class Valves(BaseModel):
        pass

    api_version = "v1"
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        self.ollama_llm = None
        self.pgvector_rag = None
        self.type = "manifold"
        self.name = "pgvector-ollama-rag-pipeline"
        
         
        try:
            if OllamaLLM is None: raise ImportError("OllamaLLM class is not available.")
            print(f"Initializing local Ollama LLM: {OLLAMA_MODEL} at {OLLAMA_URL}...")
            self.ollama_llm = OllamaLLM(
                model=OLLAMA_MODEL, 
                base_url=OLLAMA_URL, 
                temperature=0.5,
                stop=["\n\n---"]
            )
            print("Ollama LLM initialized successfully.")
        except Exception as e:
            print(f"FATAL: Ollama LLM Initialization Failed: {e}")
            self.ollama_llm = None
            
         
        print("Pipeline instance created. Starting EAGER RAG components initialization and indexing...")
        
        try:
            if self.ollama_llm:
                self.pgvector_rag = PGVectorRAG(self.embeddings, self.text_splitter, self.ollama_llm)
                self.pgvector_rag._initialize_pgvector_store()
                self.pgvector_rag.index_documents_individually()
                print("RAG Setup Complete: Documents indexed successfully on startup.")
                
                 
                self.benchmark_rag(
                    test_cases=[
                        TestCase(
                            query="What is the positive feedback climate process where a change in the area of ice alters the albedo and surface temperature of a planet?",
                            gold_answer="The Ice-albedo feedback."
                        ),
                        TestCase(
                            query="If Earth were an entirely frozen planet, what would be the most important factor causing a drop in temperature?",
                            gold_answer="Its higher reflectiveness (albedo)"
                        ),
                        TestCase(
                            query="What is the minimum thermal emittance that must be achieved for widespread PDRC efforts?",
                            gold_answer="A thermal emittance of at least 90%."
                        ),
                        TestCase(
                            query="What was Carl Friedrich Gauss's position at the University of Göttingen from 1807 until his death?",
                            gold_answer="He was the Director of the Göttingen Observatory in Germany and Professor of Astronomy"
                        ),
                        TestCase(
                            query="Name two of the masterpieces that Carl Friedrich Gauss wrote as an independent scholar.",
                            gold_answer="'Disquisitiones Arithmeticae' and 'Theoria motus corporum coelestium'"
                        ),
                    ],
                    top_k_values=[1, 5, 10,20,25,30,35],  
                    n_runs=2
                )
            else:
                self.pgvector_rag = None
                print("Skipping RAG setup: Ollama LLM failed to initialize.")
        except ConnectionError as e:
            print(f"FATAL: RAG Initialization Failed on startup. Check DB status: {e}")
            self.pgvector_rag = None
        except Exception as e:
            print(f"FATAL: RAG Setup Error on startup: {e}")
            self.pgvector_rag = None

    def benchmark_rag(self, test_cases: List[TestCase], top_k_values: List[int], n_runs: int = 1):
        """
        Executes a benchmarking run, measuring retrieval, latency, and cosine similarity accuracy.
        """
        print("\n" + "="*60)
        print(f"STARTING RAG BENCHMARK (PGVector + Ollama/{OLLAMA_MODEL})")
        print(f"Similarity Threshold: {ACCURACY_THRESHOLD}")
        print("="*60)
        
         
        if self.pgvector_rag:
            print(f"Knowledge Base Size (Raw): {self.pgvector_rag.total_docs_size_bytes / (1024**2):.2f} MB")
            print(f"Indexing Time: {self.pgvector_rag.indexing_time:.4f}s")
        else:
            print("Cannot display metrics: PGVectorRAG failed to initialize.")
            return

        print("-" * 60)
        
        retrieval_results = {}
        total_latency_results = {}
        accuracy_results = {}

        if not self.pgvector_rag or not self.ollama_llm:
            print("Cannot run full benchmark: RAG components or Ollama LLM failed to initialize.")
            return
            
        embeddings_model = self.pgvector_rag.embeddings 

         
        for k in top_k_values:
            print(f"Benchmarking k={k}...")
            total_retrieval_time = 0
            total_rag_latency = 0
            correct_answers = 0 
            total_similarity_score = 0.0 
            
            for test_case in test_cases:
                for _ in range(n_runs):
                     
                    final_answer, retrieval_latency, generation_latency = self.pgvector_rag.timed_rag_generation(
                        query=test_case.query,
                        top_k=k
                    )
                    
                     
                    is_correct, similarity_score = check_similarity_accuracy(
                        response=final_answer,
                        gold_answer=test_case.gold_answer,
                        embeddings_model=embeddings_model, 
                        threshold=ACCURACY_THRESHOLD
                    )

                    if is_correct:
                        correct_answers += 1
                        
                    total_similarity_score += similarity_score
                        
                    # 3. Time tracking
                    total_retrieval_time += retrieval_latency
                    total_rag_latency += (retrieval_latency + generation_latency)
                        

             
            total_runs = len(test_cases) * n_runs
            
            if total_runs > 0:
                avg_retrieval_time = total_retrieval_time / total_runs
                retrieval_qps = 1 / avg_retrieval_time if avg_retrieval_time > 0 else 0.0
                avg_total_latency = total_rag_latency / total_runs
                accuracy = correct_answers / total_runs
                avg_similarity = total_similarity_score / total_runs
            else:
                avg_retrieval_time = avg_total_latency = accuracy = avg_similarity = retrieval_qps = 0.0

            
            retrieval_results[k] = {"avg_time": avg_retrieval_time, "qps": retrieval_qps}
            total_latency_results[k] = {"avg_total_latency": avg_total_latency}
            accuracy_results[k] = {"accuracy": accuracy, "avg_similarity": avg_similarity} 

        
        print("\n" + "#"*60)
        print("RETRIEVAL TIME & QPS vs. TopK")
        print("#"*60)
        print(f"{'TopK':<5} | {'Avg. Retr. Time (s)':<20} | {'Queries per Sec (QPS)':<20}")
        print("-" * 60)
        for k, metrics in retrieval_results.items():
            print(f"{k:<5} | {metrics['avg_time']:.6f} | {metrics['qps']:.2f}")

        print("\n" + "#"*60)
        print("TOTAL RAG LATENCY vs. TopK (Retrieval + Generation)")
        print("#"*60)
        print(f"{'TopK':<5} | {'Avg. Total Latency (s)':<25}")
        print("-" * 60)
        for k, metrics in total_latency_results.items():
            print(f"{k:<5} | {metrics['avg_total_latency']:.6f}")
            
        print("\n" + "#"*60)
        print(f"RAG ACCURACY vs. TopK (Cosine Similarity $\\ge$ {ACCURACY_THRESHOLD})")
        print("#"*60)
        print(f"{'TopK':<5} | {'Accuracy (Hit Rate)':<25} | {'Avg. Cosine Sim.':<20}")
        print("-" * 60)
        for k, metrics in accuracy_results.items():
            print(f"{k:<5} | {metrics['accuracy']:.4f} | {metrics['avg_similarity']:.4f}")
        print("-" * 60)


    def pipelines(self) -> List[dict]:
        return [
            {
                "id": "pgvector-ollama-rag-model",
                "name": f"PGVector RAG ({OLLAMA_MODEL} Local) ",
                "description": "RAG using PGVector (PostgreSQL) and a local Ollama model, logging per-file indexing time and running a full benchmark on startup."
            }
        ]

    def pipe(self, user_message, model_id, messages, body):
        
        if not self.pgvector_rag:
            yield "RAG is unavailable. Check pipeline server logs for Ollama, PGVector connection, or indexing errors during startup."
            return
            
        
        last_user_message = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), {})
        user_query = last_user_message.get('content', 'Default query: What is the main topic?')
        
         
        response_content = self.pgvector_rag.generate_completion(user_query)
        
        yield response_content