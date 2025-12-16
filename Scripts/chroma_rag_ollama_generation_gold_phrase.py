import os
import time
import textwrap
import numpy as np
import sys
from typing import List, Optional, Union, Iterator, Generator, Literal
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


try:
    from google import genai
    from google.genai.errors import APIError
    from openai import OpenAI
    
    
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    from langchain_community.vectorstores import Chroma  
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM   
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

     
    try:
        from blueprints.function_calling_blueprint import Pipeline as Blueprint
    except ImportError:
        print("Warning: 'blueprints.function_calling_blueprint' not found. Using simple class inheritance.")
        class Blueprint(object): pass 
    
    print("All necessary libraries imported successfully.")

except ImportError as e:
    print(f"Failed to import a library: {e}. Please ensure chromadb is installed.")
    sys.exit(1)  


GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA7R_sPB8khJOAjwbhYZv2WLx_AvYsasz")   
OLLAMA_MODEL = "mistral"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
DOCS_FOLDER = 'docs'
CHROMA_DB_DIR = 'chroma_db' 
ACCURACY_THRESHOLD = 0.8


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

        Based *only* on the provided context above, answer the question. If the context does not contain the answer, state that you cannot answer based on the provided information.
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



class ChromaRAG(object):  
    """
    A class to handle the RAG process using ChromaDB for document indexing and retrieval,
    and Ollama for generation.
    """

    def __init__(self, generation_llm: OllamaLLM):
        self.name="chroma-rag-model"
        self.generation_llm = generation_llm  
        self.vectorstore = None
        self.retriever = None
        
         
        try:
            print("Initiating ChromaDB RAG setup...")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            docs_path = os.path.join(script_dir, DOCS_FOLDER) 
            chroma_path = os.path.join(script_dir, CHROMA_DB_DIR)  
            self.docs = []
            self.total_docs_size_bytes = 0 
            
            if not os.path.exists(docs_path) or not os.listdir(docs_path):
                raise FileNotFoundError(f"No .txt files found in {docs_path}")
            
            start_time_ingest = time.time()
            for file in os.listdir(docs_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(docs_path, file)
                    loader = TextLoader(file_path)
                    self.docs.extend(loader.load())
                    self.total_docs_size_bytes += os.path.getsize(file_path) 
            
             
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            self.texts = self.text_splitter.split_documents(self.docs)
            
            end_time_ingest = time.time()
            self.ingest_time = end_time_ingest - start_time_ingest
            
             
            self.ingest_speed_gb_s = (self.total_docs_size_bytes / (1024**3)) / self.ingest_time if self.ingest_time > 0 else 0
            
            print(f"Total Indexed Documents (Chunks): {len(self.texts)}")
            print(f"Document Ingest Time: {self.ingest_time:.4f}s")
            
             
            start_time_indexing = time.time()
             
            self.vectorstore = Chroma.from_documents(
                documents=self.texts, 
                embedding=self.embeddings, 
                persist_directory=chroma_path  
            )
            self.vectorstore.persist()  
            end_time_indexing = time.time()
            self.indexing_time = end_time_indexing - start_time_indexing
            print(f"INDEXING TIME IS: {self.indexing_time:.4f} seconds")
            
             
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})  
            
        except Exception as e:
            print(f"Error during ChromaRAG initialization: {e}")
            
            if os.path.exists(chroma_path):
                import shutil
                shutil.rmtree(chroma_path)
            raise e

    def timed_rag_generation(self, query: str, top_k: int) -> (str, float, float):
        """
        Performs timed RAG generation using the internal Ollama LLM.
        
        :returns: (response_content, retrieval_latency, generation_latency)
        """
        if not self.retriever:
            return "Internal Error: ChromaDB retriever not set up.", 0.0, 0.0

         
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


    def generate_completion(self, messages: List[dict], model_id: str, stream: bool = False) -> str:
        """Standard pipeline method using the default k=10."""
        last_user_message = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), None)
        user_query = last_user_message.get('content', '') if last_user_message else ""
            
        if not user_query:
            return "Error: Could not find a valid user query."
            
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
        
        self.name="chroma-rag-model_2" 
        self.type = "manifold"
        self.chroma_rag = None  
        
        
        try:
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
            return 

        
        try:
            self.chroma_rag = ChromaRAG(self.ollama_llm)
            print("Pipeline instance created! RAG Setup Complete.")
            
           
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.chroma_rag = None  
            raise e
            
    def benchmark_rag(self, test_cases: List[TestCase], top_k_values: List[int], n_runs: int = 1):
        """
        Executes a benchmarking run, measuring retrieval, latency, and cosine similarity accuracy.
        """
        print("\n" + "="*60)
        print(f"STARTING RAG BENCHMARK (ChromaDB + Ollama/{OLLAMA_MODEL})")  
        print(f"Similarity Threshold: {ACCURACY_THRESHOLD}")
        print("="*60)
        
       
        if self.chroma_rag:
            print(f"Knowledge Base Size (Raw): {self.chroma_rag.total_docs_size_bytes / (1024**2):.2f} MB")
            print(f"Indexing Time: {self.chroma_rag.indexing_time:.4f}s")
        else:
            print("Cannot display metrics: ChromaRAG failed to initialize.")
            return

        print("-" * 60)
        
        retrieval_results = {}
        total_latency_results = {}
        accuracy_results = {}

       
        if not self.chroma_rag or not self.ollama_llm:
            print("Cannot run full benchmark: RAG components or Ollama LLM failed to initialize.")
            return
            
        embeddings_model = self.chroma_rag.embeddings  

        
        for k in top_k_values:
            print(f"Benchmarking k={k}...")
            total_retrieval_time = 0
            total_rag_latency = 0
            correct_answers = 0 
            total_similarity_score = 0.0 
            
            for test_case in test_cases:
                for _ in range(n_runs):
                    
                    final_answer, retrieval_latency, generation_latency = self.chroma_rag.timed_rag_generation(
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
                        
                     
                    total_retrieval_time += retrieval_latency
                    total_rag_latency += (retrieval_latency + generation_latency)
                        
                    

             
            total_runs = len(test_cases) * n_runs
            
            if total_runs > 0:
                avg_retrieval_time = total_retrieval_time / total_runs
                retrieval_qps = 1 / avg_retrieval_time
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
        """Defines the pipeline for Open WebUI."""
        return [
            {
                "id": "chroma-ollama-rag-model", 
                "name": f"Chroma RAG ({OLLAMA_MODEL} Local)", 
                "description": "ChromaDB RAG pipeline using a local Ollama model for generation." 
            }
        ]

    def pipe(self, user_message, model_id, messages, body):
        """The main execution method for the pipeline."""
        if not self.chroma_rag: 
            yield "RAG is unavailable. Check pipeline server logs for Ollama or ChromaDB errors during startup."
            return
            
         
        response_content = self.chroma_rag.generate_completion(messages=messages, model_id=model_id, stream=False)
        yield response_content