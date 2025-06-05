from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from fastapi_cache import FastAPICache
from process_papers import PaperProcessor
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from typing import List, Dict, Optional
import logging
import time
import threading

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor variable
processor = None
initialization_status = {"status": "not_started", "message": "Initialization not started"}
initialization_lock = threading.Lock()

class PaperMetadata(BaseModel):
    title: str
    authors: str
    year: str
    journal: str
    volume: str
    issue: str
    pages: str
    doi: str
    SRID: str
    abstract: str
    text: str = Field("", description="Relevant text chunk from the paper")
    source: str

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    papers: List[PaperMetadata] = Field(..., description="Primary relevant papers with full metadata and text")
    similar_papers: List[PaperMetadata] = Field([], description="Additional similar papers with full metadata and text")

class StatusResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int = 0
    index_ready: bool = False

def initialize_processor_background():
    """Initialize the paper processor in the background"""
    global processor, initialization_status
    
    try:
        with initialization_lock:
            initialization_status = {"status": "in_progress", "message": "Loading models and initializing processor..."}
        
        # Create processor instance
        start_time = time.time()
        proc = PaperProcessor()
        
        # Process documents
        initialization_status = {"status": "in_progress", "message": "Processing papers from S3..."}
        try:
            proc.process_s3_folder("research_papers/")
        except Exception as e:
            print(f"Error processing S3 folder: {e}")
            raise
         # Update global processor only when fully initialized
        with initialization_lock:
            global processor
            processor = proc
            elapsed = time.time() - start_time
            initialization_status = {
                "status": "completed", 
                "message": f"Initialization complete in {elapsed:.2f} seconds",
                "documents_loaded": len(proc.documents),
                "index_ready": proc.index is not None,
                "tfidf_ready": proc.tfidf_matrix is not None
            }
            logger.info(f"Processor initialization completed: {len(proc.documents)} documents loaded")
            
    except Exception as e:
        logger.error(f"Error in background initialization: {str(e)}")
        with initialization_lock:
            initialization_status = {"status": "failed", "message": f"Initialization failed: {str(e)}"}

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())
    # Start initialization in background
    background_thread = threading.Thread(target=initialize_processor_background)
    background_thread.daemon = True
    background_thread.start()
    logger.info("Started background initialization thread")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current initialization status"""
    global processor, initialization_status
    
    with initialization_lock:
        status = initialization_status.copy()
    
    if processor is not None:
        status["documents_loaded"] = len(processor.documents)
        status["index_ready"] = processor.index is not None
    
    return status

@app.post("/ask", response_model=AnswerResponse)
@cache(expire=300)  # Cache for 5 minutes
async def ask_question(request: QuestionRequest):
    global processor
    
    # Check if processor is ready
    if processor is None:
        with initialization_lock:
            status = initialization_status["status"]
            message = initialization_status["message"]
        
        if status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Research paper database initialization failed: {message}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Research paper database is still initializing. Please try again in a moment."
            )
    
    # Process the question and validate it's not too short
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Log the incoming question
    logger.info(f"Processing question: '{question}'")
    
    # Check if query is very short (1-2 words)
    words = question.split()
    is_short_query = len(words) <= 2
    
    try:
        # Use more results for short queries to increase chances of relevance
        k = 5 if is_short_query else 3
        results = processor.query_papers(question, k)
        
        if not results:
            return AnswerResponse(
                answer=f"No relevant research papers found matching your query for '{question}'.",
                papers=[],
                similar_papers=[]
            )
        
        # For very short queries, analyze result relevance
        relevant_results = []
        if is_short_query:
            query_terms = set(question.lower().split())
            for result in results:
                text = result.get('text', '').lower()
                title = result.get('title', '').lower()
                abstract = result.get('abstract', '').lower()
                references = result.get('references', '').lower()
                
                # Check if any query term appears in the document
                if any(term in text for term in query_terms) or \
                   any(term in title for term in query_terms) or \
                   any(term in abstract for term in query_terms) or \
                   any(term in references for term in query_terms):
                    relevant_results.append(result)
                    
            # If we found relevant results, use only those
            if relevant_results:
                logger.info(f"Found {len(relevant_results)} directly relevant results for '{question}'")
                results = relevant_results
                
        context = [r['text'] for r in results]
        answer = processor.generate_answer(question, context)
        
        # Restructure the response to have papers and their text together
        # The primary paper is the first one
        primary_paper = None
        similar_papers = []
        
        if results:
            # First result becomes primary paper
            primary = results[0]
            primary_paper = PaperMetadata(
                title=primary.get('title', ''),
                authors=primary.get('authors', 'Unknown Authors'),
                year=primary.get('year', ''),
                journal=primary.get('journal', ''),
                volume=primary.get('volume', ''),
                issue=primary.get('issue', ''),
                pages=primary.get('pages', ''),
                doi=primary.get('doi', ''),
                # Ensure SRID is definitely not empty
                SRID=primary.get('SRID', '') or f"SRID{abs(hash(primary.get('title', ''))) % 100000000:08d}",
                abstract=primary.get('abstract', ''),
                text=primary.get('text', ''),
                source=primary.get('source', '')
            )
            # Remaining results become similar papers
            for r in results[1:]:
                similar_papers.append(PaperMetadata(
                    title=r.get('title', ''),
                    authors=r.get('authors', 'Unknown Authors'),
                    year=r.get('year', ''),
                    journal=r.get('journal', ''),
                    volume=r.get('volume', ''),
                    issue=r.get('issue', ''),
                    pages=r.get('pages', ''),
                    doi=r.get('doi', ''),
                    # Ensure SRID is definitely not empty
                    SRID=r.get('SRID', '') or f"SRID{abs(hash(r.get('title', ''))) % 100000000:08d}",
                    abstract=r.get('abstract', ''),
                    text=r.get('text', ''),
                    source=r.get('source', '')
                ))
        else:
            primary_paper = PaperMetadata(
                title="",
                authors="",
                year="",
                journal="",
                volume="",
                issue="",
                pages="",
                doi="",
                SRID="",
                abstract="",
                text="",
                source=""
            )
            
        # Log the completion of processing
        logger.info(f"Completed processing query: '{question}' with {len(results)} results")
            
        return AnswerResponse(
            answer=answer,
            papers=[primary_paper] if primary_paper else [],
            similar_papers=similar_papers
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
@app.post("/clear_cache")
async def clear_cache():
    """Clear all cache files to force fresh processing"""
    cache_files = [
        "cache/faiss_index.pkl",
        "cache/documents_cache.pkl", 
        "cache/tfidf_vectorizer.pkl",
        "cache/tfidf_matrix.npz"
    ]
    
    deleted = 0
    for file in cache_files:
        try:
            os.remove(file)
            deleted += 1
        except:
            pass
            
    return {"status": f"Deleted {deleted} cache files"}