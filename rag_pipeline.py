from langchain_core.documents.base import Document


from langchain_core.documents import Document # imports the Document type
from langchain_core.prompts import ChatPromptTemplate # imports the ChatPromptTemplate to build the prompt template
from langchain_text_splitters import RecursiveCharacterTextSplitter # imports the RecursiveCharacterTextSplitter to split the text into chunks
from langchain_huggingface import HuggingFaceEmbeddings # imports the HuggingFaceEmbeddings to embed the text
from langchain_chroma import Chroma # imports the Chroma to store the text chunks
from langchain_google_genai import ChatGoogleGenerativeAI # imports the ChatGoogleGenerativeAI to use the Google Generative AI API

import hashlib
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# ==============================================
# INGESTION AND STABLE CHUNKING
# ==============================================

def load_and_split(source_id: str, text: str, chunk_size: int=400, overlap: int=50) -> list[Document]:
    """
    Split raw text into chunks with deterministic chunk IDs and structured metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    raw_chunks = splitter.create_documents([text])

    docs = []

    for i, chunk in enumerate(raw_chunks):
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest() # hash the content to get a unique ID
        chunk_id = f"{source_id}::chunk-{i:05d}::{content_hash[:16]}" # create chunk ID from source ID and chunk index
        chunk.metadata.update({"source_id": source_id, "chunk_id": chunk_id}) # update and store source ID and chunk ID in metadata
        docs.append(chunk) # add the chunk to the list of documents

    return docs # return the list of documents


# Initialize persistent vector store with embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create the Chroma vector store
vectorstore = Chroma(
    collection_name="internal_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma_store"
) 

def ingest(documents: list[Document]) -> None:
    """
    Embed and store documents in the vector store.
    """
    vectorstore.add_documents(documents, ids=[d.metadata["chunk_id"] for d in documents])



# ==============================================
# RETRIEVAL AND LOGGING
# ==============================================

def retrieve(question: str, k: int=5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k chunks with metadata and similarity scores.
    """
    results = vectorstore.similarity_search_with_score(question, k=k, filter=filters)

    chunks = []

    for doc, score in results:
        chunks.append({
            "chunk_id": doc.metadata["chunk_id"],
            "source": doc.metadata["source_id"],
            "section": doc.metadata.get("section"),
            "score": float(score),
            "text": doc.page_content
        })

    return chunks # return the list of chunks


def log_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Print the retrieved chunk metadata and content previews for debugging.
    """
    for i, c in enumerate(chunks, start=1): # enumerate the chunks with index starting from 1
        preview = " ".join(c["text"].split())[:120] # preview the text with up to 120 characters
        print(f"[{i}] chunk_id={c['chunk_id']}, source={c['source']}, score={c['score']:.4f}")
        print(f"    {preview}")


# ==============================================
# STRUCTURED GENERATION AND VALIDATION
# ==============================================

class RAGResponse(BaseModel):
    """Structured output schema forcing exact chunk citations."""
    answer: str = Field(description="The answer to the user question based strictly on the provided context.")
    citations: List[str] = Field(description="The list of exact chunk IDs that were used to answer the question, in order of relevance.")


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a clearly delimited context block with citation markers."""
    formatted_chunks = []
    for c in chunks:
        formatted_chunks.append(
            f"--- CHUNK ID: {c['chunk_id']} | SOURCE: {c['source']} ---\n{c['text']}"
        )
    return "\n\n".join(formatted_chunks)


def validate_citations(response: RAGResponse, retrieved_chunks: List[Dict[str, Any]]) -> bool:
    """Enforce ground truth by ensuring all cited IDs exist in the retrieved chunk set."""
    valid_ids = {c["chunk_id"] for c in retrieved_chunks}
    cited_ids = set(response.citations)

    if not cited_ids:
        print("\n[VALIDATION FAILED] Response provided no citations.")
        return False

    invalid_citations = cited_ids - valid_ids
    if invalid_citations:
        print(f"\n[VALIDATION FAILED] Model cited unretrieved chunk IDs: {invalid_citations}")
        return False

    return True


def answer(question: str, k: int=2, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full RAD execution flow: Retrieve -> Prompt -> Generate -> Validate."""
    #1. Retrieve relevant chunks
    chunks = retrieve(question, k, filters)
    print("\n---- [RETRIEVAL] Retrieved chunks ----")
    log_chunks(chunks)

    if not chunks:
        return {"answer": "No relevant context found.", "citations": [], "validated": False}

    #2. Build context and prompt
    context_str = build_context_block(chunks)

    prompt = ChatPromptTemplate.from_messages([
         ("system", 
         "You are a helpful assistant. Answer the question using ONLY the context provided below. "
         "For every claim you make, you MUST cite the exact chunk_id from which it was derived.\n\n"
         "CONTEXT:\n{context}"),
        ("human", "{question}")
    ])

    #3. Model generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        api_key=API_KEY
    )
    structured_llm = llm.with_structured_output(RAGResponse) # wrap the LLM with the structured output

    formatted_prompt = prompt.format_messages(context=context_str, question=question)
    response: RAGResponse = structured_llm.invoke(formatted_prompt)

    #4. Validate grounding
    is_valid = validate_citations(response, chunks)

    return {
        "answer": response.answer,
        "citations": response.citations,
        "validated": is_valid
    }


# ==============================================
# DEMO PIPELINE
# ==============================================

if __name__ == "__main__":
    # Expanded Multi-Topic Documents
    doc1_text = """
    The Alpha-1 Engine requires high-octane synthetic coolant for peak performance. 
    Routine maintenance must be performed every 500 operating hours to prevent internal wear.
    Operating the engine above 95 degrees Celsius triggers an automatic thermal shutdown sequence.
    """
    
    doc2_text = """
    All employees entering Building B must scan their security badge at the main turnstile.
    Visitors are required to register at the front desk and wear a yellow badge at all times.
    The cafeteria serves lunch daily from 11:30 AM to 2:00 PM on the second floor.
    """

    doc3_text = """
    Production database backups execute automatically every midnight UTC and are retained for 30 days.
    Deployments to the staging server must be approved by the lead DevOps engineer.
    All API keys must be rotated every 90 days according to company security policy.
    """

    print("\n---- [INGESTION] Adding documents ----")
    all_chunks = []
    all_chunks.extend(load_and_split(source_id="manuals/engine_v1.pdf", text=doc1_text))
    all_chunks.extend(load_and_split(source_id="policy/building_access.txt", text=doc2_text))
    all_chunks.extend(load_and_split(source_id="devops/deployment_guide.txt", text=doc3_text))
    
    ingest(all_chunks)

    # Query specific to Document 1
    query = "What is the routine maintenance schedule for the Alpha-1 Engine?"
    print(f"\nProcessing query: {query}")
    result = answer(query, k=2)

    print(f"\n---- FINAL PIPELINE RESULT ----")
    print(f"Answer: {result['answer']}")
    print(f"Citations: {result['citations']}")
    print(f"Validated: {'Yes' if result['validated'] else 'No'}")