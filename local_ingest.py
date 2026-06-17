#!/usr/bin/env python3
"""
Local Ingestion Pipeline using Hugging Face (all-MiniLM-L6-v2) or Ollama (nomic-embed-text) and Supabase pgvector.

This script:
1. Reads all downloaded PDF files from the `quantum_papers/` directory.
2. Extracts text from each PDF, removing NULL bytes.
3. Splits the text into chunks of 1000 characters (with 150 overlap).
4. Generates embeddings using:
   - Hugging Face (384 dimensions, local Python execution)
   - Ollama (768 dimensions, requires background Ollama server)
5. Uploads the chunks, metadata, and embeddings to the corresponding Supabase table:
   - `hf_documents` (for Hugging Face embeddings)
   - `local_documents` (for Ollama embeddings)
"""

import os
# Force Hugging Face cache to be inside the project folder to bypass Windows PermissionError
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, ".hf_cache")

import glob
import logging
import argparse
from dotenv import load_dotenv
from supabase import create_client, Client
from pypdf import PdfReader

# LangChain text splitter & embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("local_ingest")

# 1. Parse Arguments
parser = argparse.ArgumentParser(description="Local Ingestion Pipeline")
parser.add_argument(
    "--provider",
    choices=["huggingface", "ollama"],
    default="huggingface",
    help="Embedding provider to use (default: huggingface)"
)
args = parser.parse_args()

# 2. Load Environment Variables
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY", "")  # We need the secret key to WRITE to the DB

if not supabase_url or not supabase_key:
    logger.error("Database error: SUPABASE_URL or SUPABASE_SECRET_KEY is missing from your .env file.")
    exit(1)

# Sanitize URL if it ends with /rest/v1/ or /rest/v1
if supabase_url.endswith("/rest/v1/"):
    supabase_url = supabase_url[:-9]
elif supabase_url.endswith("/rest/v1"):
    supabase_url = supabase_url[:-8]

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

# Folder paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(SCRIPT_DIR, "quantum_papers")


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extracts all text content from a PDF file.
    """
    logger.info(f"Extracting text from: {os.path.basename(filepath)}")
    try:
        reader = PdfReader(filepath)
        text = ""
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Sanitize NULL bytes which PostgreSQL doesn't support
        text = text.replace('\x00', '').replace('\u0000', '')
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF {filepath}: {e}")
        return ""


def main():
    logger.info("=" * 60)
    logger.info(f"Starting Local Ingestion Pipeline using provider: {args.provider.upper()}")
    logger.info("=" * 60)

    # Initialize selected embedding provider and DB table
    if args.provider == "huggingface":
        logger.info("Initializing Hugging Face embeddings (model: all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db_table = "hf_documents"
    else:
        logger.info("Initializing Ollama embeddings (model: nomic-embed-text)...")
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        db_table = "local_documents"

        # Verify Ollama server connection before doing work
        try:
            embeddings.embed_query("connection check")
            logger.info("Successfully connected to local Ollama server.")
        except Exception as e:
            logger.critical(
                f"Could not connect to Ollama: {e}\n"
                "Please ensure Ollama is running and you pulled the model with 'ollama pull nomic-embed-text'."
            )
    # Clear existing rows to prevent duplicates
    logger.info(f"Clearing existing entries from '{db_table}' to prevent duplicates...")
    try:
        supabase.table(db_table).delete().neq("id", -1).execute()
        logger.info(f"Successfully cleared table '{db_table}'.")
    except Exception as e:
        logger.warning(f"Could not clear table '{db_table}' (this is normal if RLS blocks deletes or table is empty): {e}")

    # Find all downloaded PDF papers
    pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in: {PAPERS_DIR}. Please run your scraper first.")
        return

    logger.info(f"Found {len(pdf_files)} PDF papers to process.")

    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    total_chunks_inserted = 0

    for idx, filepath in enumerate(pdf_files, start=1):
        filename = os.path.basename(filepath)
        paper_title = os.path.splitext(filename)[0].replace("_", " ")

        logger.info("-" * 60)
        logger.info(f"Processing Paper {idx}/{len(pdf_files)}: {paper_title}")
        
        # 1. Extract raw text
        text = extract_text_from_pdf(filepath)
        if not text.strip():
            logger.warning(f"No text extracted for '{filename}'. Skipping.")
            continue

        # 2. Chunk text
        chunks = text_splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} text chunks.")

        # 3. Generate embeddings and upload in batches
        # We process in batches of 10 to manage local load
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = embeddings.embed_documents(batch_chunks)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                continue

            # Prepare db insertion rows
            rows = []
            for chunk_text, embedding in zip(batch_chunks, batch_embeddings):
                rows.append({
                    "content": chunk_text,
                    "metadata": {
                        "title": paper_title,
                        "filename": filename
                    },
                    "embedding": embedding
                })

            # Upload to Supabase
            try:
                supabase.table(db_table).insert(rows).execute()
                total_chunks_inserted += len(rows)
                logger.info(f"Uploaded chunks {i+1} to {i+len(rows)} successfully into '{db_table}'.")
            except Exception as e:
                logger.error(f"Failed to insert batch into Supabase table '{db_table}': {e}")

    logger.info("=" * 60)
    logger.info("Local Ingestion Summary:")
    logger.info(f"Total papers processed: {len(pdf_files)}")
    logger.info(f"Total chunks inserted into '{db_table}': {total_chunks_inserted}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
