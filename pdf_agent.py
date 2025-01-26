from typing import Generator, List, Dict, Optional
import os
from pathlib import Path
import asyncio
import hashlib
from datetime import datetime
import json

from pypdf import PdfReader
from openai import AsyncOpenAI
from supabase import create_client
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from tqdm import tqdm

class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    content: str
    page_number: int
    chunk_index: int
    heading: Optional[str] = None
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict = Field(default_factory=dict)

class Document(BaseModel):
    """Represents a processed document with its chunks."""
    filename: str
    file_hash: str
    chunks: List[DocumentChunk]
    total_chunks: int
    processed_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)

class PDFEmbeddingAgent:
    """Agent for processing PDFs and storing their embeddings in Supabase."""
    
    def __init__(
        self,
        openai_api_key: str,
        supabase_url: str,
        supabase_key: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        table_name: str = "book_embeddings"
    ):
        """Initialize the PDF Embedding Agent."""
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_key)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_name = table_name
        self.tokenizer = tiktoken.encoding_for_model(embedding_model)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _extract_text_from_pdf(self, file_path: str) -> Generator[tuple[int, str], None, None]:
        """Extract text content from a PDF file."""
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                yield page_num, page.extract_text()

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        i = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.chunk_size]
            # Decode chunk tokens back to text
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
            # Move forward by chunk_size - overlap
            i += (self.chunk_size - self.chunk_overlap)

        return chunks

    def _split_text_with_langchain(self, text, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

    # def _chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
    #     """Split text into chunks, respecting code blocks and paragraphs."""
    #     chunks = []
    #     start = 0
    #     text_length = len(text)
    #
    #     while start < text_length:
    #         # Calculate end position
    #         end = start + chunk_size
    #
    #         # If we're at the end of the text, just take what's left
    #         if end >= text_length:
    #             chunks.append(text[start:].strip())
    #             break
    #
    #         # Try to find a code block boundary first (```)
    #         chunk = text[start:end]
    #         code_block = chunk.rfind('```')
    #         if code_block != -1 and code_block > chunk_size * 0.3:
    #             end = start + code_block
    #
    #         # If no code block, try to break at a paragraph
    #         elif '\n\n' in chunk:
    #             # Find the last paragraph break
    #             last_break = chunk.rfind('\n\n')
    #             if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
    #                 end = start + last_break
    #
    #         # If no paragraph break, try to break at a sentence
    #         elif '. ' in chunk:
    #             # Find the last sentence break
    #             last_period = chunk.rfind('. ')
    #             if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
    #                 end = start + last_period + 1
    #
    #         # Extract chunk and clean it up
    #         chunk = text[start:end].strip()
    #         if chunk:
    #             chunks.append(chunk)
    #
    #         # Move start position for next chunk
    #         start = max(start + 1, end)
    #
    #     return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks."""
        embeddings = []
        
        try:
            for text in texts:
                response = await self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embeddings.append(response.data[0].embedding)
            
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    async def _store_in_supabase(self, document: Document) -> None:
        """Store document chunks and their embeddings in Supabase."""
        try:
            # Prepare data for insertion
            rows = []
            for chunk in document.chunks:
                row = {
                    "file_hash": document.file_hash,
                    "filename": document.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "heading": chunk.heading,
                    "summary": chunk.summary,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "metadata": {
                        **document.metadata,
                        **chunk.metadata,
                        "processed_date": document.processed_date.isoformat(),
                    }
                }
                rows.append(row)

            # Insert in batches to avoid request size limits
            batch_size = 100
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                await asyncio.to_thread(
                    lambda: self.supabase_client.table(self.table_name)
                    .upsert(batch)
                    .execute()
                )

        except Exception as e:
            raise Exception(f"Error storing in Supabase: {str(e)}")
        
    async def get_heading_and_summary(self, chunk: str) -> Dict[str, str]:
        """Extract title and summary using GPT-4."""
        system_prompt = """You are an AI that extracts headings and summaries from documentation chunks.
            Return a JSON object with 'heading' and 'summary' keys.
            For the heading: If this seems like the start of a document, extract its heading. If it's a middle chunk, derive a descriptive title.
            For the summary: Create a concise summary of the main points in this chunk.
            Keep both heading and summary concise but informative.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting heading and summary: {e}")
            return {"title": "Error processing title", "summary": "Error processing summary"}
        
    async def process_pdf(self, file_path: str, metadata: Dict = None) -> Document:
        """Process a PDF file and store its embeddings."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            # Check if file already processed
            existing_docs = self.supabase_client.table(self.table_name)\
                .select("file_hash, chunk_index, page_number")\
                .eq("file_hash", file_hash)\
                .execute()
            
            # if existing_docs.data:
            #     print(f"File {file_path} already processed. Skipping...")
            #     return None

            # Extract text from PDF
            # text = self._extract_text_from_pdf(file_path)

            # Split into chunks
            # chunks = self._chunk_text(text)
            for page_number, text in self._extract_text_from_pdf(file_path):
                if len(text) == 1:
                    continue
                # uncomment for test
                # if page_number == 11:
                #     break
                chunks = self._split_text_with_langchain(text, self.chunk_size, self.chunk_overlap)

                # Create document chunks
                document_chunks = []
                for i, chunk_text in enumerate(chunks):
                    found = False
                    for data in existing_docs.data:
                        if data["chunk_index"] == i and data["page_number"] == page_number:
                            print(f"File {file_path}, chunk {i} and page {page_number}  already processed. Skipping...")
                            found = True
                            break
                    if found:
                        continue
                    # # Check if file already processed
                    # existing_docs = self.supabase_client.table(self.table_name)\
                    #     .select("file_hash")\
                    #     .eq("file_hash", file_hash)\
                    #     .eq("chunk_index", i)\
                    #     .eq("page_number", page_number)\
                    #     .execute()
                    
                    # if existing_docs.data:
                    #     print(f"File {file_path} already processed. Skipping...")
                    #     return None

                    # Get title and summary 
                    extracted = await self.get_heading_and_summary(chunk_text)
                    document_chunks.append(
                        DocumentChunk(
                            page_number=page_number,
                            content=chunk_text,
                            heading=extracted.get("heading"),
                            summary=extracted.get("summary"),
                            chunk_index=i,
                            metadata={
                                "chunk_size": self.chunk_size,
                                "chunk_overlap": self.chunk_overlap
                            }
                        )
                    )

                if len(document_chunks) == 0:
                    print(f"No content found in page {page_number}. Skipping...")
                    continue
                # Generate embeddings for all chunks
                print("Generating embeddings...")
                embeddings = await self._generate_embeddings([chunk.content for chunk in document_chunks])

                # Assign embeddings to chunks
                for chunk, embedding in zip(document_chunks, embeddings):
                    chunk.embedding = embedding

                # Create document
                document = Document(
                    filename=Path(file_path).name,
                    file_hash=file_hash,
                    chunks=document_chunks,
                    total_chunks=len(document_chunks),
                    processed_date=datetime.now().isoformat(),
                    metadata=metadata or {}
                )

                # Store in Supabase
                print("Storing in Supabase...")
                await self._store_in_supabase(document)

            return document or None

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise

async def process_directory(
    directory_path: str,
    openai_api_key: str,
    supabase_url: str,
    supabase_key: str,
    metadata: Dict = None
) -> None:
    """Process all PDF files in a directory."""
    agent = PDFEmbeddingAgent(
        openai_api_key=openai_api_key,
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )

    pdf_files = list(Path(directory_path).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            await agent.process_pdf(str(pdf_file), metadata)
            print(f"Successfully processed {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # Directory containing PDF files
    PDF_DIR = "data/pdfs/books"

    # Optional metadata for all documents
    METADATA = {
        "source": "company_finance_documents",
        "processed_by": "pdf_embedding_agent",
        "tag": ["finance", "trading"]
    }

    # Run the processing
    asyncio.run(process_directory(
        PDF_DIR,
        OPENAI_API_KEY,
        SUPABASE_URL,
        SUPABASE_KEY,
        METADATA
    ))
