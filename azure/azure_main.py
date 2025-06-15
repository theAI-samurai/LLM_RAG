
"""
# pip install azure-ai-documentintelligence==1.0.0
https://azuresdkdocs.z19.web.core.windows.net/python/azure-ai-documentintelligence/latest/index.html
"""

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeOutputOption, AnalyzeResult
import chromadb
from chromadb.utils import embedding_functions
import openai
from typing import List, Dict, Tuple, Optional
import tiktoken
import os


endpoint = "https://ai-rc-cmn-poc.cognitiveservices.azure.com/"
key = "azure key "
openai_key = "openaiKEY",  # Optional
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"      # "gpt-3.5-turbo"  # or "gpt-4"
CHUNK_SIZE = 64  # Number of tokens per chunk
CHUNK_OVERLAP = 8  # Overlap between chunks
TEMPERATURE = 0.7
MAX_TOKENS = 1000


class DocumentProcessor():
    def __init__(self):
        self.azure_di_client = DocumentIntelligenceClient(endpoint=endpoint,
                                                          credential=AzureKeyCredential(key)
                                                     )
        self.chromadb_client = chromadb.PersistentClient(path="chroma_db")
        self.embedding = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key[0],
                                                                     model_name=EMBEDDING_MODEL
                                                                     )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.openai_client = openai.OpenAI(api_key=openai_key[0])
        self.chat_history = []

    def extract_text_from_pdf(self, pdf_path: str) -> AnalyzeResult:
        """Extract text from PDF using Azure Document Intelligence"""
        with open(pdf_path, "rb") as f:
            poller = self.azure_di_client.begin_analyze_document("prebuilt-read",
                                                                 body=f,
                                                                 output=[AnalyzeOutputOption.PDF])
        result = poller.result()
        return result

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if end == len(tokens):
                break
            start = end - overlap

        return chunks

    def process_document(self, pdf_path: str, collection_name: str = "documents") -> Dict:
        """Process PDF document and store in ChromaDB"""
        # Extract text from PDF
        result = self.extract_text_from_pdf(pdf_path)

        # Prepare data for storage
        documents = []
        metadatas = []
        ids = []

        for pg, page in enumerate(result.pages, start=1):
            page_text = ""
            for line in page.lines:
                page_text += line.content + "\n"

            # Chunk the page text
            chunks = self.chunk_text(page_text)

            for i, chunk in enumerate(chunks):
                doc_id = f"page_{pg}_chunk_{i}"
                documents.append(chunk)
                metadatas.append({
                    "page_number": pg,
                    "chunk_number": i,
                    "source": os.path.basename(pdf_path)
                })
                ids.append(doc_id)

        # Create or get collection
        collection = self.chromadb_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding
        )

        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return {
            "total_pages": len(result.pages),
            "total_chunks": len(documents),
            "collection": collection_name
        }

    def query_document(self, query: str, collection_name: str = "documents", n_results: int = 3) -> List[Dict]:
        """Query the document collection"""
        collection = self.chromadb_client.get_collection(
            name=collection_name,
            embedding_function=self.embedding
        )

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "page_number": results['metadatas'][0][i]['page_number'],
                "chunk_number": results['metadatas'][0][i]['chunk_number'],
                "distance": results['distances'][0][i]
            })

        return formatted_results

    def get_page_context(self, page_number: int, collection_name: str = "documents") -> str:
        """Get all chunks from a specific page"""
        collection = self.chromadb_client.get_collection(
            name=collection_name,
            embedding_function=self.embedding
        )

        results = collection.get(
            where={"page_number": page_number}
        )

        # Sort chunks by chunk number
        chunks = sorted(zip(results['metadatas'], results['documents']), key=lambda x: x[0]['chunk_number'])

        return " ".join([chunk[1] for chunk in chunks])

    def generate_response(self, query: str, context: List[Dict], chat_history: Optional[List] = None) -> Dict:
        """Generate a chat response using OpenAI with document context"""
        if chat_history is None:
            chat_history = self.chat_history

        # Prepare context string
        context_str = "\n\n".join([
            f"Source: {c['chunk_number']}, Page {c['page_number']}:\n{c['content']}"
            for c in context])

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based on the provided documents. "
                        "Always cite your sources by mentioning the page number. If you don't know the answer, say so."}
        ]
        messages.extend(chat_history)
        messages.append({
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context_str}"
        })

        # Call OpenAI API
        response = self.openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS)

        answer = response.choices[0].message.content

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": [{"page": c["page_number"], "source": c["chunk_number"]} for c in context],
            "full_context": context_str
        }

    def chat(self, query: str, collection_name: str = "documents") -> Dict:
        """Complete chat workflow: retrieve + generate"""
        # Retrieve relevant content
        context = self.query_document(query, collection_name)

        # Generate response
        response = self.generate_response(query, context)

        return response

    def clear_chat_history(self):
        """Clear the conversation history"""
        self.chat_history = []


processor = DocumentProcessor()

# # Process a document (only need to do this once per document)
# pdf_path = "/home/ankit/Desktop/git_personal/rovicare/medicare form.pdf"
# processing_result = processor.process_document(pdf_path)
# print(f"Processed {processing_result['total_pages']} pages into {processing_result['total_chunks']} chunks.")
#
# # Query the document
# query = "Member name"
# results = processor.query_document(query)
#
# print(f"\nResults for query: '{query}'")
# for result in results:
#     print(f"\n\t--------Page {result['page_number']}, Chunk {result['chunk_number']}:")
#     print(result['content'])
#     print(f"Similarity distance: {result['distance']}")
#
#     # Get full page context if needed
#     full_page = processor.get_page_context(result['page_number'])
#     print(f"\nFull page {result['page_number']} context (first 500 chars):")
#     print(full_page[:500] + "...")


# --------- Method  2 : CHAT HISTORY -------------
pdf_path = "/home/ankit/Desktop/git_personal/rovicare/medicare form.pdf"
if not os.path.exists("chroma_db"):
    print("Processing document...")
    processing_result = processor.process_document(pdf_path)
    print(f"Processed {processing_result['total_pages']} pages into {processing_result['total_chunks']} chunks.")

# Chat interface
print("\nDocument Chat System - Type 'exit' to quit")
while True:
    query = input("\nYour question: ")
    if query.lower() in ['exit', 'quit']:
        break

    response = processor.chat(query)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    for source in response["sources"]:
        print(f"- Page {source['page']} of {source['source']}")

    # Optional: Show context for debugging
    # print("\nRetrieved Context:")
    # print(response["full_context"])