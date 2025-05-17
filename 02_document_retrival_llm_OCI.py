# pip install langchain openai tiktoken chromadb

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.vectorstores import Chroma
from llm import *  # Assuming this contains your OCI LLM setup
from langchain.chains import RetrievalQA

OCI_EMBED_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# Load and split the document
loader = PyPDFLoader("basics-of-data-science-kpk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(pages)

# ----------    CREATING EMBEDDING MODEL FROM ORACLE    --------------
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=OCI_EMBED_ENDPOINT,
    truncate="NONE",
    compartment_id=COMPARTMENT_ID,
    auth_type=AUTH_TYPE,
    auth_profile=OCI_PROFILE
)

# ----------   VECTOR STORE + EMBEDDING     ------ Choma DB ----------
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# ----------    Retriver chroma DB  ------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------   OCI LLM (cohere) + retrival QA chain  -----------------
llm = OCILLM(model_name=MODEL_ID_CR)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm.llm,            # llm s OCI cohere chat model
    chain_type="stuff",     # simplest approach where all retrieved documents are "stuffed" into the LLM context.
    retriever=retriever,
    return_source_documents=True
)

#   --------------  Query the DOCS  ----------------------------------
query = "What is the main topic of this document?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("\nSources:")
for doc in result['source_documents']:
    print(doc.page_content)
    print("-----")
