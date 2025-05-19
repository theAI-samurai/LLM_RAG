# pip install langchain openai tiktoken chromadb
"""
Name        :       Ankit Mishra
Description :       1. This code uses Advanced techniques,
                    2. OCI Embeddings
                    3.
"""

from langchain_community.document_loaders import PyPDFLoader
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# -----------   OCI LLM (cohere) + LLM invoke  -----------------
llm = OCILLM(model_name=MODEL_ID_CR)

#       ----------------------  METHOD 1    ---------------------------
#  ---------                 Langchain CHAT LOOP    -------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm.llm,            # llm s OCI cohere chat model
    chain_type="stuff",     # simplest approach where all retrieved documents are "stuffed" into the LLM context.
    retriever=retriever,
    return_source_documents=True
)

while True:
    query = input("YOU : ")
    if query.lower().__contains__("exit") or query.lower().__contains__("quit"):
        break
    result = qa_chain.invoke({"query": query})              # here we use invoke function
    print(f"Assistant: {result['result']}")
    print("Sources:", [doc.page_content for doc in result['source_documents']])


#       ----------------------  METHOD 2    ---------------------------
#  ---------                 Langchain CHAT LOOP    ------------------- (better use this)
while True:
    preamble = "You are a question and answer assistant."
    query = input("YOU : ")
    if query.lower().__contains__("exit") or query.lower().__contains__("quit"):
        break
    relevant_docs = retriever.get_relevant_documents(query)
    res = llm.generate_answer(documents=relevant_docs, preamble=preamble, prompt=query)
    print(res)



