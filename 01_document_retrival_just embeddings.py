# pip install langchain openai tiktoken chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.vectorstores import Chroma
from llm import *
from langchain.chains import RetrievalQA

OCI_EMBED_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

loader = PyPDFLoader("basics-of-data-science-kpk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
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
# --------------------------------------------------------------------
# # ------------      EMBEDDING DICT     -------------------------------
# embedding_dict = {}
# for i, split in enumerate(splits):
#     text = split.page_content
#     embedding = embeddings.embed_query(text)
#     embedding_dict[i] = {"embedding": embedding}

# ----------   VECTOR STORE + EMBEDDING     ------ Choma DB ----------
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# ----------    Retriver chroma DB  ------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


#   --------------  Query the DOCS  ----------------------------------
# This retrives the data
query = "What is the main topic of this document?"
res_docs = retriever.get_relevant_documents(query)
for doc in res_docs:
    print(doc.page_content)




