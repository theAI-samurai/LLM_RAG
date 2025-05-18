# pip install langchain openai tiktoken chromadb
"""
Name        :       Ankit Mishra
Description :       1. This code uses just OCI LLM and generate answer func,
                    2. generate_answer takes Documents + preamble + prompt
                    3. Docs are just split data
                    4.  this however does not return the source of docs.
"""


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain.vectorstores import Chroma
from llm import *  # Assuming this contains your OCI LLM setup
from langchain.chains import RetrievalQA

from simple_OCIchat import prompt

OCI_EMBED_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# Load and split the document
loader = PyPDFLoader("basics-of-data-science-kpk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(pages)

# -----------   OCI LLM (cohere) + LLM invoke  -----------------
llm = OCILLM(model_name=MODEL_ID_CR)

preamble = f"""You are a Question and Answer Assistant"""
prompt = f""" what is Deep Learning?"""
res = llm.generate_answer(documents=splits, preamble=preamble, prompt=prompt)
print(res)

