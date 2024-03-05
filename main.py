# Importing Necessary Libraries
import openai
import langchain
import pinecone   # Vector db
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter   # Convert text into chunks(limited token size)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as PineconeVectorStore
import os
from pinecone import Pinecone, ServerlessSpec

# Load all the environment variables
load_dotenv()

# Read the document
def read_document(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

# converting the docs into chunks (Limited token size)
def chunk_data(docs,chunk_size=1000,chunk_overlap=50):
    text_spiltter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_spiltter.split_documents(docs)
    return doc

doc=read_document('/Users/pulkitrajput/PycharmProjects/MCQ-generator/Data')  # Reading the data
documents=chunk_data(docs=doc)                  # Converting the data into chunks


# Converting text into embeddings
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

# Vector search db in pinecone
index_name="langchainvector"
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
if 'langchainvector' not in pc.list_indexes().names():
    pc.create_index(
        name='langchainvector',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp-starter',
            region='Iowa (us-central1)'
        )
    )
# Storing the embeddings in pinecone
index=PineconeVectorStore.from_documents(documents,embeddings,index_name=index_name)

# Cosine Similarity retrieve results
def retrieve_query(query,k=2):
    result=index.similarity_search(query,k=k)
    return result


# Search answer from vector db
def retrieve_answer(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response

llm=OpenAI(model_name='gpt-3.5-turbo-instruct',temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

# Our Query
our_q="Tell me something about Momentum for Nari Shakti? "
answer=retrieve_answer(our_q)
print(answer)






