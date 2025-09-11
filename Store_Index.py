from src.helper import load_file,process_documents,download_hugging_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('Pinecone_API_Key')
os.environ["Pinecone_API_Key"]=PINECONE_API_KEY

data_from_pdf=load_file(data='Data/')
chunks=process_documents(data_from_pdf)
embedding = download_hugging_embeddings()


pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="medibotics"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",
                            region="us-east-1"
            )
    )

docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embedding
)
