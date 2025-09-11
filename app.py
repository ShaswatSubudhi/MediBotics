from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('Pinecone_API_Key')
GOOGLE_2_0_FLASH_API_KEY = os.getenv('GOOGLE_API_KEY')

os.environ["Pinecone_API_Key"] = PINECONE_API_KEY
os.environ["GOOGLE_2_0_FLASH_API_KEY"] = GOOGLE_2_0_FLASH_API_KEY

embedding = download_hugging_embeddings()

index_name = "medibotics"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_output_tokens=512)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
])

question_answer = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    msg = data.get("message")
    if not msg:
        return jsonify({"reply": "No input provided"}), 400
    response = rag_chain.invoke({"input": msg})
    return jsonify({"reply": response["answer"]})

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
