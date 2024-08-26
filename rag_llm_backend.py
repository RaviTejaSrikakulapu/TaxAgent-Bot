import os
import nltk
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# nltk.download('punkt_tab')
os.environ["OPENAI_API_KEY"] = "api key"
# Path to the directory containing markdown files
markdown_directory = r"C:\Users\tejas\PycharmProjects\TaxAgent\mdfiles"


markdown_files = [
    os.path.join(markdown_directory, f)
    for f in os.listdir(markdown_directory)
    if f.endswith(".md")
]
print("Created a list of markdown files")


docs = [UnstructuredMarkdownLoader(f).load()[0] for f in markdown_files]
print("Loaded the Markdown files to doc variable")


text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print("Splitted the documents into chunks")


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(splits, embeddings)
print("Performed Embedding and stored in vector db")


retriever = vectorstore.as_retriever()
# llm = OpenAI(model="gpt-3.5-turbo")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)
def rag(user_question):
    retrieved_docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    formatted_prompt = f"Question: {user_question}\n\nContext: {context}"
    response = llm.invoke(formatted_prompt)
    return response.content

