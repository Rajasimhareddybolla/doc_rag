
import getpass
import os
from dotenv import load_dotenv

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import GithubFileLoader
from langchain_chroma import Chroma
load_dotenv()
github = os.environ.get["github"]
google_api_key = os.environ.get("GOOGLE_API_KEY")

embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
def load_github(repo  , persistent_dir = None):    
    if persistent_dir == None:
        persistent_dir = os.path.join(os.path.curdir , f"github_embeddings_{repo}")
    loader = GithubFileLoader(
        repo=repo,  # the repo name
        github_api_url="https://api.github.com",
        access_token=github,
    file_filter=lambda file_path: file_path.endswith('.md'),
    )
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = splitter.split_documents(docs)
    os.makedirs("txt/"+repo)
    with open(f"txt/{repo}.txt" , "w") as file:
        for document in text:
            file.write(str(document))
    filtered_text = filter_complex_metadata(text)
    db = Chroma.from_documents(filtered_text , embedding , persist_directory=persistent_dir)
    return db
def create_llm(query , repo ):
    persistent_dir = os.path.join(os.path.curdir , f"github_embeddings_{repo}")
    if not os.path.exists(persistent_dir):
        load_github(repo )    
    db =  Chroma(persist_directory=persistent_dir ,embedding_function= embedding)
    retriver = db.as_retriever(  search_type="similarity",k = 4)
    relevent_docs = retriver.invoke(query)
    content = ''.join([word.page_content for word in relevent_docs])
    template = f"""
    Query: {query}

    GitHub Documents Summary:
    {content}
    Source:
        gave me the metadata as well to cross verify and say dont no if you dont know properly .and be the answer to be concise.
            """
    message = [
        SystemMessage("you are a helpful and truthfull ai agent and my friend"),
        HumanMessage(template)
    ]
    
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
    response = llm.invoke(message)
    return response.content

