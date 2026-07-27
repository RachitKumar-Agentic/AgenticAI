print("before import\n")
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
print("After import\n")


load_dotenv()

def loading():
    print("loading\n")
    unstructure_loader = UnstructuredLoader("E:/AgenticAI-Udemy/rag-gist/mediumblog1.txt",chunking_strategy="basic",max_characters=100000)
    loader = unstructure_loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    document_splitted = splitter.split_documents(loader)
    embeddings_object = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)
    vector = Chroma.from_documents(document_splitted,embeddings_object,persist_directory="./chroma_db",collection_name="medium_articles")
    datasearch=vector.search("What is a Vector?","similarity")
    print(unstructure_loader)

if __name__ == "__main__":
    print("7. Main started")
    print(os.environ.get("PINECONE_API_KEY"))
    loading()
