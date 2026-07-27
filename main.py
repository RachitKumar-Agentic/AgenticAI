print("before import\n")
import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
print("After import\n")

load_dotenv()

def convert_to_vector(filepath:str,chunking_strategry:str,max_char:int):
    print("loading the file\n")
    loader = UnstructuredLoader(filepath,chunking_strategry=chunking_strategry,max_characters=max_char)
    loaded_data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(loaded_data)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


    PineconeVectorStore.from_documents(documents,embeddings,index_name=os.environ.get("INDEX_NAME"))
    print(loaded_data)



def main():
    convert_to_vector("E:/AgenticAI-Udemy/rag-gist/mediumblog1.txt","basic",1000)

if __name__ == "__main__":
    main()
