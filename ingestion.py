import os
from dotenv import load_dotenv
print("before import")
from langchain_unstructured import UnstructuredLoader
print("after import")

from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_docling.loader import DoclingLoader
from langchain_pinecone import PineconeVectorStore


from pinecone import Pinecone
def fetch_pinecone():

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["INDEX_NAME"])

    result = index.fetch(
        ids=["040102d2-0fa5-4800-ba1f-7bd3a929d4a7"]
    )

    print(result)


print("1. Starting")

load_dotenv()
print("2. dotenv loaded")

from langchain_unstructured import UnstructuredLoader
print("3. Imported UnstructuredLoader")

def load_document(file_path, chunking, max_char):
    print("4. Creating loader")
    data = UnstructuredLoader(
        file_path,
        chunking_strategy=chunking,
        max_characters=max_char,
    )

    print("5. Calling load()")
    document = data.load()

    print("6. Finished loading")
    #document[0].page_content)
    textsplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
    texts=textsplitter.split_documents(document)
    print(f"lenght of chunks : {len(texts)}")
    embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview", GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY"),output_dimensionality=768,
            )
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ.get("INDEX_NAME")) 
    #fetch_pinecone()


if __name__ == "__main__":
    print("7. Main started")
    print(os.environ.get("PINECONE_API_KEY"))
    load_document(
        "E:/AgenticAI-Udemy/rag-gist/mediumblog1.txt",
        "basic",
        100000,
    )