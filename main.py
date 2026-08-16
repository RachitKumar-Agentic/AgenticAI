import io
from dotenv import load_dotenv
import certifi
from langchain_tavily import TavilyExtract, TavilyCrawl, TavilyMap
import os
import asyncio
from langchain_core.documents import Document


load_dotenv()

certifi_path = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi_path
os.environ['REQUESTS_CA_BUNDLE'] = certifi_path
os.environ['CURL_CA_BUNDLE'] = certifi_path

tavily_crawl = TavilyCrawl()

async def main():
    print("Hello from mytrytavily!")
    res=tavily_crawl.invoke({"url":"https://docs.langchain.com/oss/python/",
    "max_depth":5,
    "instructions":"get me only ai raleted document",
    "extract_depth":"advanced"
    
    })
    
    all_doc = res['results']
    pydocument =  [Document(page_content=doc['raw_content'], metadata={"source": doc['url']}) for doc in all_doc if doc['raw_content']
        ]
    print(pydocument)

if __name__ == "__main__":
    asyncio.run(main())

