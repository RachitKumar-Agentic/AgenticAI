from dotenv import load_dotenv
import os
from typing import List
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

#
# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query (str)The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     tavliclient = TavilyClient()
#     result = tavliclient.search(query=query)
#     #return "Tokyo weather is sunny"
#     return result

class Source(BaseModel):
    url:str = Field(description="The URL of the Source")
    title:str = Field(description="The details of the job")

class AgentResponse(BaseModel):
    source:List[Source]


def main():
    print("Hello from langchain-course!")
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-latest")
    tools =[TavilySearch()]
    agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)
    #result = agent.invoke({"messages":HumanMessage(" What is the weather in Tokyo")})
    result = agent.invoke({"messages": HumanMessage(content = "search for 3 job posting for an ai engineer using langchain in the bay area on linkedin and list their source url")})
    print(result)


if __name__ == "__main__":
    main()

