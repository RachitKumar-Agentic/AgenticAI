from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
import os

load_dotenv()
MODEL = os.getenv("MODEL_NAME")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY,base_url=os.getenv("OPENAI_BASE_URL"))
CASE=[
    ("I bought a jacket, delivered 20 days ago. Can I return it?", "YES"),
    ("The shoes arrived yesterday. I want to send them back.", "YES"),
    ("My order was delivered 45 days ago. Can I still return it?", "NO"),
    ("I got a gift card last week. Can I return it?", "NO"),
    ("The underwear I ordered came 3 days ago. Return possible?", "NO"),
    ("Delivered 29 days ago, is a return still possible?", "YES"),
]

PROMPT_V1 = (    "Role:You are a ShopAssistant"
    "Scope:Only check on order , return and refund"
    "Rule:the order which are in 30 days from the order date are eligible for return and refund, underware products are not refundable or returned"
    "style:Simple answer Yes or No first and then one single line of information"
)


def answer(system, text):
    history=[]
    history.append({"role":"system","content":system})
    history.append({"role":"user","content":text})
    response = client.chat.completions.create(messages=history,model=MODEL)
    return response.choices[0].message.content

def evaluate(name,system):
    score = 0
    for text,expected in CASE:
        reply = answer(system,text)
        ok = reply.strip().upper().startswith(expected)
        score+=ok
        print("PASS" if ok else "FAIL" , "|" , text[:45] , "|" ,reply[:45] )
        print()

    print(name,score)


evaluate("v1",PROMPT_V1)

