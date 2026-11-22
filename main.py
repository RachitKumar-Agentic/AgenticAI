import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from openai.types.responses import tool

load_dotenv(override=True)

# print("KEY:", os.getenv("OPENAI_API_KEY"))
# print("URL:", os.getenv("OPENAI_BASE_URL"))
# print("MODEL:", os.getenv("MODEL_NAME"))

MODEL = os.getenv("MODEL_NAME")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

SYSTEM = (
    "Role:You are a shopassistant"
    "Rule: Only return the order who have the delivered_days_ago less than or equal to 30"
    "Scope: only return , refund the order , do not return or refund the innerwear and gift card"
    "Style:Return simple Yes or No and then one ine answer"
)


ORDER = {
    "ORD-12345678":{
        "item":"jacket",
        "status":"delivered",
        "delivered_days_ago":20
    },
    "ORD-1234567":{
        "item":"jacket",
        "status":"delivered",
        "delivered_days_ago":40
    }
}


def lookup_order(order_id):
    return ORDER.get(order_id,{"error":"No order id"})

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Lookup an order by the order_id and returns the item, status and when it was delivered",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The Order ID, like ORD-12345678",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]

# print(lookup_order("ORD-12345678"))
# print(lookup_order("1234567"))

AVAILABLE_FUNCTIONS = {
    "lookup_order":lookup_order
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
user_messages = [
    {
        "role":"user",
        "content":"Can i return the order with order id ORD-12345678 and order id ORD-1234567"
    }
]
response = client.chat.completions.create(model=MODEL, tools=TOOLS,messages=user_messages)
#print("resoponse===",response)
message=[]
message.append({"role":"system","content":SYSTEM})

response_message = response.choices[0].message
#print(response_message)

func_name=""
if response_message.tool_calls:
    message.append(response_message)
    for tool_call in response_message.tool_calls:
        #print("toolcall===",tool_call,"/n")
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if func_name:
            
            fn = AVAILABLE_FUNCTIONS[func_name]
            reply = fn(**args)
            
            message.append({"role":"tool","tool_call_id":tool_call.id,"content": json.dumps(reply)})
            
#print(func_name,args)
response_function = client.chat.completions.create(messages=message,model=MODEL)
print(response_function.choices[0].message.content)
