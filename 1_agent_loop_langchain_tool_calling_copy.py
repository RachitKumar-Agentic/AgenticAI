from dotenv import load_dotenv
from ollama import chat
from langchain.tools import tool
import json

MODEL = "qwen3:1.7b"
message = ["Hi How are you?","I want to buy laptop with budget less than 30k","It should be good to carry"]

@tool
def get_product_price(product:str) -> str:
    """ Look up the Price of a product in the catalog."""
    return product
def apply_discount(price:float,product:str)->float:
    """" Apply a discount tier to a price and return the final price.
    Available tiers : bronze, silver, gold"""
    return 1.0

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

#tool = tools["get_product_price"]("keyboard")
tool = tools["get_product_price"].invoke("keyboard")
print(tool)

tool_name = '{"product":"laptop","product2":"key"}'



tool_input_raw=tool_name.replace("'","").replace("{","").replace("}","").replace(":","=").strip()
print(tool_input_raw)
raw_args = [x.strip() for x in tool_input_raw.split(",")]
args=[x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]
print(f"args after ={args}")

print(tools)
print(type(tools))
# msgs=""
# for msg in message:
#     msgs=msgs + "\n" + msg 
#     response = chat(model=MODEL,messages=[{"role":"user","content":msgs}])
#     msgs = msgs + "\n" + response.message.content



# print(f"\n Output : {msgs}")
# print(f"\n done")