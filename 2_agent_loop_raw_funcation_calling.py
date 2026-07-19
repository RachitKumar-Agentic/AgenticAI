import ollama
from cryptography.x509 import name
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
load_dotenv()
from langchain.tools import tool
from langsmith import traceable
from ollama import chat

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"



# ---------------------- Tool (Langchain @tool decorator) -----------
@tool
def get_product_price(product: str) -> float:
    """ Look up the Price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    products={"laptop":1299.99,"headphones":149.95,"keyboard":89.50}
    return products.get(product,0)

@tool
def apply_discount(price:float,discount_tier:str) -> float:
    """" Apply a discount tier to a price and return the final price.
    Available tiers : bronze, silver, gold"""
    print(f" >> Executing apply_discount(price={price}, discount_tire='{discount_tier}')")
    discount_percentages={"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1-discount/100),2)

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

#--------------- Helper funcation to :traced Ollamma call ----------
@traceable(name="Ollama Chat",run_type="llm")
def ollama_chat(messages:str):
    return ollama.chat(model=MODEL,tools=tools_for_llm, messages=messages)

#---------------------------- Agent Loop -----------------------
@traceable(name="Langchain Agent Loop")
def run_agent(question:str):

    tools_dict= {"get_product_price":get_product_price,
                 "apply_discount":apply_discount
    }


    print(f"Question:{question}")
    print("-"*60)

    messages=[
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price"
                "returned by get_product_price -- do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them with tier to use -- do NOT assume one."
            )
        ),
        HumanMessage(content=question),

    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n ============= Iteration : {iteration}=============")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls


        #==== if not tool calls check by using if not tool_calls

        if not tool_calls:
            print(f"\n Final answer == {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args : {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))








if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    result=run_agent("What is the price of a laptop after buying a gold discount get me in INR?")


