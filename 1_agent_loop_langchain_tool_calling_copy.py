from dotenv import load_dotenv

load_dotenv()
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

@tool
def get_product_price(product:str) -> float:
    """ Get the product price """
    print(f" > Executing get_product_price for {product}")
    products={"laptop":189.67,"headphone":56.78,"keyboard":67.6}
    return products.get(product,0)


@tool
def discount_Tier(price:float,discount_tier:float) -> float:
    """ Find discount if the product on discount Tier : bronze:1,silver:2, goldL3"""

@traceable(name="Langsmith Aggent Hood")
def run_agent(question:str):
    pass

if __name__ == "__main__":
    print("Looping Under the hood agent")
    result=run_agent("Get me laotop product with gold discount tier")