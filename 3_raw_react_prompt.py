from dotenv import load_dotenv
import re
import inspect
load_dotenv()
import json

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price=float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}


def get_tool_descriptions(tools):
    descriptions = []
    for tool_name, tool_function in tools.items():
        original_function = getattr(tool_function, "__wrapper__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name},{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
 "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price"
                "returned by get_product_price -- do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them with tier to use -- do NOT assume one."

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model,messages,options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    prompt = react_prompt.format(question=question)
    scratchpad=""
    print(f"Question: {question}")
    print("=" * 60)

#"stop":["\n435436546Observation"],
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(model=MODEL, 
            messages=[{"role":"user","content":full_prompt}],
            options={"stop":["\nObservation"],"temperature":0.0})
        output = response.message.content
        print(f"LLM Output:\n{output}")
        

        print(f"[Parsing] Looking for Final Answer in LLM Output\n")
        final_answer_match = re.search(r"Final Answer:\s(.+)",output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("=" * 60)
            print(f"\n\nFinal Answer : {final_answer}")
            return final_answer
        # Process only the FIRST tool call — force one tool per iteration
        print(f" [Parsing] Looking for Action and Action Input from the LLM Output")
        action_match = re.search(r"Action:\s(.+)",output)
        action_input_match = re.search(r"Action Input:\s(.+)",output)
        if not action_match or not action_input_match:
            print(" Parsing the Error could not parse Action/Action Input from the LLM Output")
            break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")
        
        tool_input_raw=tool_input_raw.replace("'","").replace("{","").replace("}","").replace(":","=").strip()
        print(tool_input_raw)
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args=[x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]
        print(f"args after ={args}")
        
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found : Available tools are {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
            
            
        
        print(f"\b onbservation = {observation}")
        scratchpad+=f"{output}\nObservation: {observation}\nThought:"
        
      

        # Difference 7: Direct function call instead of tool.invoke()
       

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")