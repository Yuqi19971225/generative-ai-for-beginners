from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

deployment_name = "qwen-plus"

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# add your completion code
no_recipes = input("No of recipes (for example, 5): ")

ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots): ")

filter = input("Filter (for example, vegetarian, vegan, or gluten-free): ")

prompt = f"Show me {no_recipes} recipes for a dish with the following ingredients: {ingredients}. Per recipe, list all the ingredients used, no {filter}"

messages = [{"role": "user", "content": prompt}]

# make completion
completion = client.chat.completions.create(model=deployment_name, messages=messages)

# print response
print(completion.choices[0].message.content)

old_prompt_result = completion.choices[0].message.content
prompt = "Produce a shopping list for the generated recipes and please don't include ingredients that I already have."

new_prompt = f"{old_prompt_result} {prompt}"
messages = [{"role": "user", "content": new_prompt}]
completion = client.chat.completions.create(model=deployment_name, messages=messages, max_tokens=1200)

# print response
print("Shopping list:")
print(completion.choices[0].message.content)