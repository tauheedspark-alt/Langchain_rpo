#Developer : Tauheed Ahmad (tauheed.spark@gmail.com)
#version : 1.0
#!pip install langchain
#!pip install langchain-community
#!pip install langchain-huggingface
#!pip install sentence-transformers
#!pip install -U langchain-huggingface huggingface_hub
#!pip install -q langchain-openai langchain-core requests
#!pip install -U ddgs

from langchain_core.tools import tool
import requests

@tool
def get_weather_data(city : str) -> float :
  """This function provides weather of given city"""
  url = f'https://api.weatherstack.com/current?access_key=6818b73ff0ec33541be34803bfc37a9d&query={city}'
  response = requests.get(url)
  return response.json()

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()  

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from google.colab import userdata

api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
llm =HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)
model = ChatHuggingFace(llm = llm)
model_with_tools = model.bind_tools([get_weather_data,search_tool])

from langchain.agents import create_agent
agent = create_agent(
    model=model,
    tools=[search_tool, get_weather_data]
)

user_query = """Find the capital of Lucknow, then find its current weather condition"""
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_query
        }
    ]
})

messages = result["messages"][-1].content
print(messages)

