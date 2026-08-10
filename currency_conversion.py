!pip install langchain
!pip install langchain-community
!pip install langchain-huggingface
!pip install sentence-transformers
!pip install -U langchain-huggingface huggingface_hub
!pip install -q langchain-openai langchain-core requests
!pip install -U langchain langchain-core langchain-huggingface huggingface_hub

from langchain_core.tools import tool
import requests
from google.colab import userdata
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def exchange_rate(base_currency :str, target_currency :str) -> float:
  """This function fetches the exchange rate between base_currency and target_currency"""
  url =f"https://v6.exchangerate-api.com/v6/38b90e5354deaf07f21aaa7a/pair/{base_currency}/{target_currency}"
  response =requests.get(url)

  return response.json()["conversion_rate"]

@tool
def convert(base_currency_value : int , conversion_rate: float) -> float:
  """ This function multiple base_currency_value with exchange rate """
  final_value = base_currency_value * conversion_rate
  return final_value

  os.environ["USER_AGENT"] = "Mozilla/5.0"

api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
llm =HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)
model = ChatHuggingFace(llm = llm)

model_with_tools = model.bind_tools([exchange_rate,convert])

result = model_with_tools.invoke("convert 4000 USD to INR")
print(result)
final_rate =None
final_amount =None

for tool_call in result.tool_calls :
      if tool_call["name"] == "exchange_rate":
        final_rate = exchange_rate.invoke(tool_call["args"])
        print(final_rate)
      elif tool_call["name"] == "convert":
        value =tool_call["args"]["base_currency_value"]
        print(type(value))
        final_amount = convert.invoke({'base_currency_value': value, 'conversion_rate': final_rate})

print(final_amount)
