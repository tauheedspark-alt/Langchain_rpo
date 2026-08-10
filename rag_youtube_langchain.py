#Developer : Tauheed Ahmad (tauheed.spark@gmail.com)
#version : 1.0
#!pip install langchain
#!pip install langchain-community
#!pip install langchain-huggingface
#!pip install chromadb
#!pip install sentence-transformers
#!pip install -U langchain-huggingface huggingface_hub
#!pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    # Flatten transcript into plain text
    text = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

# split the text
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#both below retrun list but create document function retrun Document object which can save in vector store and vector db
chunks1 =splitter.split_text(text)
chunks = splitter.create_documents([text])


#creating vector croma db and adding cunks in vector db
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embedding_model
)

vector_store.add_documents(chunks)


# creating reterival

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})
question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

# retrieved_docs has 3 document which we retrive so before sending to llm we will merge all doc to single text
content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)


# desigining prompt

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template ="You are a helpful assistant. \
      Answer ONLY from the provided transcript context. \
      If the context is insufficient, just say you don't know. \
      {context} \
      Question: {question}",

    input_variables={'context','questions'}
)

final_prompt = prompt.invoke({'context':content_text,'question':question})


#creating llm models to send prompts
from google.colab import userdata
from os import environ
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os

os.environ["USER_AGENT"] = "Mozilla/5.0"

api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
llm =HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)
model = ChatHuggingFace(llm = llm)

result = model.invoke(final_prompt)

print(result)


# creating runnable so we do not need to call every function seperatly

from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel(
    {
        "question" : RunnablePassthrough() ,
        "context"  : retriever | RunnableLambda(format_docs)
    }
)

sequence_chain = prompt | model | parser
final_chain = parallel_chain | sequence_chain

result = final_chain.invoke(question)
print(result)
