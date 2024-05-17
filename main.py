import pdf2image
import io
import base64
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
import google.generativeai as genai
import json
from langchain_community.document_loaders import PyPDFLoader


from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import SpladeEncoder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool , initialize_agent
from langchain import SerpAPIWrapper
from langchain.agents import AgentType
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import CallbackManager
from langchain.tools.human.tool import HumanInputRun
# from pydantic import BaseModel
import os
import pinecone
import openai
import time
# from fastapi import FastAPI
from lcserve import serving

import nltk
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = [m for m in genai.list_models()]
print(models)

model = ChatGoogleGenerativeAI(model='models/gemini-pro-vision', temperature=0.8)
# images = pdf2image.convert_from_path(pdf_path = "Medical_Record_File_1.pdf", first_page=4, last_page=24)
# print(images[0])

def convert_pdf_to_base64(pdf_path, first_page=4, last_page=24, output_folder="converted_images"):

  images = pdf2image.convert_from_path(pdf_path=pdf_path, first_page=first_page, last_page=last_page)
  encoded_images_json = {}

  for i, image in enumerate(images):
    with io.BytesIO() as output:
      
      image.save(output, format="JPEG")
      image_data = output.getvalue()
     
      encoded_image = base64.b64encode(image_data).decode("utf-8")

      filename = f"page_{i+4}.jpg"  
      encoded_images_json[f"{i+4}": encoded_image]
      image.save(os.path.join("/teamspace/studios/this_studio/images", filename), format="JPEG")
  
  try:
    json_string = json.dumps(encoded_images_json, indent=4)  

    with open(pdf_path, 'w') as f:
        json.dump(data, f)
  except: 
    continue
  return encoded_images


encoded_images = convert_pdf_to_base64(pdf_path="Medical_Record_File_1.pdf",first_page=4, last_page=6)
# print(encoded_images)
# print(len(encoded_images))
text_message = {
    "type": "text",
    "text": "Your task is to return me each and every thing written in this image as a plain simple text and nothing else"
}

def get_image_text(encoded_images_json):
    image_data = {}

    for page, encoded_string in enumerate(encoded_images_json):
        image_message = {
            "type": "image_url",
            "image_url" : {"url" : f"data:image/jpeg;base64,{encoded_string}"}
        }
        message = HumanMessage(content=[text_message , image_message])

        answer = model.invoke([message])

        print(answer.content)

        image_data[page] = answer.content

    return image_data


def merge_image_text_data(filename:str, encoded_images_json:dict):
    #TODO: MATCHING THE PAGE NUMBERS OF TEXT DATA AND IMAGE DATA

    final_data = {}

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    for key in encoded_images_json:
        final_data[key] = encoded_images_json[key] + pages[key].content

    return final_data


        

@serving(websocket=True)
def get_answers(prompt:str , **kwargs):
    streaming_handler = kwargs.get('streaming_handler')

    #initialize the pinecone index
    pinecone.init(
        api_key="API_KEY",
        environment="ENV"
    )
    index = pinecone.Index("medical-records")

    #Getting our encoding models ---> dense + sparse
    embeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")
    splade_encoder = SpladeEncoder()

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade_encoder, index=index)

    #setting up open ai env
    os.environ["OPENAI_API_KEY"] = "API_KEY" 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "ORG_ID"
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo" , verbose = True , streaming=True , callback_manager=CallbackManager([streaming_handler]))

    # Creating a langchain document retriver chain
    dnd_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever , callback_manager = CallbackManager([streaming_handler]))

    os.environ["SERPAPI_API_KEY"] = "API_KEY"
    search = SerpAPIWrapper()

    #Adding tools to help the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or when you get no results from criminal-laws tool",
            return_direct=True
        ),
        Tool(
            name="medical_data",
            func=dnd_qa.run,
            description="useful for when you need to know something about medical records. Input: an objective to know more about a law and it's applications. Output: Correct interpretation of the law. Please be very clear what the objective is!",
            return_direct=True
        ),
    ]
    tools.append(HumanInputRun())

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)
    
    # Returning the thoughts of the agent with api results
    hey = agent(prompt)
    steps = []
    for line in hey["intermediate_steps"][0][0]:
        steps.append(line)
    Final_ans = hey["output"].split(" ")
    newdict = {"Thoughts":steps, "Final Answer :" : Final_ans}
    for line in newdict["Thoughts"]:
        word_split = line.split(" ")
        for word in word_split:
            # print(word, end = " ")
            yield str(str(word) + " ")
            time.sleep(0.2)
    yield b'\nFinal Answer : '
    stringk = newdict["Final Answer :"]
    # stringk = stringk.split(" ")
    for word in stringk:
        # print(word, end=" ")
        yield str(str(word) + ' ')
        time.sleep(0.2)
    return newdict


@app.get('/{prompt}')
async def main(prompt : str):
    return StreamingResponse(get_answers(prompt), media_type='text/event-stream')