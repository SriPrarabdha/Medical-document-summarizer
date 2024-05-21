from stuff_summarization import stuff_summarize
from map_reduce_summarization import map_reduce_summary

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
import google.generativeai as genai
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

# os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDx7cfKeqr0YK0TE8767lnMz6G5NmeXJBI"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0)


with open('/teamspace/studios/this_studio/Medical-document-summarizer/Extracted_Data/Medical_Record_File_3.json', 'r') as file:
    data = json.load(file)

text = ""
# print(data)
for key in data:
    # print(key)
    text= text + " " + data[key]
    # print(data[key])

chunk_size_tok = 5000
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0
)
texts_split = text_splitter.split_text(text)
# print(texts_split)

docs = [Document(page_content=text) for text in texts_split]

response = map_reduce_summary(docs, model)

print(response)

