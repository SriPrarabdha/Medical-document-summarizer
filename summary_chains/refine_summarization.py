from langchain.chains.summarize import load_summarize_chain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# os.environ["GOOGLE_API_KEY"] = "AIzaSyCP1kveVOTOIMyzvEY6Xdwpq18567ETBPU"
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# llm = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.8)

def refine_summarize(split_docs, llm):
    """
    """
    chain = load_summarize_chain(llm, chain_type="refine")
    
    summary = chain.run(split_docs)
    return summary