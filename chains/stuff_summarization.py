from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

def stuff_summarize(split_docs, model):
    
    prompt_template = """Here is a medical record file of some patient your task is to just create in detail
                summaries that has every thing related to the given context. Just provide me with summary of everything written in the context do not add anything extra from your side.
    
    Context:
    "{text}"
    """
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    response = stuff_chain.invoke(split_docs)

    return response['output_text']
