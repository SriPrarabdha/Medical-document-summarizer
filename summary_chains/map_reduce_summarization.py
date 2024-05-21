from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def map_reduce_summary(split_docs, llm):

    map_template = """Here is a medical record file of some patient your task is to just create in detail
                summaries that has every thing related to the given context. Just provide me with summary of everything written in the context do not add anything extra from your side.
    
    Context:
    "{docs}"
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    print("check 1 map_chain done")

    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, comprehensive summary that covers each and every thing
    from these docs do not add any extra inference from your side just report what is written in the
    given texts. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    print("check 2 reduce chain")

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    print("check 3 combine done")

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=5000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=True,
    )
    print("last_check")

    summary = ""

    response = map_reduce_chain.invoke(split_docs)

    for intermediate_summaries in response["intermediate_steps"]:
        summary = summary + " "+ intermediate_summaries

    summary = summary + " " + response["output_text"]

    return summary
