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

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=2000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    return map_reduce_chain.invoke(split_docs)
