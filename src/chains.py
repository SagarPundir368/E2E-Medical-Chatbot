from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Core prompt assets imported from your modular prompt script module
from src.prompt import QUERY_ENHANCEMENT_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT,HISTORY_AWARE_QUERY_PROMPT

def get_query_enhancer_chain(llm) -> ChatPromptTemplate:
    """Constructs the pre-retrieval query intelligence and translation chain.

    This chain takes a raw user query string, leverages the LLM to expand synonyms
    and isolate sub-queries, and parses the output into a structured Python dictionary.

    Args:
        llm: The underlying LangChain ChatModel or LLM instance.

    Returns:
        The assembled executable LCEL chain outputting a JSON-parsed dictionary.
    """
    # Changed variable name to input_variables=["raw_query"] to match your prompt file
    prompt = PromptTemplate(
        input_variables=["query"],
        template=QUERY_ENHANCEMENT_PROMPT
    )
    
    return prompt | llm | JsonOutputParser()



def get_history_aware_query_chain(llm):

    prompt = ChatPromptTemplate.from_messages([
        ('system',HISTORY_AWARE_QUERY_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ('human',"{query}")
    ])

    return (
        prompt
        | llm
        | StrOutputParser()
    )


def get_final_rag_chain(llm) -> ChatPromptTemplate:
    """Assembles the final grounded generation chain with message history tracking.

    Binds the strict grounding instructions, contextual layman translation rules,
    the active multi-turn conversation trace context, and the user's optimized target question.

    Args:
        llm: The underlying LangChain ChatModel or LLM instance.

    Returns:
        The assembled executable LCEL chain outputting a clean markdown string response.
    """
    # Fixed: MessagesPlaceholder spelling and added missing list element commas
    prompt = ChatPromptTemplate.from_messages([
        ("system", FINAL_RESPONSE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {query}\n\nRetrieved Context: {context}")
    ])

    return prompt | llm | StrOutputParser()