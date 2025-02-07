from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from typing import Annotated, Dict, Any
from langchain_core.tools import tool

K_PRODUCTS = 2
K_CONTENT = 3
PATH_DB = "./data/retailer_db"
COLLECTION_PRODUCTS = "products"
COLLECTION_CONTENT = "infos"
SYSTEM_MESSAGE = """You are Shokz, a helpful and conversational shopping assistant of Shokz with access to these tools:

1. retrieve_products: Use to get specific product details including ID, title, price, URL, image, and description
2. retrieve_infos: Use to get general product knowledge or company information

Key Guidelines:
1. Keep initial responses CONCISE - show products but don't overwhelm with details
2. Use a step-by-step conversational approach
3. Ask questions to understand customer needs before giving specific recommendations

When responding about products:

PART 1 - Product Display:
First, list the basic product details in this format for EACH product:
====== Recommend Product {N} ======
Title: [Product Title]
Price: [Price]
URL: [Product URL]
Image: [Image URL]
Description: [BRIEF Description - 1-2 sentences max]
================================

PART 2 - Brief Introduction & Next Steps:
- Give a VERY BRIEF overview (2-3 sentences max)
- Ask a relevant question to understand the customer's specific needs

DO NOT:
- Don't provide extensive product benefits unless asked
- Don't list all features and specifications upfront
- Don't give detailed recommendations without understanding customer needs
- Don't provide additional tips unless specifically requested
- Don't lead with the quiz recommendation - address immediate needs first

Remember:
- Always address the immediate question first
- Keep initial responses focused and brief
- Let the customer guide the conversation
- Mention the quiz as a helpful option when relevant, not as the primary solution
- Save detailed information for follow-up responses based on customer interest

Tool Usage Guidelines:
1. retrieve_products: Use first to show relevant products
2. retrieve_infos: Use when customer asks for more detailed information
"""

### --- Initialize Claude model ---###
llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0
)

# Initialize embedding model
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Load products into vector database
vector_store_products = Chroma(
    collection_name=COLLECTION_PRODUCTS,
    embedding_function=embedding_model,
    persist_directory=PATH_DB
)
retriever_products = vector_store_products.as_retriever(
    search_kwargs={"k": K_PRODUCTS})


# Load contents into vector database
vector_store_content = Chroma(
    collection_name=COLLECTION_CONTENT,
    embedding_function=embedding_model,
    persist_directory=PATH_DB
)
retriever_content = vector_store_content.as_retriever(
    search_kwargs={"k": K_CONTENT})


@tool
def retrieve_products(query: str) -> str:
    """
    Retrieve products information including product id, name, description, price, 
    image url, tags, and category from vectorstore. Use this when needing specific 
    product details or answering product-related questions.
    """
    products = retriever_products.invoke(query)
    
    formatted_results = []
    for doc in products:
        product_info = "\n=== Relevant Product ===\n"
        product_info += f"Product ID: {doc.metadata.get('id', 'N/A')}\n"
        product_info += f"Title: {doc.metadata.get('title', 'N/A')}\n"
        product_info += f"Price: ${doc.metadata.get('price', 'N/A')}\n"
        product_info += f"Product URL: {doc.metadata.get('product_url', 'N/A')}\n"
        
        if 'image_src' in doc.metadata:
            product_info += f"Image: {doc.metadata['image_src']}\n"
        
        product_info += "\nDescription:\n"
        product_info += f"{doc.page_content}\n"
        product_info += "======================\n"
        
        formatted_results.append(product_info)

    return "\n".join(formatted_results)

@tool
def retrieve_infos(query: str) -> str:
    """
    Retrieve relevant product knowledge, brand information, customer service information,
    product category, product features, and product benefits from vectorstore. Use this
    when needing general product knowledge or company information.
    """
    infos = retriever_content.invoke(query)
    
    formatted_results = []
    for doc in infos:
        info = "\n=== Relevant Information ===\n"
        # Add any metadata fields that are present
        for key, value in doc.metadata.items():
            info += f"{key}: {value}\n"
        
        info += "\nContent:\n"
        info += f"{doc.page_content}\n"
        info += "======================\n"
        
        formatted_results.append(info)
    
    return "\n".join(formatted_results)

tools = [retrieve_products, retrieve_infos]
llm_with_tools = llm.bind_tools(tools)
### --- End of Define tools ---###

sys_msg = SystemMessage(content=SYSTEM_MESSAGE)

### --- Build graph ---###
# define State (Define a custom TypedDict that includes a list of messages with add_messages reducer


# Build graph
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)

# Add assistant node
builder.add_node("assistant", assistant)

# Add tools node
builder.add_node("tools", ToolNode(tools))

# Add edges 
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition, "tools")
builder.add_edge("tools", "assistant")

memory = MemorySaver()
# Compile graph
graph = builder.compile(checkpointer=memory)
