import chainlit as cl
from langgraph.graph.state import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from typing import TypedDict, List, Any
import os

load_dotenv()

TOGETHER_key = os.getenv("TOGETHER_API_KEY")
TOGETHER_base = os.getenv("TOGETHER_API_BASE")

os.environ["OPENAI_API_KEY"] = TOGETHER_key
os.environ["OPENAI_API_BASE"] = TOGETHER_base

qdrant = None
vectorstore = None
llm = None
qa_chain = None
graph = None

class RagState(TypedDict):
    question: str
    answer: str
    sources: List[Any]

def rag_node(state: RagState) -> RagState:
    """RAG processing node"""
    try:
        query = state["question"]
        print(f"Processing query: {query}") 
        
        result = qa_chain.invoke({"query": query})  
        
        return {
            "question": query,
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    except Exception as e:
        print(f"Error in rag_node: {e}")
        return {
            "question": state["question"],
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }

@cl.on_chat_start
async def start():
    """Initialize the RAG system when a chat starts"""
    global qdrant, vectorstore, llm, qa_chain, graph
    
    await cl.Message(
        content="Welcome to Commedia Solutions Pvt. Ltd. I am comAI, your virtual assistant!\n\nI'm initializing the system... Please wait a moment.",
        author="Commedia Assistant"
    ).send()
    
    try:
        qdrant = QdrantClient(host="localhost", port=6333)
        
        collections = qdrant.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = QdrantVectorStore(
            client=qdrant,
            collection_name="website_rag",
            embedding=embedding_model,
            content_payload_key="page_content"
        )

        test_results = vectorstore.similarity_search("test", k=1)
        print(f"Test search returned {len(test_results)} results")

        llm = ChatOpenAI(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.2,
            timeout=30,
            max_retries=2
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  
            return_source_documents=True,
            chain_type="stuff"  
        )

        builder = StateGraph(RagState)
        builder.add_node("rag", rag_node)  
        builder.set_entry_point("rag")
        builder.set_finish_point("rag")
        graph = builder.compile()
        
        await cl.Message(
            content="System Ready!\n\n"
            "You can now ask me questions about Commedia Solutions and our service offerings. "
            "I'll provide you with accurate answers, complete with source references.\n\n"
            "**Example questions you can try:**\n"
            "• What services does Commedia offer?\n"
            "• How does Commedia ensure seamless project implementation?\n"
            "• What markets does Commedia operate in?\n\n"
            "Go ahead and ask me anything about Commedia Solutions!",
            author="Commedia Assistant"
        ).send()
        
    except Exception as e:
        print(f"Initialization error: {e}")
        await cl.Message(
            content=f"Error initializing the system:\n\n"
                   f"```\n{str(e)}\n```\n\n"
                   "**Please check:**\n"
                   "• Qdrant is running on localhost:6333\n"
                   "• Collection 'website_rag' exists and has data\n"
                   "• Your API keys are set correctly\n"
                   "• All dependencies are installed",
            author="Commedia Assistant"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    global graph

    if not graph:
        await cl.Message(
            content="System not initialized. Please refresh the page to restart.",
            author="Commedia Assistant"
        ).send()
        return

    user_question = message.content.strip()

    if not user_question:
        await cl.Message(
            content="Please ask a question!",
            author="Commedia Assistant"
        ).send()
        return

    thinking_msg = cl.Message(
        content="Thinking... \n\nSearching through the knowledge base for relevant information...",
        author="Commedia Assistant"
    )
    await thinking_msg.send()

    try:
        print(f"Processing question: {user_question}")  

        result = graph.invoke({
            "question": user_question,
            "answer": "",
            "sources": []
        })

        print(f"Graph result: {result}")  

        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", [])

        thinking_msg.content = f"Answer\n\n{answer}"
        await thinking_msg.send()

        if sources and len(sources) > 0:
            sources_content = "Sources\n\n"
            for i, source in enumerate(sources, 1):
                try:
                    if hasattr(source, 'page_content') and source.page_content:
                        content_preview = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
                        sources_content += f"Source {i}:\n```\n{content_preview}\n```\n\n"
                    elif hasattr(source, 'metadata') and source.metadata.get('text'):
                        content_preview = source.metadata['text'][:300] + "..." if len(source.metadata['text']) > 300 else source.metadata['text']
                        sources_content += f"Source {i}:\n```\n{content_preview}\n```\n\n"
                    else:
                        sources_content += f"Source {i}:\n*(No content available)*\n\n"
                except Exception as e:
                    sources_content += f"Source {i}:\n*(Error accessing source: {str(e)})*\n\n"

            await cl.Message(
                content=sources_content,
                author="Commedia Assistant"
            ).send()
        else:
            await cl.Message(
                content="No specific sources found for this question.",
                author="Commedia Assistant"
            ).send()

    except Exception as e:
        print(f"Error processing message: {e}")
        thinking_msg.content = (
            f"Error processing your question:\n\n"
            f"```\n{str(e)}\n```\n\n"
            "Please try rephrasing your question or check the system logs."
        )
        await thinking_msg.send()

@cl.cache
def get_custom_css():
    return """
    .message-content {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }
    .message-avatar {
        border-radius: 50%;
        border: 2px solid #4A90E2;
    }
    .message-content h1 {
        color: #2B4162;
        margin-bottom: 12px;
    }
    /* Customize assistant message bubble */
    .message.assistant {
        background-color: #e1f0ff;
        color: #0b3d91;
        border-radius: 12px;
    }
    """
if __name__ == "__main__":
    pass