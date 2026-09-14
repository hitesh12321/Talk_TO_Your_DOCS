from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int


import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from src.vector_store import VectorStore

load_dotenv()

# LLM for grading & generation
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Load existing FAISS store
vector_store = VectorStore()
vector_store.load()

def retrieve_node(state):
    """
    take user query from state and retrive top k chunks form vector store
    State se user query nikalta hai aur VectorStore se top-k chunks retrieve karta hai.
    """
    print("--- 1. RETRIEVING DOCUMENTS ---")
    
    # State se current question nikala
    question = state["question"]
    
    # Top-K chunks search (vector_store.search khud hi list return karta hai)
    documents = vector_store.search(question, k=3)
    
    # State me 'documents' key update karke return kar do
    return {"documents": documents}

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# Ya ChatGroq / ChatOpenAI import kar sakte hain

# -----------------------------
# 1. Pydantic Model for Grader Output
# -----------------------------
class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# -----------------------------
# 2. LLM & Grader Chain Setup
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Structured Output Enforce karna ('yes' ya 'no')
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])

doc_grader = grade_prompt | structured_llm_grader


# -----------------------------
# 3. Grade Documents Node Function
# -----------------------------
def grade_documents_node(state):
    """
    Iterates through retrieved documents and filters out irrelevant ones using LLM.
    """
    print("--- 2. CHECKING DOCUMENT RELEVANCE TO QUESTION ---")
    
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    
    for doc in documents:
        # LLM se relevance grade maango
        score = doc_grader.invoke({
            "question": question, 
            "document": doc.page_content
        })
        
        grade = score.binary_score.lower()
        
        if grade == "yes":
            print("  -> Grade: [RELEVANT]")
            filtered_docs.append(doc)
        else:
            print("  -> Grade: [NOT RELEVANT]")
            continue
            
    # State update: Ab state me sirf relevant documents hi bachenge
    return {"documents": filtered_docs}

def decide_to_generate(state):
    """
    Decides whether to generate an answer or rewrite the query.
    """
    filtered_docs = state["documents"]
    retry_count = state.get("retry_count", 0)

    # Agar kam se kam 1 relevant doc mil gaya, toh Answer Generate karo
    if len(filtered_docs) > 0:
        print("--- DECISION: GENERATE ANSWER ---")
        return "generate"
    
    # Agar 0 relevant docs hain aur retry 2 se kam hain, toh Query Rewrite karo
    elif retry_count < 2:
        print("--- DECISION: REWRITE QUERY ---")
        return "transform_query"
    
    # Agar max retries ho chuke hain
    else:
        print("--- DECISION: MAX RETRIES REACHED, TRY GENERATING ---")
        return "generate"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 1. Query Rewriter Prompt & Chain
# -----------------------------
system_prompt = """You are a query re-writer that converts an input question into a better version 
optimized for vectorstore retrieval. Look at the input question and reason about the underlying intent and keywords. 
Formulate an improved, concise search query."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Initial Question:\n\n{question}\n\nFormulate an improved search query:")
])

# OutputParser string me result clean kar deta hai
question_rewriter = re_write_prompt | llm | StrOutputParser()


# -----------------------------
# 2. Transform Query Node Function
# -----------------------------
def transform_query_node(state):
    """
    User question ko LLM se rewrite karwata hai aur retry_count increment karta hai.
    """
    print("--- 3. TRANSFORMING / REWRITING QUERY ---")
    
    current_question = state["question"]
    current_retry = state.get("retry_count", 0)
    
    # LLM se new rewritten question generate karo
    better_question = question_rewriter.invoke({"question": current_question})
    
    print(f"  Old Query : '{current_question}'")
    print(f"  New Query : '{better_question}'")
    
    # State Update: Updated question aur incremented retry_count return karo
    return {
        "question": better_question,
        "retry_count": current_retry + 1
    }

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 1. RAG Answer Generation Prompt & Chain
# -----------------------------
system_prompt = """You are an AI assistant for document question-answering tasks. 
Use the following pieces of retrieved context to answer the user question. 
If you don't know the answer or if the context is insufficient, state that clearly based on the provided document.
Keep your answer clear, accurate, concise, and structured."""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Retrieved Context:\n\n{context}\n\nUser Question:\n{question}\n\nAnswer:")
])

# RAG Chain
rag_chain = generate_prompt | llm | StrOutputParser()


# -----------------------------
# 2. Generate Node Function
# -----------------------------
def generate_node(state):
    """
    Uses the validated documents in state to generate a final answer via LLM.
    """
    print("--- 4. GENERATING FINAL ANSWER ---")
    
    question = state["question"]
    documents = state["documents"]
    
    # Check if documents exist; format them into a single string
    if documents:
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    else:
        context = "No relevant context found in document."
    
    # LLM Answer Generation
    generation = rag_chain.invoke({
        "context": context,
        "question": question
    })
    
    # State update: final answer stored in 'generation' key
    return {"generation": generation}

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

# Add Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile Pipeline
app = workflow.compile()