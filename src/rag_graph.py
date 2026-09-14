import os
from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.vector_store import VectorStore

load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int


# ==========================================
# 2. LLM & Vector Store Setup
# ==========================================
# Use ChatGroq if available for reliable free tier performance, fallback to Gemini
if os.getenv("GROQ_API_KEY"):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
else:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Load existing FAISS store
vector_store = VectorStore()
if os.path.exists("./vector_db"):
    try:
        vector_store.load("./vector_db")
    except Exception as e:
        print(f"Note: VectorStore load deferred ({e}). Build index first.")


# ==========================================
# 3. Grader & Chains Setup
# ==========================================
class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", grader_system_prompt),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
])

doc_grader = grade_prompt | structured_llm_grader

# Query Rewriter Chain
rewrite_system_prompt = """You are a query re-writer that converts an input question into a better version 
optimized for vectorstore retrieval. Look at the input question and reason about the underlying intent and keywords. 
Output ONLY the raw, improved, concise search query text without any preamble, conversational text, bullet points, or quotes."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system_prompt),
    ("human", "Initial Question:\n\n{question}\n\nFormulate an improved search query:")
])

question_rewriter = re_write_prompt | llm | StrOutputParser()

# Generation Chain
generate_system_prompt = """You are an AI assistant for document question-answering tasks. 
Use the following pieces of retrieved context to answer the user question. 
If you don't know the answer or if the context is insufficient, state that clearly based on the provided document.
Keep your answer clear, accurate, concise, and structured."""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", generate_system_prompt),
    ("human", "Retrieved Context:\n\n{context}\n\nUser Question:\n{question}\n\nAnswer:")
])

rag_chain = generate_prompt | llm | StrOutputParser()


# ==========================================
# 4. Node Definitions
# ==========================================
def retrieve_node(state: GraphState):
    """Retrieves top-k relevant chunks from VectorStore."""
    print("\n--- 1. RETRIEVING DOCUMENTS ---")
    if vector_store.db is None or os.path.exists("./vector_db"):
        try:
            vector_store.load("./vector_db")
        except Exception as e:
            if vector_store.db is None:
                raise RuntimeError("Vector database not found or not initialized yet. Please upload a document and build the index first.") from e

    question = state["question"]
    documents = vector_store.search(question, k=3)
    return {"documents": documents}


def grade_documents_node(state: GraphState):
    """Filters out irrelevant documents using LLM."""
    print("--- 2. CHECKING DOCUMENT RELEVANCE ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for doc in documents:
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
            
    return {"documents": filtered_docs}


def transform_query_node(state: GraphState):
    """Rewrites query if retrieved documents are irrelevant."""
    print("--- 3. TRANSFORMING / REWRITING QUERY ---")
    current_question = state["question"]
    current_retry = state.get("retry_count", 0)
    
    better_question = question_rewriter.invoke({"question": current_question})
    print(f"  Old Query : '{current_question}'")
    print(f"  New Query : '{better_question}'")
    
    return {
        "question": better_question,
        "retry_count": current_retry + 1
    }


def generate_node(state: GraphState):
    """Generates answer using approved relevant documents."""
    print("--- 4. GENERATING FINAL ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    if documents:
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    else:
        context = "No relevant context found in document."
        
    generation = rag_chain.invoke({
        "context": context,
        "question": question
    })
    
    return {"generation": generation}


def decide_to_generate(state: GraphState):
    """Conditional Edge router."""
    filtered_docs = state["documents"]
    retry_count = state.get("retry_count", 0)

    if len(filtered_docs) > 0:
        print("--- DECISION: GENERATE ANSWER ---")
        return "generate"
    elif retry_count < 2:
        print("--- DECISION: REWRITE QUERY ---")
        return "transform_query"
    else:
        print("--- DECISION: MAX RETRIES REACHED, GENERATING ---")
        return "generate"


# ==========================================
# 5. Graph Compilation
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

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

app = workflow.compile()


# ==========================================
# 6. Test Runner
# ==========================================
if __name__ == "__main__":
    test_question = "What is graph database?"
    inputs = {"question": test_question, "retry_count": 0}
    
    print(f"Starting Graph Execution for Question: '{test_question}'\n")
    final_state = app.invoke(inputs)
    
    print("\n================ FINAL ANSWER ================")
    print(final_state["generation"])

