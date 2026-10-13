"""
Lesson 7: Adaptive RAG - the graph DECIDES how to answer

Every app so far ran the same steps no matter what. Real RAG adapts:
  - Does this question even need the docs, or can the model answer directly?
  - Are the retrieved docs actually relevant to the question?
  - If not: rewrite the query and try again (a loop with a guard - lesson 3!)

One new LangChain tool today (you know it from your LangChain days):
  llm.with_structured_output(SomePydanticClass)
The LLM's answer comes back as an object with typed fields instead of
free text - perfect for router and grader decisions.

Flow:
  route_question -> direct_answer -> END                (small talk etc)
                  -> retrieve -> grade_docs -> generate -> END
                                     |            ^
                                     v (irrelevant)     |
                                 rewrite_query ----------   (max 2 rewrites)
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

from modules.llm import get_chat_model, get_embeddings

MAX_REWRITES = 2

# ---------- knowledge base (same idea as lesson 6) ----------
LESSON_DOCS = [
    "Lesson 1 taught the StateGraph core: a State (shared data as a dict), nodes (plain functions that read the state and return only the keys they change), and edges (the wiring that fixes the order). START and END mark the entry and exit, compile() makes the blueprint runnable, and invoke() runs it once.",
    "Lesson 2 put an LLM inside a node and introduced the messages key with the add_messages reducer: when nodes return messages they are APPENDED instead of overwriting, which is how conversation history accumulates in LangGraph.",
    "Lesson 3 covered conditional edges: a router function reads the state and returns the name of the next node (or END). Routers enable branching and loops - edges can point backwards - and every loop needs a stopping condition such as an attempts counter.",
    "Lesson 4 built a tool-calling agent: functions decorated with @tool, llm.bind_tools(tools), the model requesting tools through reply.tool_calls, ToolMessage results paired by tool_call_id, and the agent loop model -> tools -> model until the model gives a final answer. The prebuilt ToolNode executes tool calls automatically.",
    "Lesson 5 added memory with a checkpointer: compile(checkpointer=InMemorySaver()) saves state after every step, each thread_id is its own conversation, and invoke only needs the new message because earlier history is loaded from the checkpoint.",
    "Repo conventions: state schemas use Pydantic BaseModel, code runs in WSL Ubuntu with uv, model names come from the .env file (OPENAI_MODEL and OPENAI_EMBEDDINGS_MODEL), and lessons are numbered python files at the repo root.",
]

embeddings = get_embeddings()
vectorstore = Chroma.from_texts(texts=LESSON_DOCS, embedding=embeddings)

# ---------- state ----------
class State(BaseModel):
    """Adaptive RAG state: the decision fields drive the routing.

    Attributes:
        messages: the conversation (append via add_messages).
        question: the CURRENT search question; rewrites replace it.
        route: routing decision, "retrieve" or "direct".
        context: retrieved doc texts (replaced each retrieval).
        relevant: the grader's verdict on the docs.
        rewrites: rewrite counter - the retry loop guard.
    """

    messages: Annotated[list, add_messages] = []  # the conversation
    question: str = ""      # the CURRENT search question (rewrites replace it)
    route: str = ""         # "retrieve" or "direct"
    context: list = []      # retrieved doc texts
    relevant: bool = False  # grader verdict
    rewrites: int = 0       # loop guard

llm = get_chat_model()

# ---------- structured-output decision makers ----------
class Route(BaseModel):
    """Structured router decision.

    Attributes:
        go_to: either "retrieve" or "direct".
    """

    go_to: str   # "retrieve" or "direct"

class Grade(BaseModel):
    """Structured grader decision.

    Attributes:
        relevant: whether the context is useful for the question.
    """

    relevant: bool

router_llm = llm.with_structured_output(Route)
grader_llm = llm.with_structured_output(Grade)

# ---------- nodes ----------
def route_question(state: State):
    """LLM router: does this question need the docs or not?

    Args:
        state: current state; the last message is the user question.

    Returns:
        Partial update with the question and the route decision.
    """
    question = state.messages[-1].content
    decision = router_llm.invoke(
        "You route questions for a chatbot about a LangGraph learning repo.\n"
        "Should this question be answered from the repo's lesson documents ('retrieve')\n"
        "or answered directly without documents ('direct', e.g. greetings, general chat)?\n\n"
        "Question: " + question
    )
    route = "direct" if "direct" in decision.go_to.lower() else "retrieve"
    print("route_question -> " + route)
    return {"question": question, "route": route}

def retrieve(state: State):
    """Searches the vector store with the current (possibly rewritten) question.

    Args:
        state: current state with the active question.

    Returns:
        Partial update replacing the context.
    """
    docs = vectorstore.similarity_search(state.question, k=2)
    texts = [doc.page_content for doc in docs]
    print("retrieve: " + str(len(texts)) + " docs for: " + state.question[:50])
    return {"context": texts}

def grade_docs(state: State):
    """LLM grader: are the retrieved docs useful for the question?

    Args:
        state: current state with question and context.

    Returns:
        Partial update with the relevance verdict.
    """
    context_text = "\n\n".join(state.context)
    grade = grader_llm.invoke(
        "Is this context useful for answering the question? Answer true or false.\n\n"
        "Context:\n" + context_text + "\n\nQuestion: " + state.question
    )
    print("grade_docs -> relevant: " + str(grade.relevant))
    return {"relevant": grade.relevant}

def rewrite_query(state: State):
    """Rewrites the question for better semantic search, bumping the guard.

    Args:
        state: current state with the failed question.

    Returns:
        Partial update with the new question and rewrites+1.
    """
    reply = llm.invoke(
        "Rewrite this question to work better for semantic search over lesson notes. "
        "Reply with ONLY the new question.\n\nQuestion: " + state.question
    )
    print("rewrite_query -> " + reply.content)
    return {"question": reply.content, "rewrites": state.rewrites + 1}

def generate(state: State):
    """Answers from the retrieved context (or admits it doesn't know).

    Args:
        state: current state with question, context and conversation.

    Returns:
        Partial update appending the grounded answer.
    """
    context_text = "\n\n".join(state.context)
    system_prompt = (
        "You answer questions about a LangGraph learning repo. "
        "Use ONLY the context below. If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + context_text
    )
    reply = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
    print("generate: answered")
    return {"messages": [reply]}

def direct_answer(state: State):
    """Answers without retrieval (greetings, general chat).

    Args:
        state: current state; the last message is the user question.

    Returns:
        Partial update appending the direct reply.
    """
    reply = llm.invoke(state.messages)
    print("direct_answer: answered without retrieval")
    return {"messages": [reply]}

# ---------- routers (lesson 3) ----------
def after_route(state: State):
    """Router after route_question.

    Args:
        state: current state holding the route decision.

    Returns:
        Name of the next node.
    """
    if state.route == "direct":
        return "direct_answer"
    return "retrieve"

def after_grade(state: State):
    """Router after grade_docs: generate, rewrite, or give up gracefully.

    Args:
        state: current state with the relevance verdict and rewrite count.

    Returns:
        Name of the next node ("generate" or "rewrite_query").
    """
    if state.relevant:
        return "generate"
    if state.rewrites < MAX_REWRITES:
        return "rewrite_query"   # loop back with a better question
    return "generate"            # give up gracefully; generate will say it doesn't know

# ---------- graph ----------
graph = StateGraph(State)
graph.add_node("route_question", route_question)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_docs", grade_docs)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate", generate)
graph.add_node("direct_answer", direct_answer)

graph.add_edge(START, "route_question")
graph.add_conditional_edges("route_question", after_route)
graph.add_edge("retrieve", "grade_docs")
graph.add_conditional_edges("grade_docs", after_grade)
graph.add_edge("rewrite_query", "retrieve")   # backwards edge = retry loop
graph.add_edge("generate", END)
graph.add_edge("direct_answer", END)

app = graph.compile()

# ---------- demo: three questions, three different paths ----------
def ask(question):
    print()
    print("=" * 60)
    print("Q: " + question)
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print("A: " + result["messages"][-1].content)

ask("Hey! Not a question about lessons - just say hi back.")          # -> direct
ask("How does the agent loop know when to stop calling tools?")       # -> retrieve -> relevant -> generate
ask("Which lesson covered deploying LangGraph to Kubernetes?")        # -> retrieve -> (grade says no) -> rewrite -> ... -> "don't know"
