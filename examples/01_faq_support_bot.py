"""
Example Project 1: FAQ Customer Support Bot (e-commerce style)

Domain: customer support for a fictional phone brand "ZetaPhone".

What it shows:
  - RAG over a small FAQ knowledge base (Chroma, in-memory)
  - Adaptive routing: small talk goes direct, product questions get retrieved
  - Unanswered questions open a "ticket" in MongoDB (optional - works without)

Run:  uv run examples/01_faq_support_bot.py
Deps: OpenAI key only (Mongo optional).
"""

import os
import sys
from pathlib import Path
from typing import Annotated

# make `modules` importable when run as: uv run examples/01_faq_support_bot.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

FAQ_DOCS = [
    "ZetaPhone X1 price: the ZetaPhone X1 costs 499 dollars and the X1 Pro costs 699 dollars.",
    "ZetaPhone battery: all ZetaPhone models have a 5000 mAh battery that lasts about two days of normal use.",
    "ZetaPhone warranty: every ZetaPhone comes with a 2 year warranty covering manufacturing defects, but not water damage.",
    "ZetaPhone colors: the X1 comes in midnight black, arctic silver and sunset orange.",
    "ZetaPhone support: you can reach ZetaPhone support at support@zetaphone.example anytime.",
]


class Route(BaseModel):
    """Structured router decision.

    Attributes:
        go_to: either "retrieve" or "direct".
    """

    go_to: str


class State(BaseModel):
    """Support bot state.

    Attributes:
        messages: the conversation (append via add_messages).
        route: routing decision made by the router node.
        context: retrieved FAQ texts (replaced on each retrieval).
    """

    messages: Annotated[list, add_messages] = []
    route: str = ""
    context: list = []


def connect_mongo():
    """Connects to MongoDB for ticket storage if available.

    Returns:
        A MongoUtil bound to the tickets collection, or None when Mongo
        is unreachable (the bot then just skips ticket logging).
    """
    try:
        from modules.mongo_utils import MongoUtil

        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017").split("?")[0]
        uri += "?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000"
        util = MongoUtil(database="examples", collection="support_tickets", connection_string=uri)
        util.count_all()  # cheap probe
        return util
    except Exception as error:
        print("[mongo] off (" + str(error).split("\n")[0][:60] + "...)")
        return None


embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))
vectorstore = Chroma.from_texts(texts=FAQ_DOCS, embedding=embeddings)
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
router_llm = llm.with_structured_output(Route)
tickets = connect_mongo()


def route_question(state: State) -> dict:
    """Decides whether the question needs the FAQ docs or a direct reply.

    Args:
        state: current state; the newest message is the user question.

    Returns:
        Partial update storing the routing decision.
    """
    question = state.messages[-1].content
    decision = router_llm.invoke(
        "Route this message for a phone-shop support bot. Reply 'retrieve' if it is a "
        "product/support question (price, battery, warranty, colors, contact), "
        "else 'direct' (greetings, chit-chat).\n\nMessage: " + question
    )
    route = "direct" if "direct" in decision.go_to.lower() else "retrieve"
    print("  route ->", route)
    return {"route": route}


def retrieve(state: State) -> dict:
    """Searches the FAQ knowledge base for the newest question.

    Args:
        state: current state with the user question.

    Returns:
        Partial update with the retrieved FAQ texts.
    """
    question = state.messages[-1].content
    docs = vectorstore.similarity_search(question, k=2)
    print("  retrieved", len(docs), "FAQ docs")
    return {"context": [doc.page_content for doc in docs]}


def generate(state: State) -> dict:
    """Answers from the retrieved FAQ context (or admits gaps).

    Args:
        state: current state with question and FAQ context.

    Returns:
        Partial update appending the support answer.
    """
    system_prompt = (
        "You are ZetaPhone support. Answer using ONLY the FAQ context. "
        "If the answer is not there, reply exactly: I don't know - a ticket will be opened.\n\n"
        "FAQ:\n" + "\n\n".join(state.context)
    )
    reply = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)
    return {"messages": [reply]}


def maybe_open_ticket(state: State) -> dict:
    """Opens a support ticket when the bot could not answer.

    Args:
        state: current state ending with the support answer.

    Returns:
        Partial update appending a follow-up message when a ticket is opened.
    """
    answer = state.messages[-1].content
    if "don't know" in answer.lower() or "don’t know" in answer.lower():
        print("  no FAQ answer -> opening ticket")
        if tickets is not None:
            ticket_id = tickets.create({
                "question": state.messages[0].content,
                "status": "open",
            })
            note = "Ticket #" + str(ticket_id)[-6:] + " opened - our team will email you."
        else:
            note = "Ticket noted (Mongo offline, not persisted) - our team will email you."
        return {"messages": [SystemMessage(content=note)]}
    return {}


def direct_answer(state: State) -> dict:
    """Answers chit-chat directly without retrieval.

    Args:
        state: current state with the user message.

    Returns:
        Partial update appending a direct reply.
    """
    reply = llm.invoke(state.messages)
    return {"messages": [reply]}


def after_route(state: State) -> str:
    """Router function after route_question.

    Args:
        state: current state holding the route decision.

    Returns:
        Name of the next node.
    """
    return "direct_answer" if state.route == "direct" else "retrieve"


def build_graph():
    """Builds and compiles the support bot graph.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(State)
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("maybe_open_ticket", maybe_open_ticket)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", after_route)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "maybe_open_ticket")
    graph.add_edge("maybe_open_ticket", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


def ask(app, question: str) -> None:
    """Asks the support bot one question and prints its answer.

    Args:
        app: compiled support bot graph.
        question: the user question.
    """
    print("\nQ:", question)
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print("A:", result["messages"][-1].content)


if __name__ == "__main__":
    app = build_graph()
    ask(app, "hey there, how's your day going?")                 # direct
    ask(app, "how much does the ZetaPhone X1 Pro cost?")         # retrieve
    ask(app, "does the warranty cover water damage?")            # retrieve (edge case)
    ask(app, "can I book a showroom visit in Dubai?")            # no FAQ -> ticket
