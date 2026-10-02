"""
LangGraph API Reference - core functions, classes, parameters, returns.

This file is DOCUMENTATION, not implementation: every definition below is a
stub that mirrors the real LangGraph API and documents it in Google style.
Read it top to bottom once - after that, whenever you wonder "what can I
pass to compile()?" or "what does get_state() return?", come here.

Real import paths are noted in each docstring. Cross-references like
"used in lessons/03" point to the runnable example that uses the feature.

Contents:
    1. Constants            START, END
    2. Reducers             add_messages
    3. Graph building       StateGraph (add_node / add_edge /
                            add_conditional_edges / compile)
    4. Running graphs       invoke / stream / ainvoke / astream
    5. Inspecting state     get_state / get_state_history / update_state
    6. Runtime control      Command, recursion limits, GraphRecursionError
    7. Agents               ToolNode, create_agent (modern API)
    8. Checkpointers        InMemorySaver, SqliteSaver
    9. LangChain companions @tool, bind_tools, with_structured_output
   10. Config dicts         thread_id, checkpoint_id, recursion_limit
"""

from typing import Any, AsyncIterator, Callable, Iterator, Literal, Optional, Sequence, Union

# =====================================================================
# 1. CONSTANTS
# =====================================================================

START: str
"""Virtual node marking the graph entry. Target it in add_edge to choose
where execution begins: ``graph.add_edge(START, "my_first_node")``.
Import: ``from langgraph.graph import START``. Used in every lesson."""

END: str
"""Virtual node marking the graph exit. A router may return it to finish
(``return END``), or an edge may point to it. Import:
``from langgraph.graph import END``. Used in every lesson."""


# =====================================================================
# 2. REDUCERS
# =====================================================================

def add_messages(left: list, right: list) -> list:
    """Reducer that APPENDS messages instead of overwriting.

    Attach it to a state key so every node's return value for that key is
    merged by concatenation (with smart handling of message ids), rather
    than the default "last write wins".

    Import:
        ``from langgraph.graph.message import add_messages``

    Args:
        left: the messages already in state.
        right: the new messages a node returned.

    Returns:
        The combined message list (left + right, id-deduplicated).

    Example:
        ``messages: Annotated[list, add_messages] = []``  # lessons/02+

    Note:
        For plain lists of anything else (parallel writes), use
        ``operator.add`` as the reducer instead - see pro_topics/03.
    """


# =====================================================================
# 3. GRAPH BUILDING - StateGraph
# =====================================================================

class StateGraph:
    """Blueprint for a graph; ``compile()`` turns it into a runnable.

    Import:
        ``from langgraph.graph import StateGraph``

    The state schema may be a Pydantic ``BaseModel``, a ``TypedDict``, or a
    dataclass (this repo uses Pydantic - see lessons/01).
    """

    def __init__(self, state_schema: type) -> None:
        """Creates an empty graph bound to a state schema.

        Args:
            state_schema: the class (BaseModel/TypedDict/dataclass) whose
                fields define the shared state keys and their merge rules.
        """

    def add_node(self, name: str, action: Callable) -> None:
        """Registers a node under a string name.

        Args:
            name: the node's identifier - routers and edges refer to THIS
                string, not to the function name.
            action: one of:
                - a function ``(state) -> dict`` returning only the keys
                  to change (the normal case),
                - a Runnable (e.g. an LLM directly),
                - an entire compiled graph (subgraph pattern, lessons/10).
        """

    def add_edge(self, start: str, end: str) -> None:
        """Wires a FIXED edge: whenever ``start`` finishes, run ``end``.

        Args:
            start: a node name, or START for the entry edge. May also be a
                list of names to fan out from one source (pro_topics/03).
            end: a node name, or END to finish.
        """

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[[Any], Union[str, list]],
        path_map: Optional[dict] = None,
    ) -> None:
        """Wires a RUNTIME-DECIDED edge: after ``source``, call ``path``.

        Args:
            source: the node whose exit is being routed.
            path: a function ``(state) -> node name | END | list of names``.
                Usually a plain if-statement on state (lessons/03); can be
                an LLM call with structured output (lessons/07).
            path_map: optional dict mapping the values ``path`` returns to
                node names - useful when path returns enums/bools instead
                of exact node-name strings.
        """

    def compile(
        self,
        checkpointer: Optional[Any] = None,
        interrupt_before: Optional[list] = None,
        interrupt_after: Optional[list] = None,
        store: Optional[Any] = None,
    ) -> "CompiledGraph":
        """Validates the blueprint and returns a runnable.

        Args:
            checkpointer: saves state after every step and loads it back on
                each invoke by thread_id (lessons/05). None = no memory.
            interrupt_before: node names to PAUSE before executing
                (human-in-the-loop, lessons/08). Requires a checkpointer.
            interrupt_after: same, but pauses after the listed nodes.
            store: long-term key-value memory across threads (advanced).

        Returns:
            A compiled runnable with invoke/stream/get_state/etc. below.

        Raises:
            ValueError: on invalid wiring (unknown node names, etc.).
        """


# =====================================================================
# 4. RUNNING GRAPHS (methods of the object compile() returns)
# =====================================================================

class CompiledGraph:
    """The runnable returned by ``StateGraph.compile()``.

    Behaves like a LangChain Runnable: ``invoke``, ``stream``, ``batch``
    all exist. Below are the LangGraph-specific essentials.
    """

    def invoke(self, input: Optional[dict], config: Optional[dict] = None) -> dict:
        """Runs the whole graph once and waits for the final state.

        Args:
            input: the input dict. WITHOUT a checkpointer this is the full
                initial state; WITH a checkpointer it is only the NEW
                values (usually just the new message) - history loads
                from the thread (lessons/05). ``None`` means "no new
                input, resume where paused" (lessons/08 approval path).
            config: see section 10 (thread_id, recursion_limit, ...).

        Returns:
            The final state as a plain dict of key -> value (even when the
            schema is Pydantic - lessons/01). Includes an ``__interrupt__``
            entry when an interrupt fired (lessons/08).

        Raises:
            GraphRecursionError: if steps exceed recursion_limit.
        """

    def stream(
        self,
        input: Optional[dict],
        config: Optional[dict] = None,
        stream_mode: Union[str, list] = "values",
    ) -> Iterator[Any]:
        """Runs the graph, yielding events as they happen (lessons/09).

        Args:
            input: same semantics as invoke().
            config: see section 10.
            stream_mode: one or a list of:
                - ``"values"``: yields the FULL state dict after each step.
                - ``"updates"``: yields ``{node_name: changed_keys_dict}``
                  per step - tells you WHICH node just ran.
                - ``"messages"``: yields ``(message_chunk, metadata)``
                  tuples - LLM tokens as they generate; metadata includes
                  ``"langgraph_node"``.

        Yields:
            Depends on stream_mode (see above).

        Raises:
            GraphRecursionError: if steps exceed recursion_limit.
        """

    async def ainvoke(self, input: Optional[dict], config: Optional[dict] = None) -> dict:
        """Async twin of invoke() for async graphs (pro_topics/04).

        Args:
            input: same semantics as invoke().
            config: see section 10.

        Returns:
            The final state dict.
        """

    async def astream(
        self,
        input: Optional[dict],
        config: Optional[dict] = None,
        stream_mode: Union[str, list] = "values",
    ) -> AsyncIterator[Any]:
        """Async twin of stream() (pro_topics/04).

        Args:
            input: same semantics as invoke().
            config: see section 10.
            stream_mode: same modes as stream().

        Yields:
            Same shapes as stream(), awaited asynchronously.
        """


# =====================================================================
# 5. INSPECTING AND EDITING STATE (require a checkpointer)
# =====================================================================

class CompiledGraphInspectMixin:
    """State inspection methods (on the same compiled object)."""

    def get_state(self, config: dict) -> "StateSnapshot":
        """Reads the saved state of one thread (lessons/05, 08).

        Args:
            config: must contain ``{"configurable": {"thread_id": ...}}``.

        Returns:
            A StateSnapshot - see the StateSnapshot stub below.
        """

    def get_state_history(self, config: dict) -> Iterator["StateSnapshot"]:
        """Walks EVERY saved checkpoint of a thread, newest first
        (pro_topics/02 time travel).

        Args:
            config: thread config (a checkpoint_id may be included to
                start the walk from that point).

        Yields:
            StateSnapshot objects, newest to oldest.
        """

    def update_state(
        self,
        config: dict,
        values: dict,
        as_node: Optional[str] = None,
    ) -> dict:
        """Writes into a thread's state from outside the graph
        (lessons/08 rejection path).

        Args:
            config: the thread to update.
            values: state update dict, merged like a node return.
            as_node: attribute the write to this node - the graph then
                resumes from AFTER that node (this is how you skip a
                paused node: pretend it ran and provide its output).

        Returns:
            Info dict with the resulting config (next checkpoint id).
        """


class StateSnapshot:
    """What get_state()/get_state_history() return (lessons/05, 08).

    Attributes:
        values: dict of the state's current key -> value (e.g.
            ``values["messages"]``). Empty dict when the thread is new.
        next: tuple of node names waiting to run - non-empty when the
            graph is PAUSED at an interrupt (``next == ("call_tools",)``).
        config: the checkpoint config (contains checkpoint_id).
        tasks: task objects with pending writes/interrupts (advanced).
    """


# =====================================================================
# 6. RUNTIME CONTROL
# =====================================================================

class Command:
    """A node return that updates state AND chooses the next node
    (pro_topics/01 - the modern alternative to node + router pairs).

    Import:
        ``from langgraph.types import Command``

    Args (constructor):
        goto: next node name, a list of names, or END.
        update: state update dict applied before moving (optional).
        graph: name of a subgraph to delegate to (advanced, optional).
        resume: value to resume an ``interrupt()`` with (advanced).

    Example:
        ``return Command(goto="review", update={"draft": text})``
    """

    def __init__(self, goto: Union[str, list], update: Optional[dict] = None) -> None:
        """See the class docstring for parameter meanings."""


class GraphRecursionError(Exception):
    """Raised when a run exceeds recursion_limit steps (pro_topics/05).

    Default limit is 25. Raise-proof your loop: give every backwards edge
    a stopping condition (lessons/03), and catch this where relevant.
    """


# =====================================================================
# 7. AGENTS
# =====================================================================

class ToolNode:
    """Prebuilt node that executes the model's tool calls automatically
    (lessons/04 part 2).

    Import:
        ``from langgraph.prebuilt import ToolNode``

    Behavior: reads ``tool_calls`` from the last AI message, finds each
    function by name, runs it, returns ToolMessages with matching
    ``tool_call_id``s. Handles multiple/parallel calls in one reply.
    """

    def __init__(self, tools: Sequence, handle_tool_errors: Union[bool, Callable] = True) -> None:
        """Creates the executor node.

        Args:
            tools: the tool list given to bind_tools (same objects).
            handle_tool_errors: when True (langgraph.prebuilt default),
                exceptions become error-text ToolMessages so the model can
                react. NOTE: the modern create_agent does NOT wrap tools
                this way - prefer error-as-text returns there
                (pro_topics/05).
        """

    def __call__(self, state: Any) -> dict:
        """Executes pending tool calls; usable directly as a node.

        Args:
            state: state whose last message carries tool_calls.

        Returns:
            ``{"messages": [ToolMessage, ...]}``.
        """


def create_agent(
    model: Any,
    tools: Optional[Sequence] = None,
    system_prompt: Optional[str] = None,
    middleware: Sequence = (),
    response_format: Any = None,
    checkpointer: Any = None,
    interrupt_before: Optional[list] = None,
    interrupt_after: Optional[list] = None,
) -> CompiledGraph:
    """The MODERN one-call agent builder (pro_topics/06, examples/02).

    Import:
        ``from langchain.agents import create_agent``  (langchain package)

    Replaces the deprecated ``langgraph.prebuilt.create_react_agent``
    (which used ``prompt=`` instead of ``system_prompt=``).

    Args:
        model: a BaseChatModel (e.g. ChatOpenAI) - the RAW model; it calls
            bind_tools internally, so do not wrap it with .with_retry().
        tools: @tool-decorated functions the model may call.
        system_prompt: persona/instructions as a string.
        middleware: hooks around model & tool calls (logging, guardrails,
            approval) - the big new capability of this API.
        response_format: a Pydantic class to force a structured final
            output instead of a free-text answer.
        checkpointer: same as compile()'s - gives thread memory.
        interrupt_before / interrupt_after: same as compile()'s.

    Returns:
        A compiled graph: invoke with ``{"messages": [...]}``, get back
        ``{"messages": [...]}`` where the last item is the final answer.
    """


# =====================================================================
# 8. CHECKPOINTERS
# =====================================================================

class InMemorySaver:
    """RAM-only checkpointer - memory dies with the process (lessons/05).

    Import:
        ``from langgraph.checkpoint.memory import InMemorySaver``
    """

    def __init__(self) -> None:
        """No parameters."""


class SqliteSaver:
    """File-backed checkpointer - memory survives restarts (lessons/11).

    Import:
        ``from langgraph.checkpoint.sqlite import SqliteSaver``
        (requires the langgraph-checkpoint-sqlite package)
    """

    @classmethod
    def from_conn_string(cls, path: str) -> "SqliteSaver":
        """Opens (or creates) a sqlite checkpoint file.

        Args:
            path: database file path, e.g. "chat_memory.db".

        Returns:
            A checkpointer - use as a context manager for clean close:
            ``with SqliteSaver.from_conn_string("f.db") as cp: ...``
        """


# =====================================================================
# 9. LANGCHAIN COMPANIONS (used inside every graph)
# =====================================================================

def tool(func_or_callable: Callable) -> Any:
    """Decorator turning a function into an agent tool (lessons/04).

    Import:
        ``from langchain_core.tools import tool``

    Contract: type hints become the argument schema; the DOCSTRING is the
    description the model reads (first line = summary; Google-style Args
    sections feed parameter descriptions). The model only ever sees this
    metadata - your code runs locally when the executor invokes it.
    """

def bind_tools(tools: Sequence) -> Any:
    """ChatModel method: ``llm.bind_tools(tools)`` (lessons/04).

    Returns a model copy that advertises the tools' schemas on every call;
    replies may then carry ``reply.tool_calls`` (list of
    ``{"name": str, "args": dict, "id": str}``; empty list = no call).
    """

def with_structured_output(schema: type) -> Any:
    """ChatModel method: ``llm.with_structured_output(SomePydanticClass)``
    (lessons/07).

    Args (via method call):
        schema: a Pydantic class describing the desired output fields.

    Returns:
        A runnable whose invoke(prompt) yields an INSTANCE of that class -
        perfect for routers, graders, and data extraction (examples/04).
    """


# =====================================================================
# 10. CONFIG DICTS (second argument of invoke/stream)
# =====================================================================

# The full shape, annotated:
RUNTIME_CONFIG_EXAMPLE: dict = {
    "configurable": {
        "thread_id": "chat-1",        # REQUIRED with a checkpointer:
                                       #   one id = one conversation
        "checkpoint_id": "abc123...",  # OPTIONAL: resume/inspect a specific
                                       #   saved step (pro_topics/02)
    },
    "recursion_limit": 25,             # OPTIONAL: max supersteps before
                                       #   GraphRecursionError (default 25)
}
"""Annotated example of the runtime config dict - see lessons/05 (thread),
pro_topics/02 (checkpoint_id), pro_topics/05 (recursion_limit)."""
