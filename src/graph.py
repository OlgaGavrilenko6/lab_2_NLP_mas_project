from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import MASState, init_state
from .nodes import (
    router_node,
    planner_node,
    gather_tools_node,
    conceptual_agent_node,
    architecture_agent_node,
    coding_agent_node,
    daily_agent_node,
    literature_agent_node,
    reviewer_node,
    finalize_node,
    route_after_planner,
    route_after_reviewer,
)
from IPython.display import Image, display, Markdown

# Визуализация графа
def build_graph_with_retry_loop():
    g = StateGraph(MASState)

    g.add_node("router", router_node)
    g.add_node("planner", planner_node)

    g.add_node("gather_tools", gather_tools_node)

    g.add_node("conceptual_agent", conceptual_agent_node)
    g.add_node("architecture_agent", architecture_agent_node)
    g.add_node("coding_agent", coding_agent_node)
    g.add_node("daily_agent", daily_agent_node)
    g.add_node("literature_agent", literature_agent_node)

    g.add_node("reviewer", reviewer_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("router")
    g.add_edge("router", "planner")

    # planner -> gather_tools (первичный добор)
    g.add_edge("planner", "gather_tools")

    # gather_tools -> chosen agent (handoff по intent)
    g.add_conditional_edges(
        "gather_tools",
        lambda s: s.get("intent") or "daily",
        {
            "conceptual": "conceptual_agent",
            "architecture": "architecture_agent",
            "coding": "coding_agent",
            "daily": "daily_agent",
            "literature": "literature_agent",
        }
    )

    # Агент -> ревьюер
    for node in ["conceptual_agent","architecture_agent","coding_agent","daily_agent","literature_agent"]:
        g.add_edge(node, "reviewer")

    # Ревьюер -> интрумент
    g.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"gather_tools": "gather_tools", "finalize": "finalize"}
    )

    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())

def show_graph(app):
    g = app.get_graph()
    try:
        display(Image(g.draw_mermaid_png()))
    except Exception:
        display(Markdown(g.draw_mermaid()))


# Граф
app = build_graph_with_retry_loop()
show_graph(app)