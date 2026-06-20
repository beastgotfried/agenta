from langgraph.graph import END,START, StateGraph

from app.agent.nodes import execute_node, pick_task_node, plan_node, summarize_node
from app.agent.state import AgentState

def route_after_pick(state: AgentState) -> str: #what route the graph goes to post pick_task_node
    """ The loops brain: what to do after picking a task? If no tasks left, or we've hit the loop limit, summarize. Otherwise, execute the next task."""
    if state["current_task"] is None or state["loop_count"] >= state["max_loops"]:
        return "summarize"
    return "execute"

def build_graph(): #creating the graph with nodes and edges making sure it compiles by the agent to be run
    builder= StateGraph(AgentState)
    
    builder.add_node("plan", plan_node)
    builder.add_node("pick_task", pick_task_node)
    builder.add_node("execute", execute_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "pick_task")
    builder.add_conditional_edges("pick_task", route_after_pick, {"execute": "execute", "summarize": "summarize"},
    )
    builder.add_edge("execute", "pick_task")
    builder.add_edge("summarize", END)
    
    return builder.compile()


    