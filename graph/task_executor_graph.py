# @Time    : 2026/2/26 21:18
# @Author  : Yun
# @FileName: task_executor_graph
# @Software: PyCharm
# @Desc    :
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from auto_test_assistant.agents.task_executor import TaskExecutorAgentSystem
from auto_test_assistant.state.task_executor_state import TaskExecutorState


def build_llm() -> BaseChatModel:
    """
    构建底层 LLM。
    默认使用 OpenAI 风格接口，你可以根据自己环境替换为其他实现（如自建 LLM 网关）。
    需要环境变量：
    - OPENAI_API_KEY
    - OPENAI_MODEL
    """
    llm = init_chat_model(model="deepseek-chat", model_provider="deepseek", max_tokens=8192)
    return llm


def build_reasoner_llm() -> BaseChatModel:
    """
    构建底层 LLM。
    默认使用 OpenAI 风格接口，你可以根据自己环境替换为其他实现（如自建 LLM 网关）。
    需要环境变量：
    - OPENAI_API_KEY
    - OPENAI_MODEL
    """
    llm = init_chat_model(model="deepseek-reasoner", model_provider="deepseek", max_tokens=8192)
    return llm


def execute_task(state: TaskExecutorState):
    print(">>> execute_task_node")
    writer = get_stream_writer()
    writer({"node": ">>> execute_task_node"})
    writer({"task_executor_state": state})
    llm = build_llm()
    system = TaskExecutorAgentSystem(llm=llm)
    result = system.execute(state)
    if result.get("needs_input", False):
        question = result.get("question", "")
        user_answer = interrupt({
            "instruction": "缺失信息",
            # "task": state["task"],
            "content": question,
        })
        state["user_answer"] = user_answer
        state["question"] = question
        state["needs_input"] = True
        result = system.execute(state)
    return {
        "messages": [
            SystemMessage(
                content=f"任务 {state["task"]["content"]} 执行完成！\n"
                        f"执行结果：{result.get("execution_summary_result", {})}"
            )
        ],
        "execution_result": {
            "execution_summary_result": result.get("execution_summary_result", {}),
            "generated_files": result.get("generated_files", []),
        },
        "task": result.get("task", state["task"]),
        "needs_input": result.get("needs_input", False),
        "question": result.get("question", ""),
        "user_answer": result.get("user_answer", ""),
    }


def review_plan(state: TaskExecutorState):
    print(">>> review_plan_node")
    writer = get_stream_writer()
    writer({"node": ">>> review_plan_node"})
    writer({"task_executor_state": state})
    llm = build_reasoner_llm()
    system = TaskExecutorAgentSystem(llm=llm)
    result = system.review(state)
    writer({"result": result})
    return {"messages": [SystemMessage(content="review_plan")], "type": "review_plan",
            "reason": "该问题与软件测试无关"}


def judge_review_necessity(state: TaskExecutorState):
    llm = build_llm()
    system = TaskExecutorAgentSystem(llm=llm)
    result = system.is_need_review(state)
    if result == "END":
        return END
    return END


def build_task_executor_graph():
    """
    根据任务列表执行任务的执行器
    """
    builder = StateGraph(TaskExecutorState)
    builder.add_node("execute_task", execute_task)
    builder.add_node("review_plan", review_plan)

    builder.add_edge(START, "execute_task")
    builder.add_conditional_edges("execute_task", judge_review_necessity, ["review_plan", END])
    builder.add_edge("review_plan", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
