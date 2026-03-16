# @Time    : 2026/2/25 17:36
# @Author  : Yun
# @FileName: file_generator_graph
# @Software: PyCharm
# @Desc    :
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command

from auto_test_assistant.agents.auto_test_agent import AutoTestAgentSystem
from auto_test_assistant.graph.task_executor_graph import build_task_executor_graph
from auto_test_assistant.state.file_generator_state import FileGeneratorAgentState


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


def obtain_skill_workflow(state: FileGeneratorAgentState):
    print(">>> obtain_skill_workflow_node")
    writer = get_stream_writer()
    writer({"node": ">>> obtain_skill_workflow_node"})
    skill_docs = []
    for skill in state["skills"]:
        skill_md_path = Path(skill.path) / "SKILL.md"
        try:
            skill_doc = skill_md_path.read_text(encoding="utf-8")
            skill_docs.append(str(skill_doc))
        except Exception as e:
            skill_docs.append(str(skill.description) + f"\n\n（额外信息：读取 SKILL.md 失败: {e}）")
    return {"messages": [SystemMessage(content="已获取所有skill能力说明")], "skill_doc": skill_docs}


def planning_task_list(state: FileGeneratorAgentState):
    print(">>> planning_task_list_node")
    writer = get_stream_writer()
    writer({"node": ">>> planning_task_list_node"})
    # 根据SKILL.md描述的工作流生成 TodoList
    llm = build_llm()
    system = AutoTestAgentSystem(llm=llm, skills=state.get("skills", []))
    result = system.extract_workflow_from_skill_md(state["skill_doc"][0])
    return {"messages": [SystemMessage(content="已生成对应TodoList")], "tasks": result}


def execute_task_list(state: FileGeneratorAgentState):
    print(">>> execute_task_list_node")
    writer = get_stream_writer()
    writer({"node": ">>> execute_task_list_node"})
    writer({"file_state": state.get("single_task_result", [])})
    task_executor = build_task_executor_graph()
    config = {
        "configurable": {
            "thread_id": "task_executor_customer_123"
        }
    }
    # result = task_executor.invoke({
    #     "messages": [state["human_message"]],
    #     "human_message": state["human_message"],
    #     "task": state["tasks"][state["current_task_idx"]],
    #     "history_task_results": state["single_task_result"],
    #     "needs_input": False,
    #     "uploaded_files_metadata": state["uploaded_files_metadata"],
    # }, config=config)
    interrupted = False
    for chunk in task_executor.stream({
        "messages": [state["human_message"]],
        "human_message": state["human_message"],
        "task": state["tasks"][state["current_task_idx"]],
        "history_task_results": state["single_task_result"],
        "needs_input": False,
        "uploaded_files_metadata": state["uploaded_files_metadata"],
    }, config=config, stream_mode="updates", subgraphs=True):
        if "__interrupt__" in chunk:
            print("检测到interrupt:", chunk["__interrupt__"])
            interrupted = True
            break  # 暂停，等待用户输入
    if not interrupted:
        result = task_executor.get_state(config=config)
    else:
        writer({"interrupt": task_executor.get_state(config)})
        question = task_executor.get_state(config).values.get("content", "")
        # user_input = input(f"关于任务[{interrupt.value.get('task', {})["id"]}]：{interrupt.value.get('task', {})["content"]}  存在缺失信息！\n"
        #                    f"请回答：{question}")
        user_input = input(f"请回答：{question}")
        state["messages"].append(SystemMessage(content=question))
        state["messages"].append(SystemMessage(content=f"用户提供的回答：{user_input}"))
        result = task_executor.invoke({
            "messages": [state["human_message"]],
            "human_message": state["human_message"],
            "task": state["tasks"][state["current_task_idx"]],
            "history_task_results": state["single_task_result"],
            "needs_input": False,
            "uploaded_files_metadata": state["uploaded_files_metadata"],
        },
            Command(resume=user_input),
            config=config
        )
    writer({"result": result.values})
    state["tasks"][state["current_task_idx"]] = result.values.get("task", state["tasks"][state["current_task_idx"]])
    # writer({"execute_result": result.get("execution_result", {})})
    return {
        "messages": [
            SystemMessage(
                content=f"任务[{state["current_task_idx"]}]完成，结果为：{result.values.get("execution_result", "")}"
            )
        ],
        "current_task_idx": state["current_task_idx"] + 1,
        "single_task_result": [result.values.get("execution_result", {})]
    }


def should_continue(state: FileGeneratorAgentState):
    # 判断是否还有pending任务，无则END
    for idx, task in enumerate(state["tasks"]):
        if task["status"] == "pending":
            state["current_task_idx"] = idx
            return "execute_task_list"
    return END


def build_skill_agent():
    """
    根据skill构建专项子agent的graph
    """
    builder = StateGraph(FileGeneratorAgentState)
    builder.add_node("obtain_skill_workflow", obtain_skill_workflow)
    builder.add_node("planning_task_list", planning_task_list)
    builder.add_node("execute_task_list", execute_task_list)

    builder.add_edge(START, "obtain_skill_workflow")
    builder.add_edge("obtain_skill_workflow", "planning_task_list")
    builder.add_conditional_edges("planning_task_list", should_continue, ["execute_task_list", END])
    builder.add_conditional_edges("execute_task_list", should_continue, ["execute_task_list", END])
    # builder.add_edge("execute_task_list", END)

    checkpoint = InMemorySaver()
    graph = builder.compile(checkpointer=checkpoint)
    return graph
