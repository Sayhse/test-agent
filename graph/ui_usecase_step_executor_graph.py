# @Time    : 2026/3/4 14:44
# @Author  : Yun
# @FileName: ui_use_case_step_executor_graph
# @Software: PyCharm
# @Desc    :
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from auto_test_assistant.agents.code_generator_agent import CodeGeneratorAgentSystem
from auto_test_assistant.state.ui_usecase_step_executor_state import UiUseCaseStepExecutorState


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


def execute_use_case_by_path(state: UiUseCaseStepExecutorState):
    print(">>> execute_use_case_by_path_node")
    writer = get_stream_writer()
    writer({"node": ">>> execute_use_case_by_path_node"})
    use_cases = state.get("ui_use_cases", [])
    path = state.get("path", [])
    script_path = state.get("script_path", f"./scripts/use_case_{state.get("ui_use_case", {})["use_case_id"]}.py")

    current_use_case = state.get("current_use_case", {})
    steps = current_use_case.get("use_case_steps", [])
    writer({"steps": steps})
    llm = build_llm()
    system = CodeGeneratorAgentSystem(llm)

    while current_use_case["use_case_id"] != state.get("ui_use_case", {})["use_case_id"]:
        # 执行当前这个use_case
        system.create_agent(steps, script_path)
        while len(steps) > state.get("current_step", 0):
            # 当前用例还没执行完，继续执行
            current_step = steps[state.get("current_step", 0)]
            step_id = current_step.get("id", "")
            if step_id:
                result = system.execute_use_case_step(current_use_case.get("use_case_id", ""), step_id)
            state["current_step"] += 1
        # 当前用例执行完了，切用例
        state["current_use_case_idx"] += 1
        state["current_use_case"] = next((item for item in use_cases if item.get("use_case_id") == path[state["current_use_case_idx"]]), None)
        current_use_case = state.get("current_use_case", {})
        steps = current_use_case.get("use_case_steps", [])
        state["current_step"] = 0
    # 最后本土用例的执行
    system.create_agent(steps, script_path)
    while len(steps) > state.get("current_step", 0):
        current_step = steps[state.get("current_step", 0)]
        step_id = current_step.get("id", "")
        if step_id:
            result = system.execute_use_case_step(current_use_case.get("use_case_id", ""), step_id)
        state["current_step"] += 1
    return {"messages": [SystemMessage(content="execute_use_case_by_path")], "type": "execute_use_case_by_path",
            "reason": "该问题与软件测试无关"}


def review_use_case_step(state: UiUseCaseStepExecutorState):
    print(">>> review_use_case_step_node")
    writer = get_stream_writer()
    writer({"node": ">>> review_use_case_step_node"})
    return {"messages": [SystemMessage(content="review_use_case_step")], "type": "review_use_case_step",
            "reason": "该问题与软件测试无关"}


def judge_qualified(state: UiUseCaseStepExecutorState):
    pass


def build_usecase_step_executor_agent():
    """
    根据用例步骤描述生成代码的专项子agent的graph
    """
    builder = StateGraph(UiUseCaseStepExecutorState)
    builder.add_node("execute_use_case_step", execute_use_case_by_path)
    builder.add_node("review_use_case_step", review_use_case_step)

    builder.add_edge(START, "execute_use_case_step")
    builder.add_edge("execute_use_case_step", "review_use_case_step")
    builder.add_conditional_edges("review_use_case_step", judge_qualified, ["execute_use_case_step", END])

    checkpoint = InMemorySaver()
    graph = builder.compile(checkpointer=checkpoint)
    return graph
