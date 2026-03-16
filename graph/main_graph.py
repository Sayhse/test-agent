# @Time    : 2026/2/25 16:24
# @Author  : Yun
# @FileName: main_graph
# @Software: PyCharm
# @Desc    :
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage

from auto_test_assistant.graph.file_generator_graph import build_skill_agent
from auto_test_assistant.graph.ui_usecase_code_generator_graph import build_code_generator_agent
from auto_test_assistant.state.auto_test_agent_state import TestAgentState
from auto_test_assistant.agents.auto_test_agent import AutoTestAgentSystem

available_nodes = ["file_generation", "ui_use_case_code_generation", "code_review", "code_execution",
                   "error_analysis", "END"]


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


def routing_decision(state: TestAgentState):
    print(">>> routing_decision_node")
    writer = get_stream_writer()
    writer({"node": ">>> routing_decision_node"})
    # writer({"state": state})
    # 根据用户问题 对问题进行分类，分类结果保存到type中
    llm = build_llm()
    system = AutoTestAgentSystem(llm=llm, skills=Path(os.getenv("SKILL_DIR_ROOT")))
    result = system.handle_request(messages=state["messages"])
    # writer({"type_result": result})
    return {"messages": [SystemMessage(content=f"已经完成意图识别，下一任务类型为：{result.type}")],
            "type": result.type, "reason": result.reason, "selected_skills": result.selected_skills}


def parse_file(state: TestAgentState):
    print(">>> parse_file")
    writer = get_stream_writer()
    writer({"node": ">>> parse_file"})
    uploaded_files_metadata = []
    if state["uploaded_files"]:
        for uploaded_file in state["uploaded_files"]:
            path = Path(uploaded_file)
            if path.is_file():
                try:
                    uploaded_file_name = path.name
                except Exception as e:
                    uploaded_file_name = None
                    writer({"error": str(e)})
                uploaded_files_metadata.append({"name": uploaded_file_name, "path": str(uploaded_file)})
    # TODO:判断state["uploaded_file_content"]的长度，如果大于了多少就存进RAG，或者直接压缩？或者总结？
    if len(uploaded_files_metadata) == len(state.get("uploaded_files", [])):
        return {"messages": [SystemMessage(content="用户当前对话上传的所有文件已经全部解析成功")],
                "type": "parse_file", "uploaded_flag": False, "uploaded_files_metadata": uploaded_files_metadata}
    elif len(uploaded_files_metadata) > 0:
        return {"messages": [SystemMessage(content="用户当前对话上传的所有文件仅部分解析成功")], "type": "parse_file",
                "uploaded_flag": False, "uploaded_files_metadata": uploaded_files_metadata}
    else:
        return {"messages": [SystemMessage(content="用户当前对话上传的所有文件已经全部解析失败")],
                "type": "parse_file", "uploaded_flag": False, "uploaded_files_metadata": uploaded_files_metadata}


def file_generation(state: TestAgentState):
    print(">>> file_generation")
    writer = get_stream_writer()
    writer({"node": ">>> file_generation"})
    selected_skills = state.get("selected_skills", [])
    # writer({"selected_skills": selected_skills})
    skill_agent = build_skill_agent()
    config = {
        "configurable": {
            "thread_id": "skill_agent_customer_123"
        }
    }
    for chunk in skill_agent.stream({
        "messages": [state["human_message"]],
        "human_message": state["human_message"],
        "skills": selected_skills,
        "uploaded_files_metadata": state.get("uploaded_files_metadata", []),
        "current_task_idx": 0
    }, config=config, stream_mode="custom", subgraphs=True):
        print(chunk)
    result = skill_agent.get_state(config=config)
    # result = skill_agent.invoke({"messages": [], "selected_skills": selected_skills}, config=config)
    generated_file = result.values.get("generated_file", "")
    return {"messages": [SystemMessage(content=f"生成的文件路径为：{generated_file}")], "type": "file_generation"}


def ui_use_case_code_generation(state: TestAgentState):
    print(">>> ui_use_case_code_generation_node")
    writer = get_stream_writer()
    writer({"node": ">>> ui_use_case_code_generation_node"})
    code_generator_agent = build_code_generator_agent()
    config = {
        "configurable": {
            "thread_id": "skill_agent_customer_123"
        }
    }
    for chunk in code_generator_agent.stream({
        "messages": [state["human_message"]],
        "human_message": state["human_message"],
        "uploaded_files_metadata": state.get("uploaded_files_metadata", []),
        "ui_use_cases": []
    }, config=config, stream_mode="custom", subgraphs=True):
        print(chunk)
    return {"messages": [SystemMessage(content="ui_use_case_code_generation")], "type": "ui_use_case_code_generation"}


def code_review(state: TestAgentState):
    print(">>> code_review_node")
    writer = get_stream_writer()
    writer({"node": ">>> code_review_node"})
    return {"messages": [SystemMessage(content="code_review")], "type": "code_review",
            "reason": "该问题与软件测试无关"}


def code_execution(state: TestAgentState):
    print(">>> code_execution_node")
    writer = get_stream_writer()
    writer({"node": ">>> code_execution_node"})
    return {"messages": [SystemMessage(content="code_execution")], "type": "code_execution",
            "reason": "该问题与软件测试无关"}


def error_analysis(state: TestAgentState):
    print(">>> error_analysis_node")
    writer = get_stream_writer()
    writer({"node": ">>> error_analysis_node"})
    return {"messages": [SystemMessage(content="error_analysis")], "type": "error_analysis",
            "reason": "该问题与软件测试无关"}


def other(state: TestAgentState):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    return {"messages": [SystemMessage(content="我暂时无法回答这个问题")], "type": "other",
            "reason": "该问题与软件测试无关"}


def decide_routing(state: TestAgentState):
    if state["type"] in available_nodes:
        if state["type"] == "END":
            return END
        return state["type"]
    return "other"


def decide_to_parse_file(state: TestAgentState):
    if state.get("uploaded_flag", False):
        return "parse_file"
    return "routing_decision"


def generate_graph():
    # 构建图
    builder = StateGraph(TestAgentState)
    # 添加节点
    builder.add_node("routing_decision", routing_decision)
    builder.add_node("parse_file", parse_file)
    builder.add_node("file_generation", file_generation)
    builder.add_node("ui_use_case_code_generation", ui_use_case_code_generation)
    builder.add_node("code_review", code_review)
    builder.add_node("code_execution", code_execution)
    builder.add_node("error_analysis", error_analysis)
    builder.add_node("other", other)
    # 添加边
    builder.add_conditional_edges(START, decide_to_parse_file, ["routing_decision", "parse_file"])
    # TODO:后续删掉
    builder.add_edge("parse_file", "ui_use_case_code_generation")
    # builder.add_edge("parse_file", "routing_decision")
    builder.add_conditional_edges("routing_decision", decide_routing,
                                  ["parse_file", "file_generation", "ui_use_case_code_generation", "code_review",
                                   "code_execution",
                                   "error_analysis", "other", END])
    builder.add_edge("file_generation", "routing_decision")
    builder.add_edge("ui_use_case_code_generation", "routing_decision")
    builder.add_edge("code_review", "routing_decision")
    builder.add_edge("code_execution", "routing_decision")
    builder.add_edge("error_analysis", "routing_decision")
    builder.add_edge("other", END)
    # 构建graph
    checkpoint = InMemorySaver()
    graph = builder.compile(checkpointer=checkpoint)
    return graph
