# @Time    : 2026/2/27 9:39
# @Author  : Yun
# @FileName: task_executor
# @Software: PyCharm
# @Desc    :
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
import json
import tempfile
from pathlib import Path

from ..utils.tools import list_all_mcp_tools, read_file_tool, edit_file_tool, question_tool


class TaskExecutorAgentSystem:
    def __init__(
            self,
            llm: BaseChatModel,
    ) -> None:
        self.llm = llm
        self.tools = list_all_mcp_tools()

    def execute(self, state):
        # 从状态中提取必要信息
        task = state.get("task", {})
        task_content = task.get("content", "")
        uploaded_files_metadata = state.get("uploaded_files_metadata", [])
        history_task_results = state.get("history_task_results", [])
        needs_input = state.get("needs_input", False)
        question = state.get("question", "")
        user_answer = state.get("user_answer", "")
        human_message = state.get("human_message")
        human_message_content = human_message.content if human_message else ""

        # 定义JSON Schema
        execute_result_schema = {
            "type": "object",
            "description": "任务执行结果",
            "properties": {
                "generated_files": {
                    "type": "array",
                    "description": "生成的中间文件列表，包括在执行过程中创建或修改的所有文件",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "文件名"},
                            "path": {"type": "string", "description": "文件绝对路径"}
                        },
                        "required": ["name", "path"]
                    }
                },
                "summary": {
                    "type": "string",
                    "description": "总结性结果，简要描述任务执行情况和主要发现"
                },
                "status": {
                    "type": "string",
                    "description": "任务完成状态",
                    "enum": ["completed", "failure", "partial_completed", "skipped", "pending", "in_progress"]
                },
                "needs_input": {
                    "type": "boolean",
                    "description": "是否需要用户输入，如果结果中包含[question]标记或需要用户提供缺失信息，则返回true"
                },
                "question": {
                    "type": "string",
                    "description": "如果需要用户输入，向用户提出的具体问题"
                }
            },
            "required": ["generated_files", "summary", "status", "needs_input", "question"]
        }

        # 构建系统提示
        system_prompt = (
            "你是一个任务执行助手，负责执行具体的任务。"
            "重要指令：这是多步骤任务中的一个步骤，你只需要执行当前描述的任务，不要执行后续步骤，也不要提前生成最终输出文件。\n"
            "你可以使用以下工具来帮助你完成任务：\n"
            f"{self.tools}\n"
            "当前任务描述：\n"
            f"{task_content}\n"
            "请严格按照当前任务描述执行，只完成这个任务要求的具体操作。\n"
            "如果任务要求分析文档，只进行分析并提取信息，并生成中间文档将分析结果放入文档中。\n"
            "如果任务要求生成文档，只生成该任务指定的文档部分。\n"
            "记住：你只需要完成当前步骤，后续步骤会有其他代理执行。\n"
            "\n"
            "输出要求：\n"
            "你必须输出严格符合JSON Schema的JSON格式结果。JSON必须包含以下字段：\n"
            "1. generated_files: 生成的中间文件列表，每个文件包含name（文件名）和path（文件绝对路径）字段（如果创建或修改了文件）\n"
            "2. summary: 总结性结果，简要描述任务执行情况和主要发现\n"
            "3. status: 任务完成状态（success/failure/partial_success/skipped/pending/in_progress）\n"
            "4. needs_input: 是否需要用户输入（如果任务需要用户提供信息或确认）\n"
            "5. question: 如果需要用户输入，具体的问题内容\n"
            "\n"
            "注意：\n"
            "- 如果使用了write工具创建文件，请在generated_files中记录文件名和文件路径\n"
            "- 如果需要参考历史任务生成的文件，可以使用read工具读取历史任务生成的文件路径\n"
            "- 如果任务执行中遇到问题需要用户澄清，请设置needs_input为true并提供question\n"
            "- 如果任务成功完成，请设置status为success\n"
            "- 如果任务部分完成，请设置status为partial_success\n"
            "- 如果任务无法执行，请设置status为failure\n"
        )

        if needs_input:
            system_prompt += (f"用户补充信息如下：\n"
                              f"question：{question}\n"
                              f"answer: {user_answer}\n")

        # 添加上传的文件元数据信息（如果有）
        if uploaded_files_metadata:
            files_info = []
            for file_meta in uploaded_files_metadata:
                file_name = file_meta.get("name", "未命名文件")
                file_path = file_meta.get("path", "")
                files_info.append(f"- {file_name}: 路径 {file_path}")

            if files_info:
                files_context = "\n".join(files_info)
                system_prompt += f"\n用户上传的文件信息：\n{files_context}\n"
                system_prompt += "提示：如果需要分析文件内容，请使用read工具读取文件路径。例如：read('F:/path/to/file.md')\n"

        # 添加历史任务结果作为参考（如果有）
        if history_task_results:
            history_info = []
            all_generated_files = []  # 收集所有历史任务生成的文件

            for i, task_result in enumerate(history_task_results, 1):
                # task_result可能是字符串或包含execution_result字段的字典
                result_files = ""
                if isinstance(task_result, dict):
                    result_content = task_result.get("summary_result", str(task_result))
                    # 提取生成的文件信息
                    generated_files = task_result.get("generated_files", [])
                    if generated_files:
                        all_generated_files.extend(generated_files)
                        files_info = []
                        for file_info in generated_files:
                            if isinstance(file_info, dict):
                                name = file_info.get("name", "未命名文件")
                                path = file_info.get("path", "")
                                files_info.append(f"  - {name}: {path}")
                            elif isinstance(file_info, str):
                                files_info.append(f"  - {file_info}")
                        if files_info:
                            result_files = f"\n生成的文件：\n" + "\n".join(files_info)
                else:
                    result_content = str(task_result)
                # 截断过长的结果
                if len(result_content) > 500:
                    result_content = result_content[:500] + "...[截断]"
                history_info.append(f"历史任务{i}结果：\n{result_content}")

            if history_info:
                history_context = "\n\n".join(history_info)
                system_prompt += f"\n历史任务执行结果（供参考）：\n{history_context}\n"

            # 添加所有历史生成文件的汇总信息
            if all_generated_files:
                unique_files = []
                seen_paths = set()
                for file_info in all_generated_files:
                    if isinstance(file_info, dict):
                        path = file_info.get("path", "")
                        name = file_info.get("name", "未命名文件")
                        if path and path not in seen_paths:
                            seen_paths.add(path)
                            unique_files.append((name, path))
                    elif isinstance(file_info, str) and file_info not in seen_paths:
                        seen_paths.add(file_info)
                        unique_files.append((file_info, file_info))

                if unique_files:
                    files_summary = []
                    for name, path in unique_files:
                        files_summary.append(f"- {name}: 路径 {path}")
                    files_context = "\n".join(files_summary)
                    system_prompt += f"\n历史任务生成的所有文件（可以使用read工具读取）：\n{files_context}\n"
                    system_prompt += "提示：如果需要参考历史任务生成的文件内容，请使用read工具读取上述文件路径。例如：read('F:/path/to/file.md')\n"

        # 添加用户原始请求作为参考
        if human_message_content:
            system_prompt += f"\n用户原始请求：\n{human_message_content}\n"

        # 创建agent，指定response_format为JSON Schema
        agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=self.tools,
            response_format=execute_result_schema
        )

        # 执行任务
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": "请执行上述任务并输出JSON格式结果。"}
            ]
        })

        # 解析执行结果
        needs_input = False
        question_text = ""
        generated_files = []

        try:
            data = result["structured_response"]
            generated_files = data.get("generated_files", [])
            summary = str(data.get("summary", ""))
            status = str(data.get("status", ""))
            needs_input = data.get("needs_input", False)
            question = str(data.get("question", ""))

            final_execution_result = summary if summary else data
            task["status"] = status

            return {
                "execution_summary_result": final_execution_result,
                "task": task,
                "needs_input": needs_input,
                "question": question,
                "user_answer": "",
                "generated_files": generated_files,
            }
        except Exception as e:
            return {
                "execution_summary_result": f"任务执行失败，失败原因：{e}",
                "task": task,
                "needs_input": needs_input,
                "question": question_text,
                "user_answer": "",
                "generated_files": generated_files,
            }

    def review(self, state):
        # 从状态中提取必要信息
        execution_result = state.get("execution_result", "")
        task = state.get("task", {})
        task_content = task.get("content", "")
        uploaded_files_metadata = state.get("uploaded_files_metadata", [])
        human_message = state.get("human_message")
        human_message_content = human_message.content if human_message else ""

        # 如果没有执行结果，直接返回
        if not execution_result:
            return {
                "execution_result": "",
                "needs_input": False,
                "question": "",
                "user_answer": "",
                "imaginary_data_dict": {}
            }

        # 读取上传文件内容
        file_contents = []
        for file_meta in uploaded_files_metadata:
            file_path = file_meta.get("path", "")
            if file_path:
                content = read_file_tool.invoke({"path": file_path})
                # 检查是否读取成功（排除错误信息）
                if not content.startswith("[read] 文件不存在") and not content.startswith("[read] 读取失败"):
                    file_contents.append(content)

        # 构建文件内容上下文
        files_context = "\n\n".join(file_contents) if file_contents else "没有上传文件内容可供参考。"

        # 使用LLM检测遐想数据并检查任务符合性
        system_prompt = """你是一个事实核查助手。请比较以下任务执行结果和源文件内容，找出执行结果中哪些信息是源文件中没有的（即模型可能捏造的遐想数据）。同时检查执行结果是否符合任务要求。

任务要求：{task_content}
用户原始请求：{human_message_content}
源文件内容：
{files_context}

任务执行结果：
{execution_result}

请分析并输出JSON格式，包含以下字段：
1. "task_compliance": 布尔值，表示执行结果是否符合任务要求。
2. "imaginary_data": 数组，每个元素是一个对象，包含：
   - "description": 对缺失数据的简短描述（例如"项目预算"、"时间线"、"技术规格"等）
   - "text": 原文片段（遐想数据的具体内容）
   - "reason": 为什么认为这是遐想数据（可选）
3. "needs_clarification": 布尔值，表示是否需要用户澄清或提供更多信息（例如，如果执行结果不完整、模糊，或者需要额外信息才能继续）。
4. "clarification_question": 字符串，如果需要澄清，提供一个清晰、具体的问题来询问用户缺失的信息。

请确保"description"是简洁的名词短语，将用于占位符[请给出XXX]中的XXX。
如果执行结果不完整、模糊，或者需要用户提供更多信息才能继续，请设置needs_clarification为true并提供clarification_question。"""

        user_prompt = "请分析上述内容并输出JSON。"

        try:
            # 调用LLM
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt.format(
                    task_content=task_content,
                    human_message_content=human_message_content,
                    files_context=files_context,
                    execution_result=execution_result
                )},
                {"role": "user", "content": user_prompt}
            ])

            # 提取响应内容
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, dict) and "content" in response:
                response_text = response["content"]
            else:
                response_text = str(response)

            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                # 确保所有必需字段都存在
                analysis.setdefault("task_compliance", True)
                analysis.setdefault("imaginary_data", [])
                analysis.setdefault("needs_clarification", False)
                analysis.setdefault("clarification_question", "")
                # 调试信息
                # print(f"调试: response_text={response_text[:200]}")
                # print(f"调试: analysis={analysis}")
            else:
                # print(f"调试: 未找到JSON匹配，response_text={response_text[:500]}")
                analysis = {"task_compliance": True, "imaginary_data": [], "needs_clarification": False,
                            "clarification_question": ""}

        except Exception as e:
            # 如果解析失败，默认认为没有遐想数据
            analysis = {"task_compliance": True, "imaginary_data": [], "needs_clarification": False,
                        "clarification_question": ""}

        # 初始化遐想数据字典
        imaginary_data_dict = {}
        reviewed_result = execution_result

        # 如果有遐想数据，进行处理
        if analysis.get("imaginary_data"):
            # 创建临时文件来存储执行结果，以便使用edit工具
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(execution_result)
                temp_file_path = f.name

            try:
                # 对每个遐想数据进行替换
                for item in analysis["imaginary_data"]:
                    description = item.get("description", "未知数据")
                    text = item.get("text", "")
                    if text and description:
                        # 使用edit工具替换
                        edit_result = edit_file_tool.invoke(
                            {"path": temp_file_path, "old": text, "new": f"[请给出{description}]", "count": -1})
                        # 保存到字典
                        imaginary_data_dict[description] = text

                # 读取修改后的内容
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    reviewed_result = f.read()

            except Exception as e:
                # 如果编辑过程出错，保持原结果
                reviewed_result = execution_result
            finally:
                # 删除临时文件
                try:
                    Path(temp_file_path).unlink(missing_ok=True)
                except:
                    pass

        # 决定是否需要用户输入
        needs_input = False
        question = ""
        user_answer = ""

        if imaginary_data_dict:
            # 构建问题：针对每个遐想数据询问用户
            for description, text in imaginary_data_dict.items():
                prompt = f"{description}数据缺失，模型根据您的历史或常识性信息使用{text}替代，您是否接受？"
                # 使用question工具生成问题
                question_result = question_tool.invoke(
                    {"prompt": prompt, "question_id": f"imaginary_data_{description}"})

                # 检查question_result是否包含[question]标记
                if "[question] 需要用户回答:" in question_result:
                    needs_input = True
                    # 提取问题内容
                    match = re.search(r'\[question\] 需要用户回答: (.+)', question_result)
                    if match:
                        question = match.group(1)
                    else:
                        question = prompt
                    # 只处理第一个遐想数据的问题
                    break
        elif analysis.get("needs_clarification", False):
            # 如果没有遐想数据但需要澄清，使用LLM生成的问题
            needs_input = True
            question = analysis.get("clarification_question", "")
            if not question:
                question = "任务执行需要更多信息，请提供缺失的细节。"

        # 返回审查结果
        return {
            "execution_result": reviewed_result,
            "needs_input": needs_input,
            "question": question,
            "user_answer": user_answer,
            "imaginary_data_dict": imaginary_data_dict,
            "task_compliance": analysis.get("task_compliance", True)
        }

    def is_need_review(self, state):
        # 从状态中获取任务信息和执行结果
        task = state.get("task", {})
        task_content = task.get("content", "")
        execution_result = state.get("execution_result", {})

        # 如果没有任务内容，默认不需要review
        if not task_content:
            return "END"

        # 定义判断是否需要的JSON Schema
        review_schema = {
            "type": "object",
            "description": "判断任务是否需要review",
            "properties": {
                "needs_review": {
                    "type": "boolean",
                    "description": "是否需要review，如果任务涉及生成文件内容（如生成文档、创建文件、编写报告、填充模板等），或者执行结果表明需要人类参与（例如，包含问题、需要澄清、信息不全等），则返回true；如果任务是分析、获取、提取、制定、识别、执行、检查、评估等分析总结类任务，且执行结果完整，则返回false"
                },
                "reason": {
                    "type": "string",
                    "description": "判断理由，简要说明为什么需要或不需要review"
                }
            },
            "required": ["needs_review", "reason"]
        }

        # 构建系统提示
        system_prompt = """你是一个任务审核判断助手。你的任务是根据任务描述和执行结果判断该任务是否需要进入review流程。

判断规则：
1. 需要review的任务：
   - 任务内容涉及生成、创建、编写、输出文件内容（如生成文档、创建文件、编写报告、填充模板等）
   - 执行结果表明需要人类参与（例如，包含问题、需要澄清、信息不全等）
2. 不需要review的任务：
   - 任务内容涉及分析、获取、提取、制定、识别、执行、检查、评估等分析总结类操作，且执行结果完整
   
注意：执行结果是否生成文件不参与判断是否需要review的决策

任务描述：
{task_content}

执行结果：
{execution_result}

请根据上述规则判断该任务是否需要review，严格按照JSON Schema输出结果。"""

        try:
            # result = self.llm.invoke([
            #     {"role": "system", "content": system_prompt.format(
            #         task_content=task_content,
            #         execution_result=execution_result if execution_result else "（无执行结果）"
            #     )},
            #     {"role": "user", "content": f"请判断上述任务是否需要review。并以{review_schema}规定的json格式输出"}
            # ])
            # 创建agent进行判断
            agent = create_agent(
                model=self.llm,
                system_prompt=system_prompt.format(task_content=task_content,
                                                   execution_result=execution_result if execution_result else "（无执行结果）"),
                response_format=review_schema
            )

            # 执行判断
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": "请判断上述任务是否需要review。"}
                ]
            })

            # 提取结构化响应
            if isinstance(result, dict):
                # 尝试从structured_response提取
                structured_response = result.get("structured_response", {})
                if structured_response and isinstance(structured_response, dict):
                    needs_review = structured_response.get("needs_review", False)
                else:
                    # 尝试从output字段提取
                    output = result.get("output", "")
                    if output and isinstance(output, str):
                        # 尝试解析JSON
                        import re
                        json_match = re.search(r'\{.*\}', output, re.DOTALL)
                        if json_match:
                            analysis = json.loads(json_match.group())
                            needs_review = analysis.get("needs_review", False)
                        else:
                            needs_review = False
                    else:
                        needs_review = False
            else:
                needs_review = False

            # 根据判断结果返回
            if needs_review:
                return "review_plan"
            else:
                return "END"

        except Exception as e:
            # 如果出错，使用保守策略：如果包含生成文件类关键词，返回review_plan
            review_keywords = ["生成", "创建", "编写", "输出", "填充", "撰写", "文档", "文件", "计划", "报告"]
            for keyword in review_keywords:
                if keyword in task_content:
                    return "review_plan"
            return "END"
