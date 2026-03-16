from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time

import psutil
import requests
import logging
from io import BytesIO

import pyautogui
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.tools import tool, BaseTool
from langchain_community.document_loaders import WebBaseLoader
from auto_test_assistant.manager.operation_checkpoint_manager import OperationCheckpointManager

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LoginTest")

# ==================== 检查点管理器 ====================
checkpoint_manager = OperationCheckpointManager()

# ==================== 配置区域 ====================
num_seconds = 1.2
# 阿里云视觉模型 API 配置（兼容 OpenAI 格式）
# 获取 API Key：https://help.aliyun.com/zh/model-studio/get-api-key
ALIYUN_API_KEY = "sk-dc529e6be2344352adcfc82f48712f3e"

# 根据地域选择对应的 base_url
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 视觉模型名称（注意：必须为视觉模型，如 qwen-vl-plus / qwen3-vl-plus）
ALIYUN_MODEL = "qwen3.5-plus"  # 如果此模型不支持图像，请修改为 qwen3-vl-plus 或 qwen-vl-plus
# 截图保存路径
SCREENSHOT_DIR = "./screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def _append_operation_log(log_file_path: str, code: str) -> None:
    """追加操作代码到日志文件"""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + code + "\n")

    logger.info(f"已记录操作到: {log_file_path}")


def _ensure_str_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def encode_image_to_base64(image):
    """将 PIL Image 转换为 base64 字符串。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def call_aliyun_vision(sys_prompt, user_prompt, image) -> dict:
    """
    调用阿里云视觉模型（兼容 OpenAI 格式）。
    返回模型的文本回答。
    """
    logger.info("=" * 50)
    logger.info("调用阿里云视觉模型")
    logger.info(f"模型: {ALIYUN_MODEL}")
    logger.info(f"提示词: {user_prompt}")
    logger.info(f"截图尺寸: {image.size}")

    base64_image = encode_image_to_base64(image)

    headers = {
        "Authorization": f"Bearer {ALIYUN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": ALIYUN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sys_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    # 出于安全考虑，不打印完整 payload（包含 base64 超长内容），但可以打印简略信息
    logger.debug(f"请求URL: {ALIYUN_BASE_URL}")
    logger.debug(f"请求模型: {payload['model']}")

    start_time = time.time()
    try:
        response = requests.post(ALIYUN_BASE_URL, headers=headers, json=payload)
        elapsed = time.time() - start_time
        logger.info(f"API 响应耗时: {elapsed:.2f} 秒")
        logger.info(f"HTTP 状态码: {response.status_code}")

        response.raise_for_status()
        result = response.json()
        logger.debug(f"原始响应: {result}")

        # 兼容 OpenAI 格式的返回结构
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            content = content.strip()
            logger.info(f"模型回答: {content}")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"提取 JSON: {json_str}")
                    return json.loads(json_str)
                else:
                    logger.error(f"无法解析 JSON: {content}")
                    raise
        else:
            logger.error(f"API 返回结构异常: {result}")
            return {}
    except requests.exceptions.Timeout:
        logger.error("API 请求超时")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"响应状态码: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
        return {}
    except Exception as e:
        logger.error(f"调用阿里云视觉模型时发生未知错误: {e}", exc_info=True)
        return {}


def validate_steps_with_vision(steps: List[str], current_screenshot_path: str,
                               previous_screenshot_path: Optional[str] = None) -> tuple[bool, str]:
    """
    使用视觉模型验证步骤是否已正确执行
    
    Args:
        steps: 需要验证的步骤列表
        current_screenshot_path: 当前截图文件路径
        previous_screenshot_path: 上一个检查点的截图文件路径（可选）
        
    Returns:
        (验证是否成功, 消息)
    """
    from PIL import Image

    if not os.path.exists(current_screenshot_path):
        return False, f"当前截图文件不存在: {current_screenshot_path}"

    current_image = Image.open(current_screenshot_path)
    previous_image = None
    if previous_screenshot_path and os.path.exists(previous_screenshot_path):
        previous_image = Image.open(previous_screenshot_path)

    output_schema = {
        "type": "object",
        "description": "测试步骤执行结果验证",
        "properties": {
            "validation_result": {
                "type": "str",
                "description": "如果所有步骤都正确执行，返回 success ，否则返回 failure "
            },
            "confidence": {
                "type": "float",
                "description": "0.0 到 1.0 之间的置信度分数"
            },
            "message": {
                "type": "string",
                "description": "详细的验证说明"
            }
        }
    }

    # 构建系统提示词
    sys_prompt = f"""Role
你是一个专业的 UI 自动化测试验证助手。你的任务是分析两个屏幕截图（前后状态）和一系列操作步骤，判断这些步骤是否已正确执行。

Task
1. 分析提供的操作步骤列表
2. 对比前后两个截图（如果提供了上一个截图）
3. 判断每个步骤是否已在界面上正确执行
4. 考虑UI状态变化、元素可见性、文本内容等
5. 给出整体验证结论

Output Schema
请严格遵守以下 JSON 结构： 
{output_schema}

Constraints
- 如果所有步骤都正确执行，返回 "validation_result": "success"
- 如果有任何步骤未正确执行或状态不符合预期，返回 "validation_result": "failure"
- 置信度表示你对判断的把握程度
- 消息应简洁明了地说明验证结果

User Input
操作步骤列表：
"""

    steps_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(steps)])
    user_prompt = f"""请验证以下操作步骤是否已正确执行：

操作步骤：
{steps_text}

{"请注意：提供了前后两个截图进行对比。" if previous_image else "请注意：只提供了当前截图，无法进行前后对比。"}
请仔细分析截图内容，判断每个步骤是否已正确完成。"""

    # 调用视觉模型
    answer = call_aliyun_vision(sys_prompt, user_prompt, current_image)

    if not answer:
        return False, "模型验证调用失败"

    validation_result = answer.get("validation_result", "failure")
    confidence = answer.get("confidence", 0.0)
    message = answer.get("message", "无详细消息")

    success = validation_result.lower() == "success"
    return success, f"{message} (置信度: {confidence})"


def kill_process_by_name(process_name):
    """通过进程名终止进程"""
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            try:
                proc.terminate()
                print(f"进程 {process_name} (PID: {proc.info['pid']}) 已终止。")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"终止失败: {e}")


@tool("invoke_model_tool", return_direct=False, description="截图并调用视觉模型获取控件坐标")
def invoke_model_tool(step_id: int, use_case_id: Optional[str] = None, operation_log_path: Optional[str] = None) -> str:
    """截图并调用视觉模型获取控件在屏幕上的位置坐标。
    参数：
        step_id: 当前步骤的id
        use_case_id: 可选，测试用例ID，用于检查点管理
        operation_log_path: 操作日志文件路径
    """
    step_list = []
    image = pyautogui.screenshot()
    timestamp = int(time.time() * 1000)
    filename = f"full_screen_{timestamp}.png"
    if filename:
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        image.save(filepath)
        logger.info(f"截图已保存: {filepath}")
    else:
        logger.debug("截图未保存文件")
    # 检查点验证逻辑
    if use_case_id is not None:
        logger.info(f"操作日志文件路径：{operation_log_path}")
        # 设置当前测试用例
        checkpoint_manager.set_current_test_case(use_case_id, operation_log_path)
        # 如果没有提供步骤，使用空列表
        output_dir = Path("json")
        output_file = output_dir / "ui_use_cases.json"
        with open(output_file, "r", encoding="utf-8") as f:
            ui_use_cases = json.load(f)
        use_case_dict = {uc['use_case_id']: uc for uc in ui_use_cases}

        # 根据 ID 获取用例
        target_use_case = use_case_dict.get(use_case_id)
        if target_use_case:
            step_list = target_use_case.get("use_case_steps", [])

        prev_checkpoint = checkpoint_manager.get_current_checkpoint()

        # 获取截图文件路径（如果保存了）
        screenshot_path = None
        if filename:
            screenshot_path = os.path.join(SCREENSHOT_DIR, filename)

        if prev_checkpoint:
            # 存在上一个checkpoint，表示当前步骤不是该用例第一次调用视觉模型，需要对上一次调用视觉模型的结果进行review
            target_index = next(i for i, step in enumerate(step_list) if step['id'] == step_id)
            result = step_list[:target_index]
            # 调用验证并创建检查点（暂时假设验证成功）
            validation_success = checkpoint_manager.validate_and_create_checkpoint(
                operation_log_path=operation_log_path,
                steps=result,
                current_screenshot_path=screenshot_path,
                model_validator=validate_steps_with_vision,
                step_id=step_id
            )
            if not validation_success:
                # 验证失败，回退到上一个检查点
                if prev_checkpoint:
                    rollback_info = checkpoint_manager.rollback_to_checkpoint(prev_checkpoint.checkpoint_id,
                                                                              operation_log_path)
                    logger.warning(f"步骤验证失败，已回滚到检查点: {rollback_info['checkpoint_id']}\n"
                                   f"回退的步骤id为：{rollback_info['first_step_to_redo']}")

                    # 回退回去之后初始化全部操作然后redo我们操作日志啊
                    kill_process_by_name("clouddesktop-qml.exe")
                    # 启动新进程
                    bin_dir = r'D:\Software\CtyunCloud\CtyunClouddeskPublic\bin'
                    exe_name = 'clouddesktop-qml.exe'
                    exe_path = os.path.join(bin_dir, exe_name)

                    if os.path.isfile(exe_path):
                        # 使用绝对路径 + 工作目录启动
                        subprocess.Popen([exe_path], cwd=bin_dir)
                        time.sleep(2)
                        logger.info("已经重启天翼云应用")

                        # 执行操作日志脚本，恢复现场
                        subprocess.run(['python', operation_log_path])
                    else:
                        logger.error(f"错误：{exe_path} 不存在")

                    return (f"步骤验证失败，已回滚到检查点: {rollback_info['checkpoint_id']}\n"
                            f"回退的步骤id为：{rollback_info['first_step_to_redo']}")
                else:
                    logger.warning("步骤验证失败，但没有上一个检查点可回滚")
                    return "步骤验证失败，但没有上一个检查点可回滚"
        else:
            # 不存在上一个检查点，直接创建checkpoint
            target_index = next(i for i, step in enumerate(step_list) if step['id'] == step_id)
            result = step_list[:target_index]
            checkpoint_manager.create_checkpoint(screenshot_path=screenshot_path, metadata={"step_id": step_id},
                                                 steps=result, operation_log_path=operation_log_path)

    screen_x, screen_y = pyautogui.size()

    sys_prompt = """Role
你是一个专业的 UI 自动化视觉定位助手。你的任务是分析提供的截图，根据用户的文本描述，精准定位界面中的目标控件，并输出其相对位置比例。

Task
识别图像中符合用户描述的 UI 控件。
确定该控件的几何中心点。
计算该中心点距离图像左上角原点 (0,0) 的水平距离占图像总宽度的百分比 (x_percent)，以及垂直距离占图像总高度的百分比 (y_percent)。
以严格的 JSON 格式输出结果，不要包含任何 Markdown 标记（如 ```json）、解释性文字或额外内容。
Output Schema
请严格遵守以下 JSON 结构： {"target_description": "用户输入的控件描述", "status": "found 或 not_found", "location_percentage": {"x": 浮点数，0 到 100 之间，保留两位小数，表示水平方向百分比, "y": 浮点数，0 到 100 之间，保留两位小数，表示垂直方向百分比 }, "confidence": 0.0 到 1.0 之间的置信度分数 }

Constraints
坐标原点定义为图像的左上角 (0,0)，右下角为 (100,100)。
如果找不到对应的控件，"status" 设为 "not_found"，"location_percentage" 中的 x 和 y 均设为 0，"confidence" 设为 0.0。
输出必须是纯文本 JSON，严禁使用 Markdown 代码块包裹。
确保 JSON 语法正确，可直接被代码解析。
不需要输出具体的像素值或屏幕分辨率，仅需比例。
User Input
控件描述"""
    # 获取当前步骤描述
    step_dict = {step['id']: step['value'] for step in step_list}
    step_desc = step_dict.get(step_id)

    if step_desc is None:
        logger.error("无法获取步骤描述")
        return "无法获取步骤描述"
    user_prompt = f"请根据描述找出你需要的控件的中心坐标百分比：{step_desc}。"

    answer = call_aliyun_vision(sys_prompt, user_prompt, image)

    if not answer:
        logger.warning("模型未返回有效回答，无法解析坐标")
        return "模型未返回有效回答，无法解析坐标"

    # 解析坐标
    if answer.get("status", "not_found") == "not_found":
        logger.warning("模型未找到控件位置")
        return "模型未找到控件位置"
    else:
        x_percent = answer["location_percentage"]["x"]
        y_percent = answer["location_percentage"]["y"]
        x = int(x_percent / 100 * screen_x)
        y = int(y_percent / 100 * screen_y)
        logger.info(f"该控件坐标为：[x:{x},y:{y}]")
        return f"该控件坐标为：[x:{x},y:{y}]"


@tool("click_tool", return_direct=False, description="使用键鼠操作中 鼠标单击 操作")
def click_tool(x: int, y: int, operation_log_path: Optional[str] = None) -> str:
    """使用键鼠操作中 鼠标单击 操作。参数：x=点击坐标的x值，y=点击坐标的y值，operation_log_path=可选的操作日志文件路径"""
    try:
        pyautogui.click(x, y, duration=num_seconds)
        if operation_log_path:
            log_code = f"pyautogui.click({x}, {y}, duration={num_seconds})"
            _append_operation_log(operation_log_path, log_code)
        return "点击操作完成"
    except Exception as e:
        return f"点击操作失败：{e}"


@tool("moveTo_tool", return_direct=False, description="使用键鼠操作中 鼠标移动 操作")
def moveTo_tool(x: int, y: int, operation_log_path: Optional[str] = None) -> str:
    """使用键鼠操作中 鼠标移动 操作。参数：x=移动到坐标的x值，y=移动到坐标的y值，operation_log_path=可选的操作日志文件路径"""
    try:
        pyautogui.moveTo(x, y, duration=num_seconds)
        if operation_log_path:
            log_code = f"pyautogui.moveTo({x}, {y}, duration={num_seconds})"
            _append_operation_log(operation_log_path, log_code)
        return "移动操作完成"
    except Exception as e:
        return f"移动操作失败：{e}"


@tool("dragTo_tool", return_direct=False, description="使用键鼠操作中 鼠标拖拽 操作")
def dragTo_tool(x: int, y: int, operation_log_path: Optional[str] = None) -> str:
    """使用键鼠操作中 鼠标拖拽 操作。参数：x=拖拽到坐标的x值，y=拖拽到坐标的y值，operation_log_path=可选的操作日志文件路径"""
    try:
        pyautogui.dragTo(x, y, duration=num_seconds, button='left')
        if operation_log_path:
            log_code = f"pyautogui.dragTo({x}, {y}, duration={num_seconds}, button='left')"
            _append_operation_log(operation_log_path, log_code)
        return "拖拽操作完成"
    except Exception as e:
        return f"拖拽操作失败：{e}"


@tool("typewrite", return_direct=False, description="使用键鼠操作中 鼠标拖拽 操作")
def typewrite(input_str: str, operation_log_path: Optional[str] = None) -> str:
    """使用键鼠操作中 键盘输入 操作。参数：input_str=输入的字符串，operation_log_path=可选的操作日志文件路径"""
    try:
        pyautogui.typewrite(input_str, interval=0.25)
        if operation_log_path:
            log_code = f"pyautogui.typewrite('{input_str}', interval=0.25)"
            _append_operation_log(operation_log_path, log_code)
        return "键盘输入操作完成"
    except Exception as e:
        return f"键盘输入操作失败：{e}"


@tool("read", return_direct=False, description="读取文件内容")
def read_file_tool(path: str) -> str:
    """读取指定文件的全部内容。参数: path=文件路径(相对或绝对)。"""
    p = _ensure_str_path(path)
    if not p.is_file():
        return f"[read] 文件不存在: {p}"
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[read] 读取失败: {e}"


@tool("write", return_direct=False, description="写入/创建文件")
def write_file_tool(path: str, content: str) -> str:
    """写入/创建文件，覆盖原内容。参数: path=文件路径, content=写入内容。"""
    p = _ensure_str_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[write] 已写入文件: {p}"
    except Exception as e:
        return f"[write] 写入失败: {e}"


@tool("edit", return_direct=False, description="精确编辑文件内容（替换字符串）")
def edit_file_tool(path: str, old: str, new: str, count: int = -1) -> str:
    """
    在文件中进行字符串替换。
    参数:
    - path: 文件路径
    - old: 需要被替换的原字符串
    - new: 新字符串
    - count: 替换次数，默认 -1 表示全部替换
    """
    p = _ensure_str_path(path)
    if not p.is_file():
        return f"[edit] 文件不存在: {p}"
    try:
        text = p.read_text(encoding="utf-8")
        new_text = text.replace(old, new, count if count >= 0 else text.count(old))
        p.write_text(new_text, encoding="utf-8")
        return f"[edit] 已替换 {path} 中的内容"
    except Exception as e:
        return f"[edit] 编辑失败: {e}"


@tool("glob", return_direct=False, description="文件模式匹配（*.py, **/*.js等）")
def glob_tool(pattern: str, root: str = ".") -> List[str]:
    """按模式匹配文件，例如: pattern='**/*.py', root='.'。返回匹配到的相对路径列表。"""
    base = _ensure_str_path(root)
    matches = [str(p.relative_to(base)) for p in base.rglob(pattern)]
    return matches


@tool("grep", return_direct=False, description="文件内容正则搜索")
def grep_tool(pattern: str, root: str = ".", ignore_case: bool = True) -> List[str]:
    """
    在目录下递归搜索包含正则 pattern 的文件行。
    返回格式: 'relative/path:行号:内容'
    """
    base = _ensure_str_path(root)
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)
    results: List[str] = []

    for file in base.rglob("*"):
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = str(file.relative_to(base))
                results.append(f"{rel}:{i}:{line}")
    return results


@tool("bash", return_direct=False, description="执行shell命令（git、npm、docker等）")
def bash_tool(command: str, cwd: Optional[str] = None, timeout: int = 600) -> str:
    """执行 shell 命令（Windows 下使用 PowerShell / *nix 使用 /bin/bash），返回 stdout/stderr。"""
    workdir = _ensure_str_path(cwd) if cwd else None
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir) if workdir else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return f"[bash] exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "[bash] 命令执行超时"
    except Exception as e:
        return f"[bash] 执行失败: {e}"


@dataclass
class TodoItem:
    id: str
    content: str
    status: str = "pending"


_TODO_STORE: dict[str, TodoItem] = {}

item_schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "content", "status"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "任务唯一标识符"
                    },
                    "content": {
                        "type": "string",
                        "description": "任务描述（应遵循SMART原则：具体、可衡量、可实现、相关、有时限）"
                    },
                    "status": {
                        "type": "string",
                        "description": "任务状态",
                        "enum": ["pending", "in-progress", "completed", "failed", "cancelled"],
                        "default": "pending"
                    },
                    "priority": {
                        "type": "string",
                        "description": "任务优先级",
                        "enum": ["must-have", "should-have", "could-have", "won't-have", "high", "medium", "low"],
                        "default": "low"
                    },
                    "depends_on": {
                        "type": "array",
                        "description": "依赖的任务ID列表",
                        "items": {
                            "type": "string"
                        },
                        "default": []
                    },
                    "completion_criteria": {
                        "type": "string",
                        "description": "完成标准（明确定义任务完成的判断条件）",
                        "default": ""
                    },
                    "estimated_tools": {
                        "type": "array",
                        "description": "预估需要的工具列表",
                        "items": {
                            "type": "string",
                            "enum": ["read", "write", "edit", "glob", "grep", "bash", "todowrite", "task", "skill",
                                     "question", "webfetch"]
                        },
                        "default": []
                    }
                }
            }
        }
    },
    "required": ["items"]
}


@tool("todowrite", return_direct=False, description="创建和管理结构化任务列表，支持思维链工作流程",
      args_schema=item_schema)
def todowrite_tool(items: List[dict]) -> str:
    """
    创建或更新结构化任务列表。

    参数: items=[{id, content, status, priority?, depends_on?, completion_criteria?}]

    必需字段：
    - id: 任务唯一标识符（字符串）
    - content: 任务描述（应遵循SMART原则：具体、可衡量、可实现、相关、有时限）
    - status: 任务状态，可选值：pending, in-progress, completed, failed, cancelled

    可选字段（可在content中描述，或作为独立字段）：
    - priority: 优先级（must-have/should-have/could-have/won't-have 或 high/medium/low）
    - depends_on: 依赖的任务ID列表（字符串数组）
    - completion_criteria: 完成标准（明确定义任务完成的判断条件）
    - estimated_tools: 预估需要的工具列表（如 ["read", "write", "bash"]）

    使用场景：
    1. 战略规划阶段：创建完整的TodoList，包含所有任务及其依赖关系
    2. 执行阶段：更新任务状态（pending -> in-progress -> completed）
    3. 反思阶段：调整任务列表，添加新任务或修改现有任务

    示例：
    [
        {
            "id": "1",
            "content": "分析需求文档，提取核心功能模块（必须完成）",
            "status": "pending",
            "priority": "must-have"
        },
        {
            "id": "2",
            "content": "创建测试计划文档结构（依赖任务1）",
            "status": "pending",
            "depends_on": ["1"],
            "priority": "must-have"
        }
    ]
    """
    for raw in items:
        # 提取所有字段，包括可选字段
        item_data = {
            "id": str(raw.get("id")),
            "content": str(raw.get("content")),
            "status": str(raw.get("status", "pending")),
        }
        # 保留其他可选字段到 metadata
        metadata = {}
        for key in ["priority", "depends_on", "completion_criteria", "estimated_tools"]:
            if key in raw:
                metadata[key] = raw[key]

        item = TodoItem(
            id=item_data["id"],
            content=item_data["content"],
            status=item_data["status"],
        )
        # 将元数据存储到 item 的 __dict__ 中（如果 TodoItem 支持）
        if metadata:
            item.__dict__.update(metadata)

        _TODO_STORE[item.id] = item

    # 返回当前所有任务，包括元数据
    result_dict = {}
    for k, v in _TODO_STORE.items():
        result_dict[k] = vars(v)

    return "[todowrite] 当前任务列表: " + json.dumps(
        result_dict, ensure_ascii=False, indent=2
    )


@tool("task", return_direct=False, description="启动专用代理进行复杂探索（代码库分析、多步骤任务）")
def task_tool(description: str) -> str:
    """
    启动专用代理进行复杂探索的占位符工具。
    目前仅记录任务描述，真实实现可在此基础上扩展。
    """
    return f"[task] 已记录复杂任务描述，稍后由专用代理处理: {description}"


# @tool("skill", return_direct=False)
# def skill_tool(name: str, action: str = "describe") -> str:
#     """
#     skill 工具占位符。
#     目前支持: action='describe' 时，返回该技能的用途说明（由上层注入）。
#     实际的技能路由由主 agent / 子 agent 控制。
#     """
#     return f"[skill] 请求技能: {name}, action={action}（实际执行由多agent系统负责）"


# 全局存储用户问答（用于测试和预配置答案）
_QUESTION_ANSWER_STORE: Dict[str, str] = {}
# 预配置答案文件路径（可选）
_QUESTION_ANSWER_FILE = Path("question_answers.json")


@tool("question", return_direct=False, description="交互式提问（收集需求、澄清指令）")
def question_tool(prompt: str, question_id: Optional[str] = None) -> str:
    """
    交互式提问工具，支持Human-in-the-Loop机制。
    
    参数:
    - prompt: 向用户提问的具体问题
    - question_id: 可选的问题标识符，用于从预配置答案中查找回答
    
    工作流程:
    1. 检查是否有预配置答案（通过question_id或prompt匹配）
    2. 如果有预配置答案，直接返回答案
    3. 如果没有预配置答案，返回结构化的问题信息供上层处理
    4. 记录问题到日志文件便于调试
    
    预配置答案支持:
    - 全局字典 _QUESTION_ANSWER_STORE
    - 外部文件 question_answers.json
    - 环境变量（未来扩展）
    
    返回格式:
    - 如果有答案: [question] 用户已回答: {答案内容}
    - 如果没有答案: [question] 需要用户回答: {问题内容} [ID: {question_id}]
    """
    import os
    from datetime import datetime

    # 生成问题ID（如果未提供）
    if question_id is None:
        # 基于prompt生成简单的哈希ID
        import hashlib
        question_id = hashlib.md5(prompt.encode()).hexdigest()[:8]

    # 尝试从全局存储获取答案
    if question_id in _QUESTION_ANSWER_STORE:
        answer = _QUESTION_ANSWER_STORE[question_id]
        return f"[question] 用户已回答（来自内存存储）: {answer}"

    # 尝试从预配置答案文件获取答案
    if _QUESTION_ANSWER_FILE.is_file():
        try:
            answers_data = json.loads(_QUESTION_ANSWER_FILE.read_text(encoding="utf-8"))
            if isinstance(answers_data, dict):
                # 尝试通过question_id查找
                if question_id in answers_data:
                    answer = answers_data[question_id]
                    return f"[question] 用户已回答（来自文件存储）: {answer}"
                # 尝试通过prompt关键词查找
                for key, value in answers_data.items():
                    if isinstance(key, str) and key in prompt:
                        return f"[question] 用户已回答（关键词匹配）: {value}"
        except Exception as e:
            # 文件读取失败，继续使用标准流程
            pass

    # 检查环境变量中是否有预配置答案（格式：QUESTION_ANSWER_{QUESTION_ID}）
    env_key = f"QUESTION_ANSWER_{question_id.upper()}"
    if env_key in os.environ:
        answer = os.environ[env_key]
        return f"[question] 用户已回答（来自环境变量）: {answer}"

    # 没有预配置答案，返回结构化问题信息
    # 记录问题到日志文件（便于调试和跟踪）
    log_dir = Path("question_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"questions_{datetime.now().strftime('%Y%m%d')}.log"

    question_record = {
        "timestamp": datetime.now().isoformat(),
        "question_id": question_id,
        "prompt": prompt,
        "status": "pending"
    }

    try:
        # 追加记录到日志文件
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(question_record, ensure_ascii=False) + "\n")
    except Exception:
        # 日志写入失败不影响主要功能
        pass

    # 返回结构化的问题信息
    # 注意：在纯后端模式下，上层需要解析此输出并获取用户回答
    # 然后通过某种机制（如更新预配置答案）提供回答
    structured_output = {
        "type": "question",
        "question_id": question_id,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "instructions": "请提供此问题的答案，可通过以下方式：1. 更新question_answers.json文件 2. 设置环境变量 3. 通过其他交互机制"
    }

    return f"[question] 需要用户回答: {prompt}\n" + \
        f"[question] 问题ID: {question_id}\n" + \
        f"[question] 结构化信息: {json.dumps(structured_output, ensure_ascii=False)}"


def set_question_answer(question_id: str, answer: str) -> None:
    """
    设置预配置答案到内存存储。
    用于测试和开发阶段提供模拟用户回答。
    """
    global _QUESTION_ANSWER_STORE
    _QUESTION_ANSWER_STORE[question_id] = answer


def load_question_answers_from_file(file_path: Optional[str] = None) -> bool:
    """
    从JSON文件加载预配置答案。
    
    参数:
        file_path: JSON文件路径，默认为 question_answers.json
        
    返回:
        bool: 是否成功加载
    """
    global _QUESTION_ANSWER_STORE, _QUESTION_ANSWER_FILE

    if file_path:
        target_file = Path(file_path)
    else:
        target_file = _QUESTION_ANSWER_FILE

    if not target_file.is_file():
        return False

    try:
        answers_data = json.loads(target_file.read_text(encoding="utf-8"))
        if isinstance(answers_data, dict):
            _QUESTION_ANSWER_STORE.update(answers_data)
            return True
    except Exception as e:
        print(f"[question] 加载预配置答案失败: {e}")

    return False


def save_question_answers_to_file(file_path: Optional[str] = None) -> bool:
    """
    将当前内存中的预配置答案保存到JSON文件。
    
    参数:
        file_path: JSON文件路径，默认为 question_answers.json
        
    返回:
        bool: 是否成功保存
    """
    global _QUESTION_ANSWER_STORE, _QUESTION_ANSWER_FILE

    if file_path:
        target_file = Path(file_path)
    else:
        target_file = _QUESTION_ANSWER_FILE

    try:
        # 确保目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(json.dumps(_QUESTION_ANSWER_STORE, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        print(f"[question] 保存预配置答案失败: {e}")

    return False


# 初始化时尝试加载预配置答案文件
try:
    load_question_answers_from_file()
except Exception:
    # 初始化失败不影响主要功能
    pass


class WebURLInput(BaseModel):
    """链接解析输入"""
    urls: list[str] = Field(description="用户问题中携带的所有url")


@tool("webfetch", return_direct=False, args_schema=WebURLInput,
      description="这是一个解析链接的工具助手，可以获取用户发送的链接并读取链接内所有内容，返回的为链接中内容")
def webfetch_tool(urls: list[str]) -> str:
    """获取链接中的内容"""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    web_content = docs[0].page_content
    return web_content


def list_all_mcp_tools() -> List[BaseTool]:
    """返回需要注册到所有 agent 的 MCP 风格工具列表。"""
    return [
        invoke_model_tool,
        read_file_tool,
        write_file_tool,
        edit_file_tool,
        glob_tool,
        grep_tool,
        bash_tool,
        todowrite_tool,
        task_tool,
        # skill_tool,
        question_tool,
        webfetch_tool,
        click_tool,
        moveTo_tool,
        dragTo_tool,
        typewrite
    ]
