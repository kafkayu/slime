import re
import json
import asyncio
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])

class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result


# Global tool registry instance
tool_registry = ToolRegistry()

def postprocess_responses(resp: str) -> str:
    """确保模型输出格式正确，只保留最后一个重要块"""
    if "<tool_call>" in resp:
        matches = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    if "```python" in resp:
        matches = list(re.finditer(r"```python\s*.*?```", resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    if "Answer:" in resp and "\\boxed{" in resp:
        matches = list(re.finditer(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp



async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """
    执行工具或代码。
    返回:
        next_obs: 工具执行结果文本
        done: 是否已经得到最终 Answer
    """
    action = None
    content = ""

    # 检查 <tool_call>
    tool_call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)
    if tool_call_match:
        try:
            tool_data = json.loads(tool_call_match.group(1))
            tool_name = tool_data.get("name")
            arguments = tool_data.get("arguments", {})
            if tool_name:
                async with SEMAPHORE:
                    result = await tool_registry.execute_tool(tool_name, arguments)
                next_obs = f"\n<interpreter>\n{result}\n</interpreter>\n"
                return next_obs, False
        except Exception:
            pass

    # 检查 Answer
    answer_match = re.search(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", prediction, re.DOTALL)
    if answer_match:
        return "", True

    return "", False

class DummyEnv:
    """示例环境，用于多轮生成和 Answer 判断"""

    def __init__(self):
        self.done = False
        self.steps = 0

    def reset(self):
        self.done = False
        self.steps = 0
        return {"obs_str": "初始观察"}, {"reset_info": True}

    def step(self, action_text: str):
        """
        根据模型输出判断是否完成
        """
        self.steps += 1
        if "Answer:" in action_text:
            self.done = True
        obs_str = f"环境返回第{self.steps}步的观察"
        step_info = {"step": self.steps}
        return {"obs_str": obs_str}, self.done, step_info

    def close(self):
        pass
