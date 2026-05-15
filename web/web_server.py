import os
import sys
import yaml
import signal
import asyncio
import threading
import subprocess
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="RoboMaster Training WebUI Backend")

# ==========================================
# 0. 动态解析项目根目录
# ==========================================
ROOT_DIR = Path(__file__).parent.parent.absolute()

# ==========================================
# 1. 跨域与静态资源配置
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(ROOT_DIR / "model_res", exist_ok=True)
os.makedirs(ROOT_DIR / "data", exist_ok=True)

app.mount("/model_res", StaticFiles(directory=str(ROOT_DIR / "model_res")), name="model_res")
app.mount("/data", StaticFiles(directory=str(ROOT_DIR / "data")), name="data")

@app.get("/")
async def serve_index():
    index_path = ROOT_DIR / "web" / "index.html"
    if not index_path.exists():
        return {"error": "找不到 index.html，请检查文件路径"}
    return FileResponse(index_path)

# ==========================================
# 2. Config.yaml 配置读写接口
# ==========================================
CONFIG_PATH = ROOT_DIR / "config.yaml"

class ConfigUpdate(BaseModel):
    config: Dict[str, Any]

@app.get("/api/config")
async def get_config():
    if not CONFIG_PATH.exists():
        return {"error": f"未找到配置文件: {CONFIG_PATH}"}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {"status": "success", "data": data}

@app.post("/api/config")
async def update_config(payload: ConfigUpdate):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload.config, f, default_flow_style=False, allow_unicode=True)
        return {"status": "success", "message": "配置已保存"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 3. 终极版进程管家 (支持双向交互)
# ==========================================
class ProcessManager:
    def __init__(self):
        self.active_process: subprocess.Popen = None
        self.module_name: str = ""

    async def run_script(self, module_name: str, websocket: WebSocket):
        if self.active_process is not None and self.active_process.poll() is None:
            await websocket.send_text("\n\r[系统提示] 当前已有任务正在运行，请先停止！\n\r")
            return

        self.module_name = module_name
        
        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = "120"
        env["PYTHONIOENCODING"] = "utf-8" 
        env["PYTHONUNBUFFERED"] = "1" 
        
        # 【核心修复】：强行欺骗 rich 库开启交互模式，恢复光标的原地刷新能力！
        env["FORCE_INTERACTIVE"] = "1"
        env["FORCE_TERMINAL"] = "1"

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

        # 【核心修改 1】：增加 stdin=subprocess.PIPE，打通输入管道
        self.active_process = subprocess.Popen(
            [sys.executable, "-m", module_name],
            stdin=subprocess.PIPE,  
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            env=env,
            cwd=str(ROOT_DIR), 
            creationflags=creationflags
        )

        await websocket.send_text(f"\n\r[系统提示] 🚀 正在启动独立进程: {module_name}\n\r")

        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def reader_thread():
            try:
                while True:
                    chunk = self.active_process.stdout.read1(1024)
                    if not chunk:
                        break
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception:
                pass
            finally:
                loop.call_soon_threadsafe(q.put_nowait, b"")

        threading.Thread(target=reader_thread, daemon=True).start()

        try:
            while True:
                chunk = await q.get()
                if not chunk: 
                    break
                await websocket.send_text(chunk.decode('utf-8', errors='replace'))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WebSocket 发送异常: {e}")

        def wait_process():
            self.active_process.wait()
            return self.active_process.returncode

        code = await asyncio.to_thread(wait_process)
        
        try:
            color = "\x1b[32m" if code == 0 else "\x1b[31m" 
            await websocket.send_text(f"\n\r{color}[系统提示] 进程已结束 (退出码: {code})\x1b[0m\n\r")
        except:
            pass
            
        self.active_process = None

    # 【新增方法】：用于将前端的按键写入到 Python 进程中
    def write_input(self, data: str):
        if self.active_process and self.active_process.poll() is None and self.active_process.stdin:
            try:
                # 【修正】：将前端传来的 \r 统一替换为 Python 期望的 \n
                data = data.replace('\r', '\n')
                self.active_process.stdin.write(data.encode('utf-8'))
                self.active_process.stdin.flush()
            except Exception as e:
                print(f"写入输入流异常: {e}")

    def trigger_interrupt(self):
        if self.active_process is None or self.active_process.poll() is not None:
            return False, "没有正在运行的任务"

        try:
            if sys.platform == "win32":
                os.kill(self.active_process.pid, signal.CTRL_C_EVENT)
            else:
                self.active_process.send_signal(signal.SIGINT)
            return True, "已发送 Ctrl+C 中断信号，等待进程安全保存并退出..."
        except Exception as e:
            self.active_process.kill()
            return False, f"优雅中断失败，已强制击杀进程: {str(e)}"

process_manager = ProcessManager()

@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    await websocket.accept()
    process_task = None  # 用于保存后台正在运行的进程任务
    try:
        while True:
            # 这里的 receive_text 现在永远不会被阻塞了
            data = await websocket.receive_text()
            
            if data.startswith("start:"):
                module_name = data.split("start:")[1]
                # 【核心修复】：使用 create_task 将耗时的脚本运行放入后台并发执行！
                # 这样主循环就能瞬间回到 receive_text 状态，继续监听你的键盘和中断信号
                process_task = asyncio.create_task(process_manager.run_script(module_name, websocket))
                
            elif data == "stop":
                success, msg = process_manager.trigger_interrupt()
                await websocket.send_text(f"\n\r\x1b[33m[系统提示] {msg}\x1b[0m\n\r")
                
            elif data.startswith("input:"):
                input_str = data[len("input:"):]
                process_manager.write_input(input_str)
                
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=True)