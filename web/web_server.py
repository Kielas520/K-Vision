import uvicorn
import subprocess
import sys
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# 模拟原 main.py 的菜单选项
MENU_OPTIONS = {
    "1": {"desc": "数据预处理", "module": "src.data_process.process"},
    "2": {"desc": "开启模型训练", "module": "src.training.train"},
    "3": {"desc": "模型格式导出", "module": "src.training.export"},
    "4": {"desc": "实时推理演示", "module": "src.demo.demo"},
}

@app.get("/")
async def get():
    return HTMLResponse(Path("web\index.html").read_text(encoding="utf-8"))

@app.websocket("/ws/run/{cmd_id}")
async def websocket_endpoint(websocket: WebSocket, cmd_id: str):
    await websocket.accept()
    
    if cmd_id not in MENU_OPTIONS:
        await websocket.send_text("### ❌ 错误：无效的任务 ID")
        await websocket.close()
        return

    module = MENU_OPTIONS[cmd_id]["module"]
    await websocket.send_text(f"### 🚀 正在启动: {MENU_OPTIONS[cmd_id]['desc']}\n---")

    # 使用 subprocess 运行模块，并捕获输出
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", module],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # 实时读取子进程输出并推送到前端
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # 过滤掉一些干扰字符，直接发给前端渲染
                await websocket.send_text(line.strip())
        
        await websocket.send_text(f"\n---\n### ✅ 任务已完成 (退出码: {process.returncode})")
    except Exception as e:
        await websocket.send_text(f"\n### ❌ 运行异常: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    # 自动打开 Chrome 浏览器
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)