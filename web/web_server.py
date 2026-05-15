import os
import asyncio
import sys
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pathlib import Path
from winpty import PtyProcess

app = FastAPI()

# 确保 Python 输出不被缓存，这样网页才能实时看到结果
os.environ["PYTHONUNBUFFERED"] = "1"

@app.get("/")
async def get():
    # 假设 index.html 放在同级目录
    return HTMLResponse(Path("web\index.html").read_text(encoding="utf-8"))

@app.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # 1. 启动 Windows 伪终端进程
    # 使用 sys.executable 确保调用的是当前虚拟环境的 python
    # 运行 python -m main 触发你的交互菜单
    proc = PtyProcess.spawn([sys.executable, "-m", "main"])
    
    async def read_from_pty():
        """读取进程输出发送到网页"""
        try:
            while True:
                # 采用非阻塞读取，Windows 下由于 winpty 限制，我们需要一个小循环
                if proc.isalive():
                    data = proc.read(1024)
                    if data:
                        await websocket.send_text(data)
                await asyncio.sleep(0.01) # 防止死循环占用 100% CPU
        except Exception as e:
            print(f"Read error: {e}")

    async def write_to_pty():
        """接收网页输入写入进程"""
        try:
            while True:
                data = await websocket.receive_text()
                if proc.isalive():
                    proc.write(data)
        except Exception as e:
            print(f"Write error: {e}")

    # 并发处理读写
    try:
        await asyncio.gather(read_from_pty(), write_to_pty())
    except Exception:
        pass
    finally:
        if proc.isalive():
            proc.terminate()

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    # 启动后自动打开 Chrome
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)