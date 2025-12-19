import json
from typing import List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI()

# Allow browser access (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.viewers: Set[WebSocket] = set()
        self.pi_connections: Set[WebSocket] = set()

    async def connect_viewer(self, websocket: WebSocket):
        await websocket.accept()
        self.viewers.add(websocket)

    def disconnect_viewer(self, websocket: WebSocket):
        self.viewers.discard(websocket)

    async def connect_pi(self, websocket: WebSocket):
        await websocket.accept()
        self.pi_connections.add(websocket)

    def disconnect_pi(self, websocket: WebSocket):
        self.pi_connections.discard(websocket)

    async def broadcast_to_viewers(self, message: str):
        dead: List[WebSocket] = []
        for viewer in list(self.viewers):
            try:
                await viewer.send_text(message)
            except Exception:
                dead.append(viewer)
        for v in dead:
            self.viewers.discard(v)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def index():
    # Simple HTML page with canvas and JS to draw bounding boxes
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Smart Camera Viewer</title>
  <style>
    body {
      margin: 0;
      background: #111;
      color: #eee;
      display: flex;
      flex-direction: column;
      align-items: center;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      height: 100vh;
    }
    header {
      padding: 10px 16px;
      width: 100%;
      box-sizing: border-box;
      background: #1e1e1e;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    #status {
      font-size: 14px;
      color: #8bc34a;
    }
    #wrapper {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      overflow: auto;
    }
    canvas {
      background: #000;
      box-shadow: 0 0 20px rgba(0,0,0,0.8);
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
  <header>
    <div>Smart Camera Viewer (Backpack Detection)</div>
    <div id="status">Connecting...</div>
  </header>
  <div id="wrapper">
    <canvas id="canvas"></canvas>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = wsProtocol + "://" + window.location.host + "/ws/viewer";
    let ws;

    function connect() {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        statusEl.textContent = "Connected";
        statusEl.style.color = "#8bc34a";
      };

      ws.onclose = () => {
        statusEl.textContent = "Disconnected (reconnecting...)";
        statusEl.style.color = "#ff9800";
        setTimeout(connect, 2000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (!data.image) {
            return;
          }
          const boxes = data.boxes || [];
          const image = new Image();
          image.onload = () => {
            // Resize canvas to image dimensions
            canvas.width = image.width;
            canvas.height = image.height;

            // Draw base image
            ctx.drawImage(image, 0, 0, image.width, image.height);

            // Draw boxes
            ctx.lineWidth = 2;
            ctx.strokeStyle = "lime";
            ctx.font = "16px sans-serif";
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.textBaseline = "top";

            boxes.forEach((b) => {
              // b = [x, y, w, h, conf] all normalized (0–1)
              const x = b[0] * image.width;
              const y = b[1] * image.height;
              const w = b[2] * image.width;
              const h = b[3] * image.height;
              const conf = b[4];

              ctx.beginPath();
              ctx.rect(x, y, w, h);
              ctx.stroke();

              const label = "Backpack " + (conf * 100).toFixed(1) + "%";
              const textWidth = ctx.measureText(label).width;
              const padding = 4;
              const boxHeight = 18;

              // Label background
              ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
              ctx.fillRect(x, y - boxHeight, textWidth + padding * 2, boxHeight);

              // Label text
              ctx.fillStyle = "lime";
              ctx.fillText(label, x + padding, y - boxHeight + 2);

              ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            });
          };
          image.src = "data:image/jpeg;base64," + data.image;
        } catch (e) {
          console.error("Error handling message:", e);
        }
      };
    }

    connect();
  </script>
</body>
</html>
    """
    return HTMLResponse(html_content)


@app.websocket("/ws/viewer")
async def websocket_viewer(websocket: WebSocket):
    await manager.connect_viewer(websocket)
    try:
        while True:
            # Viewers only receive data; no need to read messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_viewer(websocket)
    except Exception:
        manager.disconnect_viewer(websocket)


@app.websocket("/ws/pi")
async def websocket_pi(websocket: WebSocket):
    await manager.connect_pi(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            # Optional: validate JSON
            try:
                data = json.loads(message)
                if "image" not in data or "boxes" not in data:
                    continue
            except json.JSONDecodeError:
                continue
            await manager.broadcast_to_viewers(message)
    except WebSocketDisconnect:
        manager.disconnect_pi(websocket)
    except Exception:
        manager.disconnect_pi(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7777, reload=False)


