"""Local API bridge for Asset Lab.

Keeps provider keys on the local machine and proxies OpenAI-compatible calls
from the browser to the configured upstream service.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
import cv2
import numpy as np
from sam2_segmentor import get_segmentor, last_error

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.local.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}

def sam2_status(cfg: dict) -> dict:
    checkpoint = cfg.get("sam2_checkpoint", "")
    return {"configured": bool(checkpoint and Path(checkpoint).is_file() and cfg.get("sam2_config")), "device": cfg.get("sam2_device", "cuda"), "fallback": "grabcut", "error": last_error()}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def upstream(path: str, payload: dict, api_key: str, base_url: str) -> tuple[int, bytes]:
    url = base_url.rstrip("/") + path
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180,
        )
        return response.status_code, response.content
    except Exception as exc:  # provider errors are returned as JSON to the UI
        return 502, json.dumps({"error": {"message": str(exc)}}).encode("utf-8")


def image_edit(image_data: str, prompt: str, api_key: str, base_url: str, model: str, size: str, quality: str) -> tuple[int, bytes]:
    try:
        header, encoded = image_data.split(",", 1)
        raw = base64.b64decode(encoded)
        mime = header.split(";")[0].split(":", 1)[1]
        ext = mime.split("/")[-1].replace("jpeg", "jpg")
        response = requests.post(
            base_url.rstrip("/") + "/images/edits",
            files={"image": (f"source.{ext}", raw, mime)},
            data={"model": model, "prompt": prompt, "size": size, "quality": quality, "n": "1"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180,
        )
        return response.status_code, response.content
    except Exception as exc:
        return 502, json.dumps({"error": {"message": str(exc)}}).encode("utf-8")


def inline_image_urls(raw: bytes) -> bytes:
    """Download provider URLs server-side so the browser has a stable preview."""
    try:
        result = json.loads(raw.decode("utf-8"))
        for item in result.get("data", []):
            url = item.get("url")
            if url and not item.get("b64_json"):
                image = requests.get(url, timeout=90)
                image.raise_for_status()
                item["b64_json"] = base64.b64encode(image.content).decode("ascii")
                item["mime_type"] = image.headers.get("Content-Type", "image/png").split(";")[0]
        return json.dumps(result, ensure_ascii=False).encode("utf-8")
    except Exception:
        return raw


def segment_assets(image_data: str, assets: list[dict]) -> list[dict]:
    """Create local transparent PNG previews from model-provided boxes.

    This lightweight GrabCut stage is intentionally conservative; SAM2 can
    replace it later without changing the browser contract.
    """
    _, encoded = image_data.split(",", 1)
    source = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("图片无法解码")
    h, w = source.shape[:2]
    output = []
    cfg = load_config()
    sam = get_segmentor(cfg.get("sam2_checkpoint", ""), cfg.get("sam2_config", ""), cfg.get("sam2_device", "cuda"))
    valid = []
    for asset in assets[:20]:
        b = asset.get("bbox", [0, 0, 1, 1])
        x, y, bw, bh = [float(v) for v in b[:4]]
        x1, y1 = max(0, int(x * w)), max(0, int(y * h))
        x2, y2 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        valid.append((asset, x1, y1, x2, y2))
    sam_masks = []
    if sam and valid:
        try:
            sam_masks = sam.masks_for(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), [[x1, y1, x2, y2] for _, x1, y1, x2, y2 in valid])
        except Exception:
            sam_masks = []
    for index, (asset, x1, y1, x2, y2) in enumerate(valid):
        crop = source[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if sam_masks and index < len(sam_masks):
            alpha = (sam_masks[index][y1:y2, x1:x2] * 255).astype(np.uint8)
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = alpha
        else:
            mask = np.zeros((ch, cw), np.uint8); inset = max(2, min(cw, ch) // 40)
            rect = (inset, inset, max(1, cw - 2 * inset), max(1, ch - 2 * inset))
            try:
                bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(crop, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
                alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = alpha
            except cv2.error:
                rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = 255
        ok, encoded_png = cv2.imencode(".png", rgba)
        if ok:
            item = dict(asset)
            item["image_data"] = "data:image/png;base64," + base64.b64encode(encoded_png.tobytes()).decode("ascii")
            output.append(item)
    return output


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        if content_type == "application/json":
            content_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            cfg = load_config()
            result = {
                "image_configured": bool(cfg.get("image_api_key")),
                "vision_configured": bool(cfg.get("vision_api_key")),
                "image_model": cfg.get("image_model", "gpt-image-2"),
                "vision_model": cfg.get("vision_model", "gpt-5.6"),
                "sam2": sam2_status(cfg),
            }
            self._send(200, json.dumps(result).encode("utf-8"))
            return
        self.serve_file(self.path)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"error":{"message":"Invalid JSON"}}')
            return
        cfg = load_config()
        base = cfg.get("base_url", "https://yjapi.manqiaotechnology.com/v1")
        if self.path == "/api/image/generate":
            key = cfg.get("image_api_key", "")
            if not key:
                self._send(400, json.dumps({"error": {"message": "请先在 config.local.json 填写 image_api_key"}}, ensure_ascii=False).encode("utf-8"))
                return
            payload.setdefault("model", cfg.get("image_model", "gpt-image-2"))
            status, raw = upstream("/images/generations", payload, key, base)
            self._send(status, inline_image_urls(raw) if status < 400 else raw)
            return
        if self.path == "/api/image/edit":
            key = cfg.get("image_api_key", "")
            image_data = payload.get("image_data", "")
            if not key:
                self._send(400, json.dumps({"error": {"message": "请先在 config.local.json 填写 image_api_key"}}, ensure_ascii=False).encode("utf-8"))
                return
            if not image_data.startswith("data:image/"):
                self._send(400, b'{"error":{"message":"image_data must be a data URL"}}')
                return
            mode = payload.get("mode", "保真增强")
            prompts = {
                "保真增强": "严格保留原图中的角色身份、四条手臂、蛇身结构、面具、胸甲、法器、比例、姿势和构图。只做边缘清理、背景清洁、轻微锐化。禁止新增、删除、替换或重新设计任何部件。",
                "局部补全": "严格保留原图角色的四条手臂、蛇身、金属面具、胸甲、法器和构图。只补全被遮挡或缺失的小区域，材质和设计与相邻区域一致。禁止改变角色结构。",
                "创意变体": "保持原图角色的四条手臂、蛇身轮廓、姿势和主要装备不变，只改变指定的颜色或材质细节；不要重新设计角色，不要增删肢体。",
            }
            status, raw = image_edit(image_data, prompts.get(mode, prompts["保真增强"]), key, base, cfg.get("image_model", "gpt-image-2"), payload.get("size", "1024x1024"), payload.get("quality", "low"))
            self._send(status, inline_image_urls(raw) if status < 400 else raw)
            return
        if self.path == "/api/analyze":
            key = cfg.get("vision_api_key", "")
            image_data = payload.get("image_data", "")
            if not key:
                self._send(400, json.dumps({"error": {"message": "请先在 config.local.json 填写 vision_api_key"}}, ensure_ascii=False).encode("utf-8"))
                return
            if not image_data.startswith("data:image/"):
                self._send(400, b'{"error":{"message":"image_data must be a data URL"}}')
                return
            target = payload.get("target", "全部对象")
            vision_prompt = (
                "你是游戏美术资产分析器。请分析这张图片，列出可独立拆出的游戏资产。"
                f"重点范围：{target}。只返回 JSON，不要 Markdown，不要解释。格式为 "
                '{"assets":[{"name":"中文名称","type":"角色|装备|服装|场景","score":0.0,"bbox":[x,y,w,h],"notes":"简短说明"}]}。'
                "同一物件的不同视角合并为一项，最多返回 20 项，score 为 0 到 1。"
                "bbox 使用 0 到 1 的归一化坐标，表示对象在原图中的 x、y、宽度、高度；无法定位时使用 [0,0,1,1]。"
            )
            chat_payload = {"model": cfg.get("vision_model", "gpt-5.6"), "messages": [{"role": "user", "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": image_data}},
            ]}], "temperature": 0.1, "max_tokens": 1200}
            status, raw = upstream("/chat/completions", chat_payload, key, base)
            if status >= 400:
                self._send(status, raw)
                return
            try:
                response = json.loads(raw.decode("utf-8"))
                content = response["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                content = content.strip().replace("```json", "").replace("```", "").strip()
                result = json.loads(content)
                assets = []
                for i, asset in enumerate(result.get("assets", [])):
                    score = max(0.0, min(1.0, float(asset.get("score", 0.7))))
                    raw_box = asset.get("bbox", [0, 0, 1, 1])
                    try:
                        box = [max(0.0, min(1.0, float(v))) for v in raw_box[:4]]
                        if len(box) != 4: box = [0, 0, 1, 1]
                    except (TypeError, ValueError):
                        box = [0, 0, 1, 1]
                    assets.append({"name": str(asset.get("name", "未命名资产")), "type": str(asset.get("type", "场景")), "score": round(score * 100), "bbox": box, "icon": f"{i+1:02d}", "c": "#c8f36b"})
                self._send(200, json.dumps({"assets": assets}, ensure_ascii=False).encode("utf-8"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send(502, json.dumps({"error": {"message": f"视觉模型返回格式异常: {exc}"}}, ensure_ascii=False).encode("utf-8"))
            return
        if self.path == "/api/segment":
            try:
                segmented = segment_assets(payload.get("image_data", ""), payload.get("assets", []))
                self._send(200, json.dumps({"assets": segmented}, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                self._send(400, json.dumps({"error": {"message": str(exc)}}, ensure_ascii=False).encode("utf-8"))
            return
        if self.path == "/api/chat":
            key = cfg.get("vision_api_key", "")
            if not key:
                self._send(400, json.dumps({"error": {"message": "请先在 config.local.json 填写 vision_api_key"}}, ensure_ascii=False).encode("utf-8"))
                return
            payload.setdefault("model", cfg.get("vision_model", "gpt-5.6"))
            self._send(*upstream("/chat/completions", payload, key, base))
            return
        self._send(404, b'{"error":{"message":"Not found"}}')

    def serve_file(self, url_path: str) -> None:
        relative = url_path.lstrip("/") or "index.html"
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents and target != ROOT:
            self._send(403, b"Forbidden", "text/plain")
            return
        if not target.is_file():
            self._send(404, b"Not found", "text/plain")
            return
        data = target.read_bytes()
        self._send(200, data, mimetypes.guess_type(str(target))[0] or "application/octet-stream")


if __name__ == "__main__":
    port = int(os.environ.get("ASSET_LAB_PORT", "8080"))
    print(f"Asset Lab running at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
