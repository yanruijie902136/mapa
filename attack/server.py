import argparse
import asyncio
import gc
import io
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from hf_models.utils import load_model
from PIL import Image

logging.basicConfig(level=logging.INFO)


class LazyModelManager:
    def __init__(self, keep_loaded=False):
        self.current_name = None
        self.current_model = None
        self.keep_loaded = keep_loaded
        self.lock = asyncio.Lock()

    async def run(self, model_name, callback):
        async with self.lock:
            model = self._load_if_needed(model_name)
            try:
                return callback(model)
            finally:
                if not self.keep_loaded:
                    self.unload()

    def _load_if_needed(self, model_name):
        if self.current_name == model_name and self.current_model is not None:
            return self.current_model

        self.unload()
        self.current_name = model_name
        self.current_model = load_model(model_name)
        return self.current_model

    def unload(self):
        if self.current_model is not None:
            logging.info("Unloading model: %s", self.current_name)
        self.current_name = None
        self.current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                logging.debug("torch.cuda.ipc_collect() skipped", exc_info=True)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="qwen2_5_7b")
    parser.add_argument("--llm_7b_name", type=str, default="qwen2_5_7b")
    parser.add_argument(
        "--lvlm_name",
        type=str,
        default="qwen2_5_vl_3b",
        choices=["", "llava", "llama", "qwen", "qwen2_5_vl_3b", "qwen3_vl_4b"],
    )
    parser.add_argument("--load_clip", action="store_true")
    parser.add_argument("--load_sd", action="store_true")
    parser.add_argument("--load_sem_relevance", action="store_true")
    parser.add_argument("--load_toxigen", action="store_true")
    parser.add_argument("--load_llama_guard", action="store_true")
    parser.add_argument("--keep_model_loaded", action="store_true")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--port", type=int, default=8000)

    return parser


async def read_images(files: List[UploadFile] = None):
    images = []
    if files:
        for file in files:
            contents = await file.read()
            try:
                images.append(Image.open(BytesIO(contents)).convert("RGB"))
            except Exception as exc:
                return None, {"error": f"Could not process {file.filename}: {exc}"}
    return images, None


def main(args):
    app = FastAPI()
    model_manager = LazyModelManager(keep_loaded=args.keep_model_loaded)
    allowed_lvlm_model_aliases = {
        "llava",
        "llama",
        "qwen",
        "qwen2_5_vl_3b",
        "qwen3_vl_4b",
    }
    allowed_llm_models = {
        args.llm_name,
        args.llm_7b_name,
        "qwen2_5_7b",
        "llama3_1_8b",
        "harmbench_mistral_cls",
        "strongreject_judge",
        "harmbench_judge",
        "mistral",
        "mistral_7b",
    }

    @app.middleware("http")
    async def log_request_time(request: Request, call_next):
        now = datetime.now().isoformat()
        logging.info("%s %s at %s", request.method, request.url, now)
        return await call_next(request)

    @app.on_event("shutdown")
    async def unload_model_on_shutdown():
        model_manager.unload()

    @app.get("/llm_gen")
    async def llm_generation(request: Request):
        data = await request.json()
        model_name = data.get("model_name")
        if model_name not in allowed_llm_models:
            raise HTTPException(status_code=400, detail=f"Model not found: {model_name}.")

        return await model_manager.run(
            model_name,
            lambda model: model.generate(
                data.get("full_prompt"),
                data.get("max_new_tokens"),
                data.get("temperature"),
                data.get("top_p"),
            ),
        )

    if args.lvlm_name:
        allowed_lvlm_models = set(allowed_lvlm_model_aliases)
        if args.load_llama_guard:
            allowed_lvlm_models.add("llama_guard")

        @app.post("/lvlm_gen")
        async def lvlm_generation(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            metadata_dict = json.loads(metadata)
            model_name = metadata_dict.get("model_name") or args.lvlm_name
            if model_name not in allowed_lvlm_models:
                raise HTTPException(status_code=400, detail=f"Model not found or not enabled: {model_name}.")

            images, error = await read_images(files)
            if error:
                return error

            return await model_manager.run(
                model_name,
                lambda model: model.generate(
                    metadata_dict["conv"],
                    images,
                    metadata_dict["max_new_tokens"],
                    metadata_dict["temperature"],
                    metadata_dict["top_p"],
                ),
            )

    if args.load_clip:

        @app.get("/clip_gen")
        async def clip_generation(request: Request):
            data = await request.json()
            return await model_manager.run("clip", lambda model: model.encode(data.get("text")))

    if args.load_sd:

        @app.get("/sd_gen")
        async def sd_generation(request: Request):
            data = await request.json()
            img = await model_manager.run(
                "stable_diffusion",
                lambda model: model.generate(
                    data.get("text"),
                    data.get("size_px"),
                    data.get("num_inference_steps"),
                    data.get("guidance_scale"),
                ),
            )
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="image/png")

    if args.load_sem_relevance:

        @app.post("/sem_relevance")
        async def sem_relevance_generate(request: Request):
            data = await request.json()
            similarity = await model_manager.run(
                "sem_relevance",
                lambda model: model.compute_similarity(data.get("text1", ""), data.get("text2", "")),
            )
            return {"similarity": similarity}

    if args.load_toxigen:
        from hf_models import Toxigen

        toxigen_model: Toxigen = Toxigen()

        @app.post("/toxigen")
        async def toxigen_generate(request: Request):
            data = await request.json()
            texts = [text[:256] for text in data.get("texts", [])]
            return {"result": toxigen_model.get_toxicity(texts)}

    if args.load_llama_guard:

        @app.post("/llama_guard")
        async def llama_guard_generate(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            metadata_dict = json.loads(metadata)
            images, error = await read_images(files)
            if error:
                return error

            return await model_manager.run(
                "llama_guard",
                lambda model: model.generate(
                    metadata_dict["conv"],
                    images,
                    metadata_dict["max_new_tokens"],
                    metadata_dict["temperature"],
                    metadata_dict["top_p"],
                ),
            )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main(config().parse_args())