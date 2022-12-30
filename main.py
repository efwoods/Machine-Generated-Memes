from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml import obtain_image
import io
import json
from PIL import Image
from image_manager import Image_Manager
import datetime

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "message":"Hello World"}

class Item(BaseModel):
    name: str
    price: float
    tags: list = []

@app.post("/items")
def create_item(item: Item):
    return item

@app.get("/generate-memory")
def generate_image_memory(
    prompt: str,
    *,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt, 
        num_inference_steps=num_inference_steps, 
        seed=seed,
        guidance_scale=guidance_scale
    )
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

@app.get("/generate")
def generate_image(
    prompt: str,
    *,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt, 
        num_inference_steps=num_inference_steps, 
        seed=seed,
        guidance_scale=guidance_scale
    )
    image.save("image.png")
    
    return FileResponse("image.png")

@app.get("/generate-meme")
def generate_image(
    prompt: str,
    *,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt, 
        num_inference_steps=num_inference_steps, 
        seed=seed,
        guidance_scale=guidance_scale
    )
    image.save("image.png")
    response = json.loads(prompt)
    with Image.open(f"image.png").convert(
        "RGBA"
    ) as base:
        overlay_image = Image_Manager.add_text(
            base=base,
            text=prompt,
            position=(500, 385),
            font_size=30,
            text_color="black",
            rotate_degrees=20,
            wrapped_width=22,
        )
        watermark = Image_Manager.add_text(
                base=base,
                text="machine-generated-memes",
                position=(25, 600),
                font_size=25,
                text_color="white",
            )
    base = Image.alpha_composite(base, watermark)
    out = Image.alpha_composite(base, overlay_image)
    if out.mode in ("RGBA", "P"):
        out = out.convert("RGB")
        date = datetime.datetime.now()
        image_name = f"{date}.jpg"
        file_location = f"makememe/static/creations/{image_name}"
        out.save(file_location)
    return image_name
    # return FileResponse("image.png")