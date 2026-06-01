import os
import re
from pathlib import Path
import httpx
from urllib.parse import quote
import time

def gemini_generate_image_bytes(prompt: str, retries: int = 3) -> bytes:
    encoded_prompt = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    for attempt in range(retries):
        try:
            response = httpx.get(url, timeout=180, follow_redirects=True)
            
            if response.status_code == 402:
                wait = (attempt + 1) * 10  
                print(f"[Pollinations] 402 rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            if response.status_code != 200:
                raise RuntimeError(f"Pollinations failed: {response.status_code}")
            
            return response.content

        except httpx.ReadTimeout:
            wait = (attempt + 1) * 10
            print(f"[Pollinations] Timeout on attempt {attempt+1}, waiting {wait}s...")
            time.sleep(wait)
    
    raise RuntimeError("Pollinations failed after 3 retries")
def generate_and_place_image(state: dict) -> dict:

    plan = state["plan"]

    md = (
        state.get("md_with_placeholders")
        or state.get("merged_md", "")
    )

    image_specs = (
        state.get("image_specs")
        or []
    )

    output_dir = Path("generated_blogs")
    output_dir.mkdir(exist_ok=True)

    safe_title = re.sub(r'[<>:"/\\|?*]', '', plan.blog_title)
    blog_filename = f"{safe_title.strip().replace(' ', '_').lower()}.md"
    blog_path = output_dir / blog_filename

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ✅ track generated images for DB storage
    generated_images = []

    if not image_specs:
        blog_path.write_text(md, encoding="utf-8")
        return {"final": md, "generated_images": []}

    for spec in image_specs:

        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        img_bytes = None

        if not out_path.exists():
            try:
                img_bytes = gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)

            except Exception as e:
                prompt_block = f"""
> IMAGE GENERATION FAILED

Caption:
{spec.get('caption', '')}

ALT:
{spec.get('alt', '')}

PROMPT:
{spec['prompt']}

ERROR:
{e}
"""
                md = md.replace(placeholder, prompt_block)
                continue

        else:
            # already on disk, read bytes for DB
            img_bytes = out_path.read_bytes()

        # ✅ collect for DB
        generated_images.append({
            "filename": filename,
            "alt": spec.get("alt", ""),
            "caption": spec.get("caption", ""),
            "image_data": img_bytes,
        })

        img_md = (
            f"![{spec['alt']}](images/{filename})\n"
            f"*{spec['caption']}*"
        )
        md = md.replace(placeholder, img_md)

    blog_path.write_text(md, encoding="utf-8")

    return {
        "final": md,
        "generated_images": generated_images  # ✅ FastAPI reads this to save to DB
    }