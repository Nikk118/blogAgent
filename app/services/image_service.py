import os
import re

from pathlib import Path


def gemini_generate_image_bytes(prompt: str) -> bytes:

    from google import genai
    from google.genai import types

    api_key = os.environ.get(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set."
        )

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(

        model="gemini-2.5-flash-image",

        contents=prompt,

        config=types.GenerateContentConfig(

            response_modalities=["IMAGE"],

            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(
        response,
        "parts",
        None
    )

    if (
        not parts
        and getattr(response, "candidates", None)
    ):

        try:
            parts = (
                response
                .candidates[0]
                .content
                .parts
            )

        except Exception:
            parts = None

    if not parts:

        raise RuntimeError(
            "No image content returned."
        )

    for part in parts:

        inline = getattr(
            part,
            "inline_data",
            None
        )

        if (
            inline
            and getattr(inline, "data", None)
        ):
            return inline.data

    raise RuntimeError(
        "No inline image bytes found."
    )


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

    output_dir = Path(
        "generated_blogs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    safe_title = re.sub(
        r'[<>:"/\\|?*]',
        '',
        plan.blog_title
    )

    blog_filename = (
        f"{safe_title.strip().replace(' ', '_').lower()}.md"
    )

    blog_path = (
        output_dir / blog_filename
    )

    images_dir = (
        output_dir / "images"
    )

    images_dir.mkdir(
        exist_ok=True
    )

    if not image_specs:

        blog_path.write_text(
            md,
            encoding="utf-8"
        )

        return {
            "final": md
        }

    for spec in image_specs:

        placeholder = spec["placeholder"]

        filename = spec["filename"]

        out_path = (
            images_dir / filename
        )

        if not out_path.exists():

            try:

                img_bytes = (
                    gemini_generate_image_bytes(
                        spec["prompt"]
                    )
                )

                out_path.write_bytes(
                    img_bytes
                )

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

                md = md.replace(
                    placeholder,
                    prompt_block
                )

                continue

        img_md = (
            f"![{spec['alt']}](images/{filename})\n"
            f"*{spec['caption']}*"
        )

        md = md.replace(
            placeholder,
            img_md
        )

    blog_path.write_text(
        md,
        encoding="utf-8"
    )

    return {
        "final": md
    }