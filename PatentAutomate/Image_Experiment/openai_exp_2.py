import os
import sys
import base64
import argparse

# Add utils/ to the Python path dynamically
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

from utils import client

# 🔧 System prompt focused on extracting labeled parts ONLY
SYSTEM_PROMPT = """You are a specialized assistant for analyzing patent figures. Your task is to extract labeled components from the diagram based on both the figure and the claims.

Your response must:
- Output a numbered list of the labeled components from the figure (e.g., 102, 104, etc.).
- For each label, include a short description (just a few words or one sentence).
- Only include information visible or implied from the image and claims.
- Write in clean, plain English.
- Do NOT generate full paragraphs or patent-style prose.
- Do NOT mention or describe the claims explicitly.
- Do NOT include any markdown or formatting (no bold, italics, or code). Just plain text.

Strictly avoid:
- Advocating or subjective language such as "critical", "important", "necessary", "the invention", "must", etc.
- Any language that limits claim scope or implies essentiality
- Markdown formatting or bullets
"""

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_component_list(claims_text, image_path):
    image_base64 = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "List all labeled components visible in the image below. "
                            "For each, provide a short description. Do not repeat claim language.\n\n"
                            f"{claims_text}"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

def generate_final_description(claims_text, image_path, component_list_text):
    image_base64 = encode_image_base64(image_path)

    FINAL_DESCRIPTION_PROMPT = """You are a specialized patent drafting assistant. Your job is to write a detailed technical description of a patent figure using the provided claims and the list of identified components.

Guidelines:
- Use formal patent language (e.g., “The apparatus includes…”, “The system comprises…”).
- Refer to labeled elements by their numbers (e.g., “element 102”, “member 104”).
- Use the component list to guide your terminology and naming.
- Do not copy claim language. Instead, explain how components work or interact.
- Focus on structure, configuration, operation, and technical details.
- Do NOT include any markdown or bullet points — write flowing paragraphs only.
- The style should match that of a patent detailed description section.

Now generate the technical description using the image, claims, and labeled component list below.
"""

    user_prompt = (
        f"{FINAL_DESCRIPTION_PROMPT}\n\n"
        f"--- CLAIMS ---\n{claims_text}\n\n"
        f"--- COMPONENT LIST ---\n{component_list_text}\n\n"
        f"Write the description below:"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FINAL_DESCRIPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def main():
    claims_path = "../claims/claims_image.txt"
    image_path = "./Patent_fig_1.png"
    component_list_path = "image_component_list_generated.txt"
    component_output_path = "image_component_list_generated.txt"
    final_output_path = "multimodal_description_exp2.txt"

    # Step 1: Read claims
    with open(claims_path, "r", encoding="utf-8") as f:
        claims_text = f.read()

    # Step 2: Generate list of components
    component_list = generate_component_list(claims_text, image_path)
    with open(component_output_path, "w", encoding="utf-8") as f:
        f.write(component_list)
    print(f"\nComponent list saved to: {component_output_path}")

    # Step 3: Generate full technical description using component list + image + claims
    full_description = generate_final_description(claims_text, image_path, component_list)
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(full_description)
    print(f"\nPatent description saved to: {final_output_path}")

if __name__ == "__main__":
    main()
