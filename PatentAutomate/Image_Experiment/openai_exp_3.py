import os
import sys
import base64
import argparse

# Add utils/ to the Python path dynamically
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

from utils import client

# Updated prompt remains unchanged
SYSTEM_PROMPT = """You are a specialized patent drafting assistant. You generate detailed technical descriptions based on patent claim text and diagrammatic figures.

Your response must:
- Use formal patent language (but avoid invention-centric phrases).
- Refer to specific labeled elements in the image (e.g., "element 102", "member 103", etc.)
- Describe structural relationships, mechanisms, and configurations.
- Never mention claims explicitly or use phrases like "as claimed".
- Avoid markdown formatting or bullet points — write in complete, flowing paragraphs.
- Do not include patent profanity 

Strictly avoid:
- Advocating or subjective language such as "critical", "important", "necessary", "the invention", "must", etc.
- Any language that limits claim scope or implies essentiality
- Markdown formatting or bullets
"""

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_multimodal_description(claims_text, component_list_text, image_path):
    image_base64 = encode_image_base64(image_path)

    # Combine claims and component list into one user prompt
    user_text = (
        "Write a detailed technical description of the invention using the claims, image, "
        "and the labeled components list. Describe the structural configuration and reference labeled parts of the figure where applicable:\n\n"
        "--- CLAIMS ---\n"
        f"{claims_text}\n\n"
        "--- COMPONENT LIST ---\n"
        f"{component_list_text}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text
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
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def main():
    claims_path = "../claims/claims_image.txt"
    components_path = "components_image.txt"  # <-- NEW FILE containing component list
    image_path = "./Patent_fig_1.png"
    output_path = "multimodal_description_exp3.txt"

    with open(claims_path, "r", encoding="utf-8") as f:
        claims_text = f.read()

    with open(components_path, "r", encoding="utf-8") as f:
        component_list_text = f.read()

    # Updated call to pass both claims and component list
    description = generate_multimodal_description(claims_text, component_list_text, image_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(description)

    print(f"\nMultimodal description saved to: {output_path}")

if __name__ == "__main__":
    main()
