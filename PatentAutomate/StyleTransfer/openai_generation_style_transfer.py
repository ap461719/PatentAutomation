import re
import argparse
import sys
import os
from docx import Document

# -----------------------------------------------------------
#  Utils import
# -----------------------------------------------------------
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils",
    )
)
from utils import chatgpt_chatbot, EMBODIMENT_PROMPT_IN_CONTEXT  # noqa: E402

# -----------------------------------------------------------
#  System & user messages (unchanged)
# -----------------------------------------------------------
SYSTEM_MESSAGE = """You are a specialized patent attorney assistant designed to generate detailed descriptions of patent applications. 

Your expertise includes:
- Writing in precise legal language style appropriate for patent applications
- Understanding patent claim structures and dependencies
- Generating comprehensive technical descriptions that support patent claims
- Avoiding repetitive language while maintaining legal precision

CRITICAL - AVOID PATENT PROFANITY:
- NEVER explicitly reference claim numbers or use phrases like "as described in claim X", "this claim specifies", "according to claim 1", "claim 2 teaches"
- NEVER use invention-centric language like "The present invention", "The invention provides", "According to the invention", "The inventive system"
- NEVER start sentences with "The present invention relates to" or "The system of the present invention"
- NEVER use vague references like "as claimed herein", "according to the disclosure", "as taught by the invention"
- Use claims only as context to understand what technical elements to describe
- Focus on technical implementation details rather than restating claim language
- Do not include redundant explanations of the same concepts
- Use formal, technical language appropriate for patent documentation
- Do not include any markdown formatting in your responses
- Ensure descriptions are detailed enough to support the patent claims without directly referencing them
- Maintain consistency in terminology throughout the description
- Start sentences with specific technical components, not generic invention references"""

INIT_INSTRUCTION = '''Generate a technical description focusing on molecular mechanisms, structural design rationale, and technical implementation. Use varied sentence openings and avoid repetitive language patterns. Avoid all patent profanity including invention-centric language and claim references:

{claim}'''


CONTINUE_INSTRUCTION = '''Continue with additional technical details using completely different sentence structures and terminology. Focus on new technical aspects without repeating previous explanations. Avoid all patent profanity including invention-centric language and claim references:

{claim}'''

# -----------------------------------------------------------
#  Style-transfer helper (NEW)
# -----------------------------------------------------------
STYLE_PROMPT = """
Rewrite the following patent-description paragraph so it reads like formal USPTO prose:
tighten legal language, vary sentence starts, remove repetition, and keep it 100 % factual.
Do NOT add new subject matter—only polish wording.

Paragraph:
"{para}"
"""

def stylize(paragraph: str) -> str:
    """One short GPT call that polishes a single paragraph."""
    msgs = [
        {"role": "system", "content": "You are an expert patent editor."},
        {"role": "user",   "content": STYLE_PROMPT.format(para=paragraph)},
    ]
    return chatgpt_chatbot(msgs).strip()

# -----------------------------------------------------------
#  TXT → DOCX (unchanged)
# -----------------------------------------------------------
def txt_to_docx(txt_path: str, docx_path: str) -> None:
    doc = Document()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            doc.add_paragraph(line.rstrip())
    doc.save(docx_path)
    print(f"Also exported DOCX to '{docx_path}'")

# -----------------------------------------------------------
#  Claim splitting (unchanged)
# -----------------------------------------------------------
def split_claims(claims: str) -> list[str]:
    pattern = re.compile(r"^\d{1,3}\.")
    lines, buf, out = claims.split("\n"), "", []
    for line in reversed(lines):
        if pattern.match(line.strip()):
            out.insert(0, line + ("\n" + buf if buf else ""))
            buf = ""
        else:
            buf = line + buf
    return out

# -----------------------------------------------------------
#  Chunk-wise generation **with style transfer**
# -----------------------------------------------------------
def generate_chunked_desc(claims: str) -> str:
    claims_split = split_claims(claims)
    polished = []

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": INIT_INSTRUCTION.format(claim=claims_split[0])},
    ]

    # first chunk
    raw = chatgpt_chatbot(messages)
    pretty = stylize(raw)
    polished.append(pretty)
    messages.append({"role": "assistant", "content": pretty})

    # subsequent chunks
    for claim in claims_split[1:]:
        messages.append({"role": "user", "content": CONTINUE_INSTRUCTION.format(claim=claim)})
        raw = chatgpt_chatbot(messages)
        pretty = stylize(raw)
        polished.append(pretty)
        messages.append({"role": "assistant", "content": pretty})

    return "\n\n".join(polished)

# -----------------------------------------------------------
#  Full-shot generation (one style pass)
# -----------------------------------------------------------
def generate_full_desc(claims: str) -> str:
    enhanced_prompt = """You are a specialized patent attorney assistant … {claims}"""
    msgs = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": enhanced_prompt.format(claims=claims)},
    ]
    raw = chatgpt_chatbot(msgs)
    return stylize(raw)

# -----------------------------------------------------------
#  Dispatcher (unchanged)
# -----------------------------------------------------------
def run_openai_generation(claims, method):
    if method == "full":
        return generate_full_desc(claims)
    if method == "chunked":
        return generate_chunked_desc(claims)
    raise ValueError(f"Method '{method}' not supported. Use 'full' or 'chunked'.")

# -----------------------------------------------------------
#  CLI entry-point (unchanged)
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate patent descriptions using OpenAI API")
    parser.add_argument("--claims", default="../claims/claims_doc.txt",
                        help="Path to claims file")
    parser.add_argument("--method", choices=["full", "chunked"], default="chunked",
                        help="Generation method")
    parser.add_argument("--output", default=None,
                        help="Output TXT filename")
    parser.add_argument("--docx", action="store_true",
                        help="Also export the description as .docx")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    claims_path = os.path.join(script_dir, args.claims)

    # load claims
    try:
        with open(claims_path, "r", encoding="utf-8") as f:
            claims = f.read()
        print("Loaded claims from:", claims_path)
    except Exception as e:
        print("Error reading claims file:", e)
        return

    # generate
    try:
        print(f"Generating description with '{args.method}' method …")
        description = run_openai_generation(claims, args.method)

        # write txt
        out_txt = os.path.join(script_dir,
                               args.output or f"{args.method}_description_openai.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(description)
        print("TXT saved to", out_txt)

        # optional docx
        if args.docx:
            txt_to_docx(out_txt, os.path.splitext(out_txt)[0] + ".docx")

    except Exception as e:
        print("Error generating description:", e)

if __name__ == "__main__":
    main()
