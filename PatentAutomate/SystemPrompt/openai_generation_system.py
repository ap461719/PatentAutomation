import re
import argparse
import sys
import os
from docx import Document          # ← NEW: for .docx export

# -----------------------------------------------------------
#  Utils import
# -----------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from utils import chatgpt_chatbot, EMBODIMENT_PROMPT_IN_CONTEXT

# -----------------------------------------------------------
#  System & User Messages (unchanged)
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
#  Helper: TXT → DOCX
# -----------------------------------------------------------
def txt_to_docx(txt_path: str, docx_path: str) -> None:
    """Convert a plain-text file to a .docx (one paragraph per line)."""
    doc = Document()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc.add_paragraph(line.rstrip())
    doc.save(docx_path)
    print(f"Also exported DOCX to '{docx_path}'")

# -----------------------------------------------------------
#  Claim splitting & generation logic (unchanged)
# -----------------------------------------------------------
def split_claims(claims):
    pattern = re.compile(r'^\d{1,3}\.')
    split_claims_newline = claims.split('\n')
    final_split_claims, full_line = [], ''
    for line in reversed(split_claims_newline):
        if pattern.match(line.strip()):
            line_to_add = line + "\n" + full_line if full_line else line
            final_split_claims.insert(0, line_to_add)
            full_line = ''
        else:
            full_line = line + full_line
    return final_split_claims

def generate_chunked_desc(claims):
    claims_split = split_claims(claims)
    full_generation = ""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": INIT_INSTRUCTION.format(claim=claims_split[0])}
    ]
    generation = chatgpt_chatbot(messages)
    full_generation += generation + '\n'
    messages.append({'role': 'assistant', 'content': generation})
    for claim in claims_split[1:]:
        messages.append({'role': 'user', 'content': CONTINUE_INSTRUCTION.format(claim=claim)})
        generation = chatgpt_chatbot(messages)
        full_generation += generation + '\n'
        messages.append({'role': 'assistant', 'content': generation})
    return full_generation

def generate_full_desc(claims):
    enhanced_prompt = """You are a specialized patent attorney assistant. Generate a detailed description of a patent application based on the provided claims. 

CRITICAL REQUIREMENTS:
- NEVER explicitly reference claim numbers or use phrases like "as described in claim X", "this claim specifies", "claim 1", etc.
- NEVER use invention-centric language like "The present invention", "The invention provides", "According to the invention"
- NEVER start sentences with "The present invention relates to" or "The system of the present invention"
- NEVER use vague references like "as claimed herein", "according to the disclosure"
- Use claims only as context to understand what technical elements to describe
- Avoid repetitive phrases like "The present invention relates to..." or "The present invention further..."
- Vary sentence structure and openings throughout the description
- Focus on technical implementation details rather than restating claim language
- Do not include redundant explanations of the same concepts
- Write in formal, technical language appropriate for patent documentation
- Do not include any markdown formatting
- Generate a comprehensive description that supports all the claims without directly referencing them
- Start sentences with specific technical components or systems, not generic invention references

Given the claims below, generate a detailed technical description:

{claims}"""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": enhanced_prompt.format(claims=claims)}
    ]
    return chatgpt_chatbot(messages)

def run_openai_generation(claims, method):
    if method == 'full':
        return generate_full_desc(claims)
    if method == 'chunked':
        return generate_chunked_desc(claims)
    raise ValueError(f"Method '{method}' not supported. Use 'full' or 'chunked'.")

# -----------------------------------------------------------
#  CLI entry-point
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate patent descriptions using OpenAI API')
    parser.add_argument('--claims', default='../claims/claims_doc.txt',
                        help='Path to claims file (default: ../claims/claims_doc.txt)')
    parser.add_argument('--method', choices=['full', 'chunked'], default='chunked',
                        help='Generation method (default: chunked)')
    parser.add_argument('--output', default=None,
                        help='Output TXT filename (default: <method>_description_openai.txt)')
    parser.add_argument('--docx', action='store_true',
                        help='Also export the description as .docx')
    args = parser.parse_args()

    # Resolve paths
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    claims_path = os.path.join(script_dir, args.claims)

    # Load claims
    try:
        with open(claims_path, 'r', encoding='utf-8') as f:
            claims = f.read()
        print(f"Loaded claims from: {claims_path}")
    except Exception as e:
        print(f"Error reading claims file: {e}")
        return

    # Generate description
    try:
        print(f"Generating description with '{args.method}' method …")
        description = run_openai_generation(claims, args.method)

        # TXT output path (same folder as script)
        output_txt = os.path.join(
            script_dir,
            args.output or f'{args.method}_description_openai.txt'
        )
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(description)
        print(f"TXT saved to '{output_txt}'")

        # Optional DOCX export
        if args.docx:
            output_docx = os.path.splitext(output_txt)[0] + '.docx'
            txt_to_docx(output_txt, output_docx)

    except Exception as e:
        print(f"Error generating description: {e}")

if __name__ == "__main__":
    main()
