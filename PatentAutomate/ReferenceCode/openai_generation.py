import re
import argparse
from utils.utils import chatgpt_chatbot, EMBODIMENT_PROMPT_IN_CONTEXT

INIT_INSTRUCTION = 'You are an assistant designed to generate the detailed description of a patent application. Make sure the description you write is written in a legal language style. Do not include any markdown formatting in the description. Begin by writing the detailed description for this claim:\n{claim}'
CONTINUE_INSTRUCTION = 'Continue by writing the detailed description for this claim, do not include any markdown formatting in the description:\n{claim}'

def split_claims(claims):
    pattern = re.compile(r'^\d{1,3}\.')
    split_claims_newline = claims.split('\n')
    final_split_claims = []
    full_line = ''
    for line in reversed(split_claims_newline):
        if bool(pattern.match(line.strip())):
            line_to_add = line + "\n" + full_line[:] if full_line != '' else line
            final_split_claims.insert(0, line_to_add)
            full_line = ''
        else:
            full_line = line + full_line
    return final_split_claims

def generate_chunked_desc(claims):
    claims_split = split_claims(claims)
    full_generation = ""
    messages = [{"role": "user", "content": INIT_INSTRUCTION.format(claim=claims_split[0])}]
    for claim in claims_split[1:]:
        generation = chatgpt_chatbot(messages)
        # generation = ollama_chatbot(messages, '')
        full_generation += generation + '\n'
        messages.append({'role': 'assistant', 'content': generation})
        messages.append({'role': 'user', 'content': CONTINUE_INSTRUCTION.format(claim=claim)})
    return full_generation

def generate_full_desc(claims):
    description = chatgpt_chatbot([{'role': 'user', 'content': EMBODIMENT_PROMPT_IN_CONTEXT.format(claims=claims)}])
    return description

def run_openai_generation(claims, method):
    if method == 'full':
        return generate_full_desc(claims)
    elif method == 'chunked':
        return generate_chunked_desc(claims)
    else:
        print("method not supported")
        return ''
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims', type=str, default='./claims.txt', help='path to claims')
    args = parser.parse_args()

    with open(args.claims) as f:
        claims = f.read()

    # one_shot_desc = run_openai_generation(claims, 'full')
    chunked_desc = run_openai_generation(claims, 'chunked')

    # with open('openai_full_sample.txt', 'w') as f:
    #     f.write(one_shot_desc)
    
    with open('chunked_sample.txt', 'w') as f:
        f.write(chunked_desc)