import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

def chatgpt_chatbot(messages, model="gpt-4o"):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.1,
    )
    answer = completion.choices[0].message.content.strip()
    return answer

# Updated prompt that avoids claim references and repetitive language
EMBODIMENT_PROMPT_IN_CONTEXT = """You are a specialized patent attorney assistant. Generate a detailed technical description of a patent application based on the provided claims.

CRITICAL REQUIREMENTS:
- NEVER explicitly reference claim numbers or use phrases like "as described in claim X", "this claim specifies", "claim 1", etc.
- Use claims only as context to understand what technical elements to describe
- Avoid repetitive phrases like "The present invention relates to..." or "The present invention further..."
- Vary sentence structure and openings throughout the description
- Focus on technical implementation details rather than restating claim language
- Do not include redundant explanations of the same concepts
- Write in formal, technical language appropriate for patent documentation
- Do not include any markdown formatting
- Generate a comprehensive description that supports all the claims without directly referencing them

STYLE GUIDELINES:
- Use varied sentence starters such as: "A novel system comprises...", "The disclosed method involves...", "An isolated compound exhibits...", "The technique encompasses...", etc.
- Focus on technical mechanisms and implementation details
- Describe the structural and functional relationships between components
- Explain how different elements work together to achieve the desired outcome
- Use precise technical terminology consistent throughout
- Avoid redundant explanations of the same technical concepts
- Write in a flowing, coherent manner that builds understanding progressively

EXAMPLE STYLE (note the varied openings and technical focus):

Sample Claims Input:
1. A method for reducing chemotherapy-induced alopecia comprising topically administering a vasoconstrictor...
2. The method of claim 1, wherein the vasoconstrictor comprises epinephrine...

Good Output Style:
"Chemotherapy and radiotherapy treatments for cancer patients are associated with severe side effects due to cytotoxicity affecting normal cells, particularly epithelial populations within hair follicles, skin epidermis, and gastrointestinal mucosa. The disclosed therapeutic approach addresses these limitations through topical administration of vasoconstrictor compounds..."

(Notice: No claim references, varied sentence openings, technical focus)

Now generate a description for the following claims:

Input:
{claims}

Output:
"""