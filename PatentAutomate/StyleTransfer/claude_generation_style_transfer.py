import re
import argparse
from utils.utils_claude import chatgpt_chatbot, EMBODIMENT_PROMPT_IN_CONTEXT

# System message - sets the AI's role and behavior
SYSTEM_MESSAGE = """You are a specialized patent attorney assistant designed to generate detailed descriptions of patent applications. 

Your expertise includes:
- Writing in precise legal language style appropriate for patent applications
- Understanding patent claim structures and dependencies
- Generating comprehensive technical descriptions that support patent claims
- Avoiding repetitive language while maintaining legal precision

Critical Guidelines:
- NEVER explicitly reference claim numbers or use phrases like "as described in claim X" or "this claim specifies"
- Use claims only as context to understand what technical elements to describe
- Focus on technical implementation details rather than restating claim language
- Do not include redundant explanations of the same concepts
- Use formal, technical language appropriate for patent documentation
- Do not include any markdown formatting in your responses
- Ensure descriptions are detailed enough to support the patent claims without directly referencing them
- Maintain consistency in terminology throughout the description"""

# Good and bad examples for style transfer
GOOD_EXAMPLE = """GOOD EXAMPLE of patent description writing:

The MRF system 300 includes a control system 302, which manages, commands, directs, and/or regulates actions and/or behaviors of various components, devices, and/or systems of an MRF using, for example, one or more control loops or other like mechanisms. In particular, the control system 302 receives data from a variety of sources and uses these inputs to control various components, devices, and/or systems of the MRF. The various sources can include a set of sensors 321-1 to 321-N (where N is a number), a set of material handling units (MHUs) 322-1 to 322-M (where M is a number), and/or one or more AI/ML systems 312.

The control system 302 receives inputs (e.g., data streams 331, 332) from some or all components of the MRF and provides autonomous control of the MRF based on those inputs. The control system 302 is embodied as one or more computer devices and/or software that runs on the one or more computer devices to carry out, operate, or execute the techniques disclosed herein."""

BAD_EXAMPLE = """BAD EXAMPLE to avoid:

The present invention relates to a control system for MRFs. As described in claim 1, the control system manages components. The invention further includes sensors as specified in the claims. This claim teaches that data streams are received. The present invention also relates to autonomous control as described in claim 2. The control system of the present invention comprises computer devices as claimed."""

STYLE_TRANSFER_SECTION = f"""
WRITING STYLE GUIDELINES:

{GOOD_EXAMPLE}

{BAD_EXAMPLE}

Notice how the good example:
- Uses varied sentence structures and technical detail
- Avoids repetitive phrases like "The present invention relates to..."
- Never references claim numbers directly
- Focuses on technical implementation rather than restating claims
- Uses specific component numbers and technical terminology
- Provides substantive technical information

The bad example shows what to avoid:
- Repetitive sentence openings
- Direct claim references ("as described in claim X")
- Vague, non-technical language
- Simply restating claim language without added detail

Follow the style of the GOOD EXAMPLE in your response.
"""

# User messages - specific instructions for each interaction
INIT_INSTRUCTION = f'''Generate a technical description focusing on molecular mechanisms, structural design rationale, and technical implementation. Use varied sentence openings and avoid repetitive language patterns:

{STYLE_TRANSFER_SECTION}

Claims to describe:
{{claim}}'''

CONTINUE_INSTRUCTION = f'''Continue with additional technical details using completely different sentence structures and terminology. Focus on new technical aspects without repeating previous explanations:

{STYLE_TRANSFER_SECTION}

Additional claims:
{{claim}}'''



def split_claims(claims):
    """Split claims text into individual claims, handling multi-line claims properly."""
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
    """Generate description by processing claims one by one in a conversation."""
    claims_split = split_claims(claims)
    full_generation = ""
    
    # Initialize with system message
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": INIT_INSTRUCTION.format(claim=claims_split[0])}
    ]
    
    # Process first claim
    generation = chatgpt_chatbot(messages)
    full_generation += generation + '\n'
    messages.append({'role': 'assistant', 'content': generation})
    
    # Process remaining claims
    for claim in claims_split[1:]:
        messages.append({'role': 'user', 'content': CONTINUE_INSTRUCTION.format(claim=claim)})
        generation = chatgpt_chatbot(messages)
        full_generation += generation + '\n'
        messages.append({'role': 'assistant', 'content': generation})
    
    return full_generation

def generate_full_desc(claims):
    """Generate complete description in one API call."""
    enhanced_prompt = f"""You are a specialized patent attorney assistant. Generate a detailed description of a patent application based on the provided claims. 

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

{STYLE_TRANSFER_SECTION}

Given the claims below, generate a detailed technical description:

{{claims}}"""
    
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": enhanced_prompt.format(claims=claims)}
    ]
    description = chatgpt_chatbot(messages)
    return description

def run_claude_generation(claims, method):
    """Run patent description generation using specified method."""
    if method == 'full':
        return generate_full_desc(claims)
    elif method == 'chunked':
        return generate_chunked_desc(claims)
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'full' or 'chunked'.")

def main():
    """Main function to handle command line arguments and file operations."""
    parser = argparse.ArgumentParser(description='Generate patent descriptions using Claude API with style transfer')
    parser.add_argument('--claims', type=str, default='./claims.txt', 
                       help='Path to claims file (default: ./claims.txt)')
    parser.add_argument('--method', type=str, choices=['full', 'chunked'], default='chunked',
                       help='Generation method: full or chunked (default: chunked)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (default: auto-generated based on method)')
    
    args = parser.parse_args()
    
    # Read claims file
    try:
        with open(args.claims, 'r', encoding='utf-8') as f:
            claims = f.read()
    except FileNotFoundError:
        print(f"Error: Claims file '{args.claims}' not found.")
        return
    except Exception as e:
        print(f"Error reading claims file: {e}")
        return
    
    # Generate description
    try:
        print(f"Generating description using Claude API with {args.method} method and built-in MRF style examples...")
        description = run_claude_generation(claims, args.method)
        
        # Determine output filename
        if args.output:
            output_file = args.output
        else:
            output_file = f'{args.method}_description_claude.txt'
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(description)
        
        print(f"Description generated successfully and saved to '{output_file}'")
        
    except Exception as e:
        print(f"Error generating description: {e}")

if __name__ == "__main__":
    main()