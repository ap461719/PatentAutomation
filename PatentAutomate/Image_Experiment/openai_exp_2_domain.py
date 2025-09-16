# import os
# import sys
# import base64
# import argparse

# # Add utils/ to the Python path dynamically
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

# from utils import client

# # 🔧 System prompt focused on extracting labeled parts ONLY --> doesn't matter the formatting, do not restrict it. Incorporate some chain of thought. 
# SYSTEM_PROMPT = """You are a specialized assistant for analyzing patent figures. Your task is to extract labeled components from the diagram based on both the figure and the claims.

# Your response must:
# - Output a numbered list of the labeled components from the figure (e.g., 102, 104, etc.).
# - Write a description for each component 
# """

# def encode_image_base64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def generate_component_list(claims_text, image_path):
#     image_base64 = encode_image_base64(image_path)

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": (
#                             "List all labeled components visible in the image below. "
#                             f"{claims_text}"
#                         )
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{image_base64}",
#                             "detail": "high"
#                         }
#                     }
#                 ]
#             }
#         ],
#         temperature=0.1
#     )

#     return response.choices[0].message.content.strip()

# def generate_final_description(claims_text, image_path, component_list_text):
#     image_base64 = encode_image_base64(image_path)

#     FINAL_DESCRIPTION_PROMPT = """You are a specialized patent drafting assistant. Your job is to write a detailed technical description of a patent figure using the provided claims and the list of identified components.

# Guidelines:
# - Use formal patent language (e.g., “The apparatus includes…”, “The system comprises…”).
# - Refer to labeled elements by their numbers (e.g., “element 102”, “member 104”).
# - Use the component list to guide your terminology and naming.
# - Do not copy claim language. Instead, explain how components work or interact.
# - Focus on structure, configuration, operation, and technical details.
# - Do NOT include any markdown or bullet points — write flowing paragraphs only.
# - The style should match that of a patent detailed description section.

# Now generate the technical description using the image, claims, and labeled component list below.
# """

#     user_prompt = (
#         f"{FINAL_DESCRIPTION_PROMPT}\n\n"
#         f"--- CLAIMS ---\n{claims_text}\n\n"
#         f"--- COMPONENT LIST ---\n{component_list_text}\n\n"
#         f"Write the description below:"
#     )

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": FINAL_DESCRIPTION_PROMPT},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_prompt},
#                     {"type": "image_url", "image_url": {
#                         "url": f"data:image/png;base64,{image_base64}",
#                         "detail": "high"
#                     }}
#                 ]
#             }
#         ],
#         temperature=0.2
#     )

#     return response.choices[0].message.content.strip()

# def main():
#     claims_path = "../claims/claims_image.txt"
#     image_path = "./Patent_fig_1.png"
#     component_list_path = "image_component_list_generated.txt"
#     component_output_path = "image_component_list_generated.txt"
#     final_output_path = "multimodal_description_exp2.txt"

#     # Step 1: Read claims
#     with open(claims_path, "r", encoding="utf-8") as f:
#         claims_text = f.read()

#     # Step 2: Generate list of components
#     component_list = generate_component_list(claims_text, image_path)
#     with open(component_output_path, "w", encoding="utf-8") as f:
#         f.write(component_list)
#     print(f"\nComponent list saved to: {component_output_path}")

#     # Step 3: Generate full technical description using component list + image + claims
#     full_description = generate_final_description(claims_text, image_path, component_list)
#     with open(final_output_path, "w", encoding="utf-8") as f:
#         f.write(full_description)
#     print(f"\nPatent description saved to: {final_output_path}")

# if __name__ == "__main__":
#     main()




import os
import sys
import base64
import argparse
import logging
import json
import time
from typing import Dict, Optional, Tuple

# Add utils/ to the Python path dynamically
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

from utils import client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('patent_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model": "gpt-4o",
    "temperature_analysis": 0.1,
    "temperature_validation": 0.1,
    "temperature_description": 0.2,
    "api_delay": 1.0,  # seconds between API calls
    "max_retries": 3
}

# Enhanced system prompts
DOMAIN_DETECTION_PROMPT = """You are a technical domain classifier for patent figures. Analyze this patent figure and identify:

1. The technical domain (semiconductor, mechanical, electrical, chemical, optical, etc.)
2. The main device/system type (e.g., semiconductor package, circuit board, mechanical assembly)
3. Key structural patterns and characteristics you observe
4. Level of complexity (simple/moderate/complex)

Provide a brief 2-3 sentence technical context that will help with component identification."""

COMPONENT_ANALYSIS_PROMPT = """You are a specialized assistant for analyzing patent figures with expertise in technical terminology across multiple domains.

ANALYSIS PROCESS:
1. First, consider the overall device type and technical domain provided
2. Examine each numbered element systematically 
3. Consider spatial relationships and hierarchical structure
4. Use domain-appropriate technical terminology
5. Think step-by-step about each component's likely function

OUTPUT FORMAT:
- Numbered list matching exactly what's visible in the figure (e.g., 1, 2, 3a, 3b, etc.)
- For each: [Number]: [Precise Technical Description] - [Function/Purpose if determinable]

TECHNICAL GUIDELINES:
- Semiconductor domain: use terms like die, package, bond pad, trace, via, substrate, lead frame
- Mechanical domain: use terms like housing, bearing, shaft, coupling, fastener
- Electrical domain: use terms like conductor, insulator, terminal, contact, lead
- Be specific about materials, structures, and spatial relationships
- Avoid generic terms when specific technical terms exist"""

VALIDATION_PROMPT = """You are a technical validation expert for patent component analysis. 

Your task is to review component identifications for:
1. Technical terminology accuracy for the given domain
2. Logical consistency between related components
3. Completeness (any obvious missing elements?)
4. Spatial relationship accuracy

For each component, provide:
- Confidence score (1-10, where 10 = highly confident)
- Brief reasoning for score
- Alternative identification if confidence < 7
- Overall domain consistency assessment

Be critical but constructive in your evaluation."""

FINAL_DESCRIPTION_PROMPT = """You are a specialized patent drafting assistant. Write a detailed technical description of this patent figure using formal patent language.

REQUIREMENTS:
- Use formal patent terminology ("The apparatus comprises...", "The system includes...")
- Reference labeled elements by their numbers consistently
- Focus on structure, configuration, and technical relationships
- Use the validated component list as your terminology guide
- Write in flowing paragraphs (NO bullet points or markdown)
- Include technical details about materials, connections, and spatial arrangements
- Describe how components interact or relate to each other

STYLE: Match the detailed description section of a professional patent application."""

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 with error handling."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        raise

def make_api_call_with_retry(messages: list, temperature: float = 0.1, max_retries: int = 3) -> str:
    """Make API call with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = CONFIG["api_delay"] * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying API call in {wait_time} seconds...")
                time.sleep(wait_time)
            
            response = client.chat.completions.create(
                model=CONFIG["model"],
                messages=messages,
                temperature=temperature
            )
            
            # Rate limiting
            time.sleep(CONFIG["api_delay"])
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All API call attempts failed")
                raise
    
    return ""

def detect_domain_context(image_path: str) -> str:
    """Pre-analyze image to determine technical domain and context."""
    logger.info("Step 1: Analyzing domain context...")
    
    try:
        image_base64 = encode_image_base64(image_path)
        
        messages = [
            {"role": "system", "content": DOMAIN_DETECTION_PROMPT},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze this patent figure for technical domain and device type:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        
        result = make_api_call_with_retry(messages, CONFIG["temperature_analysis"])
        logger.info(f"Domain context identified: {result[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in domain detection: {str(e)}")
        return "Unable to determine domain context - proceeding with general analysis"

def generate_component_list(claims_text: str, image_path: str, domain_context: str) -> str:
    """Generate component list with enhanced domain awareness."""
    logger.info("Step 2: Generating component list with domain context...")
    
    try:
        image_base64 = encode_image_base64(image_path)
        
        enhanced_prompt = f"""
DOMAIN CONTEXT: {domain_context}

CLAIMS CONTEXT:
{claims_text}

INSTRUCTIONS:
Using the domain context above, analyze this patent figure and identify all labeled components.
Think through each numbered element systematically, considering:
- The technical domain and device type
- Spatial relationships between components
- Appropriate technical terminology for this domain
- Any context provided in the claims

Provide a numbered list of all visible labeled components with precise technical descriptions.
"""

        messages = [
            {"role": "system", "content": COMPONENT_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        
        result = make_api_call_with_retry(messages, CONFIG["temperature_analysis"])
        logger.info("Component list generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in component list generation: {str(e)}")
        raise

def validate_component_list(component_list_text: str, domain_context: str) -> str:
    """Validate component list for technical accuracy and consistency."""
    logger.info("Step 3: Validating component accuracy...")
    
    try:
        validation_prompt = f"""
DOMAIN CONTEXT: {domain_context}

COMPONENT LIST TO VALIDATE:
{component_list_text}

Please review this component list for technical accuracy within the given domain context.
Evaluate each component identification and provide constructive feedback.
"""

        messages = [
            {"role": "system", "content": VALIDATION_PROMPT},
            {"role": "user", "content": validation_prompt}
        ]
        
        result = make_api_call_with_retry(messages, CONFIG["temperature_validation"])
        logger.info("Component validation completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in component validation: {str(e)}")
        return "Validation failed - proceeding with unvalidated component list"

def refine_component_list(original_list: str, validation_results: str, domain_context: str) -> str:
    """Refine component list based on validation feedback."""
    logger.info("Step 3b: Refining component list based on validation...")
    
    try:
        refinement_prompt = f"""
DOMAIN CONTEXT: {domain_context}

ORIGINAL COMPONENT LIST:
{original_list}

VALIDATION FEEDBACK:
{validation_results}

Based on the validation feedback, provide a refined and corrected component list.
Keep the same format but improve technical accuracy where needed.
Only make changes where the validation clearly indicates improvements.
"""

        messages = [
            {"role": "system", "content": "You are refining a component list based on expert validation feedback."},
            {"role": "user", "content": refinement_prompt}
        ]
        
        result = make_api_call_with_retry(messages, CONFIG["temperature_analysis"])
        logger.info("Component list refinement completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in component refinement: {str(e)}")
        logger.info("Using original component list")
        return original_list

def generate_final_description(claims_text: str, image_path: str, component_list: str, domain_context: str) -> str:
    """Generate final technical description using all context."""
    logger.info("Step 4: Generating comprehensive technical description...")
    
    try:
        image_base64 = encode_image_base64(image_path)

        user_prompt = f"""
DOMAIN CONTEXT: {domain_context}

CLAIMS:
{claims_text}

VALIDATED COMPONENT LIST:
{component_list}

Using all the context above, write a detailed technical description of this patent figure.
Focus on the structure, configuration, and technical relationships between components.
Use formal patent language and reference components by their numbers consistently.
"""

        messages = [
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
        ]
        
        result = make_api_call_with_retry(messages, CONFIG["temperature_description"])
        logger.info("Final description generated successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in final description generation: {str(e)}")
        raise

def save_output(filename: str, content: str) -> None:
    """Save content to file with error handling."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving {filename}: {str(e)}")

def save_analysis_summary(outputs: dict) -> None:
    """Save a JSON summary of the entire analysis."""
    try:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_used": CONFIG,
            "files_generated": list(outputs.keys()),
            "analysis_steps": [
                "Domain context detection",
                "Component list generation", 
                "Component validation",
                "Component refinement",
                "Final description generation"
            ]
        }
        
        with open("analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Analysis summary saved")
        
    except Exception as e:
        logger.error(f"Error saving analysis summary: {str(e)}")

def main():
    """Main execution function with comprehensive error handling."""
    try:
        # File paths
        claims_path = "../claims/claims_image.txt"
        image_path = "./Temp_Patent_fig_1.png"
        
        logger.info("Starting enhanced patent figure analysis...")
        logger.info(f"Claims file: {claims_path}")
        logger.info(f"Image file: {image_path}")
        
        # Verify files exist
        if not os.path.exists(claims_path):
            logger.error(f"Claims file not found: {claims_path}")
            return
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return

        # Read claims
        with open(claims_path, "r", encoding="utf-8") as f:
            claims_text = f.read()
        logger.info(f"Claims loaded: {len(claims_text)} characters")

        # Step 1: Detect domain context
        domain_context = detect_domain_context(image_path)
        
        # Step 2: Generate component list with domain context
        component_list_raw = generate_component_list(claims_text, image_path, domain_context)
        
        # Step 3: Validate component list
        validation_results = validate_component_list(component_list_raw, domain_context)
        
        # Step 3b: Refine component list based on validation
        component_list_final = refine_component_list(component_list_raw, validation_results, domain_context)
        
        # Step 4: Generate final description
        final_description = generate_final_description(claims_text, image_path, component_list_final, domain_context)

        # Save all outputs
        outputs = {
            "01_domain_context.txt": domain_context,
            "02_component_list_raw.txt": component_list_raw,
            "03_validation_results.txt": validation_results,
            "04_component_list_final.txt": component_list_final,
            "05_final_description.txt": final_description
        }
        
        for filename, content in outputs.items():
            save_output(filename, content)
        
        # Save analysis summary
        save_analysis_summary(outputs)
        
        logger.info("Enhanced patent figure analysis completed successfully!")
        logger.info("Generated files:")
        for filename in outputs.keys():
            logger.info(f"  - {filename}")
        logger.info("  - analysis_summary.json")
        logger.info("  - patent_analysis.log")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Patent Figure Analysis")
    parser.add_argument("--claims", help="Path to claims file", default="../claims/claims_image.txt")
    parser.add_argument("--image", help="Path to patent figure", default="./Temp_Patent_fig_1.png")
    parser.add_argument("--model", help="OpenAI model to use", default="gpt-4o")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.model != "gpt-4o":
        CONFIG["model"] = args.model
    
    main()
