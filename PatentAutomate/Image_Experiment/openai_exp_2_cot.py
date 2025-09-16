import os
import sys
import base64
import argparse
import logging
import json
import time
from typing import Dict, Optional, Tuple, List

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
    "temperature_reasoning": 0.15,  # Added for reasoning steps
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

# Chain-of-Thought Reasoning Prompts
INITIAL_OBSERVATION_PROMPT = """You are analyzing a patent figure. First, make initial observations WITHOUT trying to identify specific components yet.

OBSERVE AND DESCRIBE:
1. Overall shape and structure of the device/system
2. Number of distinct regions or sections visible
3. Types of lines used (solid, dashed, cross-hatching patterns)
4. Presence of any symmetry or repeated patterns
5. Relative sizes and proportions of different areas
6. Any visible connections, boundaries, or interfaces
7. Count of all visible reference numbers/labels

DO NOT identify what components are yet - just describe what you SEE visually.
Be thorough but objective in your observations."""

STRUCTURAL_ANALYSIS_PROMPT = """Based on the initial observations, now analyze the STRUCTURAL RELATIONSHIPS in the figure.

ANALYZE:
1. Hierarchical relationships - what appears to contain or support what?
2. Connection patterns - how do different parts connect or interface?
3. Layering - are there multiple layers or levels visible?
4. Flow patterns - any indication of material/signal/force flow?
5. Groupings - which numbered elements seem to belong together functionally?
6. Boundary conditions - what defines the edges or limits of the system?

Focus on HOW things relate to each other structurally, not WHAT they are yet."""

PATTERN_RECOGNITION_PROMPT = """Now apply pattern recognition based on the domain context and structural analysis.

IDENTIFY PATTERNS:
1. Common configurations for this domain (e.g., chip-on-substrate, housing-with-internals)
2. Standard component arrangements you recognize
3. Typical numbering conventions (e.g., 10x series for one subsystem, 20x for another)
4. Material indications from cross-hatching or shading patterns
5. Size relationships that suggest component types
6. Symmetries that indicate specific design approaches

Connect these patterns to likely component categories for this technical domain."""

CONTEXTUAL_REASONING_PROMPT = """Using the claims context and all previous analysis, reason about the likely identity of each component.

REASONING PROCESS:
1. Match claim language to visible structures
2. Consider functional requirements implied by claims
3. Apply domain-specific knowledge about typical implementations
4. Use spatial logic - what must be present for the claimed invention to work?
5. Consider manufacturing/assembly constraints
6. Identify any innovative or non-standard elements

For each numbered element, provide your reasoning chain:
- What you observe about it
- How it relates to other elements
- Why you think it serves a particular function
- Your confidence level in the identification"""

COMPONENT_SYNTHESIS_PROMPT = """Based on all the reasoning steps above, now synthesize a precise component list.

For each numbered element visible in the figure, provide:
[Number]: [Technical Term] - [Brief functional description]

Requirements:
- Use domain-appropriate technical terminology
- Be specific rather than generic
- Include all visible reference numbers
- Maintain consistency with the reasoning developed
- Flag any uncertain identifications with (probable) or (possible)

Order the list numerically as they appear in the figure."""

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

def perform_chain_of_thought_analysis(image_path: str, claims_text: str, domain_context: str) -> Dict[str, str]:
    """Perform multi-step chain-of-thought reasoning for component identification."""
    logger.info("Step 2: Beginning chain-of-thought analysis...")
    
    cot_results = {}
    image_base64 = encode_image_base64(image_path)
    
    # Step 2a: Initial Observation
    logger.info("Step 2a: Making initial observations...")
    try:
        messages = [
            {"role": "system", "content": INITIAL_OBSERVATION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Domain Context: {domain_context}\n\nObserve this patent figure:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        cot_results["initial_observations"] = make_api_call_with_retry(messages, CONFIG["temperature_reasoning"])
        logger.info("Initial observations completed")
    except Exception as e:
        logger.error(f"Error in initial observation: {str(e)}")
        cot_results["initial_observations"] = "Error in observation step"
    
    # Step 2b: Structural Analysis
    logger.info("Step 2b: Analyzing structural relationships...")
    try:
        messages = [
            {"role": "system", "content": STRUCTURAL_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Previous observations:\n{cot_results['initial_observations']}\n\nAnalyze the structural relationships:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        cot_results["structural_analysis"] = make_api_call_with_retry(messages, CONFIG["temperature_reasoning"])
        logger.info("Structural analysis completed")
    except Exception as e:
        logger.error(f"Error in structural analysis: {str(e)}")
        cot_results["structural_analysis"] = "Error in structural analysis step"
    
    # Step 2c: Pattern Recognition
    logger.info("Step 2c: Recognizing domain-specific patterns...")
    try:
        messages = [
            {"role": "system", "content": PATTERN_RECOGNITION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Domain: {domain_context}
                    
Observations: {cot_results['initial_observations']}

Structural Analysis: {cot_results['structural_analysis']}

Identify patterns:"""},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        cot_results["pattern_recognition"] = make_api_call_with_retry(messages, CONFIG["temperature_reasoning"])
        logger.info("Pattern recognition completed")
    except Exception as e:
        logger.error(f"Error in pattern recognition: {str(e)}")
        cot_results["pattern_recognition"] = "Error in pattern recognition step"
    
    # Step 2d: Contextual Reasoning with Claims
    logger.info("Step 2d: Applying contextual reasoning with claims...")
    try:
        messages = [
            {"role": "system", "content": CONTEXTUAL_REASONING_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""CLAIMS:
{claims_text}

ANALYSIS SO FAR:
Domain: {domain_context}
Patterns: {cot_results['pattern_recognition']}
Structure: {cot_results['structural_analysis']}

Apply reasoning to identify each component:"""},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        cot_results["contextual_reasoning"] = make_api_call_with_retry(messages, CONFIG["temperature_reasoning"])
        logger.info("Contextual reasoning completed")
    except Exception as e:
        logger.error(f"Error in contextual reasoning: {str(e)}")
        cot_results["contextual_reasoning"] = "Error in contextual reasoning step"
    
    # Step 2e: Component Synthesis
    logger.info("Step 2e: Synthesizing final component list...")
    try:
        messages = [
            {"role": "system", "content": COMPONENT_SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Based on all reasoning:

INITIAL OBSERVATIONS:
{cot_results['initial_observations']}

STRUCTURAL ANALYSIS:
{cot_results['structural_analysis']}

PATTERN RECOGNITION:
{cot_results['pattern_recognition']}

CONTEXTUAL REASONING:
{cot_results['contextual_reasoning']}

Synthesize the final component list:"""},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }}
                ]
            }
        ]
        cot_results["component_synthesis"] = make_api_call_with_retry(messages, CONFIG["temperature_analysis"])
        logger.info("Component synthesis completed")
    except Exception as e:
        logger.error(f"Error in component synthesis: {str(e)}")
        cot_results["component_synthesis"] = "Error in synthesis step"
    
    return cot_results

def generate_component_list(claims_text: str, image_path: str, domain_context: str) -> str:
    """Generate component list with enhanced domain awareness."""
    logger.info("Step 3: Generating component list with domain context...")
    
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

def validate_component_list(component_list_text: str, domain_context: str, cot_reasoning: Dict[str, str]) -> str:
    """Validate component list for technical accuracy and consistency."""
    logger.info("Step 4: Validating component accuracy...")
    
    try:
        validation_prompt = f"""
DOMAIN CONTEXT: {domain_context}

REASONING CHAIN SUMMARY:
- Pattern Recognition: {cot_reasoning.get('pattern_recognition', 'N/A')[:500]}...
- Contextual Analysis: {cot_reasoning.get('contextual_reasoning', 'N/A')[:500]}...

COMPONENT LIST TO VALIDATE:
{component_list_text}

Please review this component list for technical accuracy within the given domain context.
Consider the reasoning chain that led to these identifications.
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
    logger.info("Step 4b: Refining component list based on validation...")
    
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
    logger.info("Step 5: Generating comprehensive technical description...")
    
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

def save_analysis_summary(outputs: dict, cot_steps: List[str]) -> None:
    """Save a JSON summary of the entire analysis."""
    try:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_used": CONFIG,
            "files_generated": list(outputs.keys()),
            "analysis_steps": [
                "Domain context detection",
                "Chain-of-thought reasoning:",
            ] + [f"  - {step}" for step in cot_steps] + [
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
        
        logger.info("Starting enhanced patent figure analysis with chain-of-thought reasoning...")
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
        
        # Step 2: Perform chain-of-thought analysis
        cot_results = perform_chain_of_thought_analysis(image_path, claims_text, domain_context)
        
        # Use the synthesized component list from CoT
        component_list_raw = cot_results.get("component_synthesis", "")
        
        # Step 3: Validate component list with CoT context
        validation_results = validate_component_list(component_list_raw, domain_context, cot_results)
        
        # Step 4: Refine component list based on validation
        component_list_final = refine_component_list(component_list_raw, validation_results, domain_context)
        
        # Step 5: Generate final description
        final_description = generate_final_description(claims_text, image_path, component_list_final, domain_context)

        # Save all outputs
        outputs = {
            "01_domain_context.txt": domain_context,
            "02a_initial_observations.txt": cot_results.get("initial_observations", ""),
            "02b_structural_analysis.txt": cot_results.get("structural_analysis", ""),
            "02c_pattern_recognition.txt": cot_results.get("pattern_recognition", ""),
            "02d_contextual_reasoning.txt": cot_results.get("contextual_reasoning", ""),
            "02e_component_synthesis.txt": cot_results.get("component_synthesis", ""),
            "03_validation_results.txt": validation_results,
            "04_component_list_final.txt": component_list_final,
            "05_final_description.txt": final_description
        }
        
        for filename, content in outputs.items():
            save_output(filename, content)
        
        # Save analysis summary with CoT steps
        cot_steps = [
            "Initial observations",
            "Structural analysis",
            "Pattern recognition",
            "Contextual reasoning",
            "Component synthesis"
        ]
        save_analysis_summary(outputs, cot_steps)
        
        logger.info("Enhanced patent figure analysis with chain-of-thought completed successfully!")
        logger.info("Generated files:")
        for filename in outputs.keys():
            logger.info(f"  - {filename}")
        logger.info("  - analysis_summary.json")
        logger.info("  - patent_analysis.log")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Patent Figure Analysis with Chain-of-Thought")
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