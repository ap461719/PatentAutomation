## Overview
**PatentAutomate** is a research-driven framework for automating patent drafting using Large Language Models (LLMs).  
It transforms raw patent claims and figures into structured, legally styled descriptions by integrating:  

- **Multimodal reasoning** (claims text + figure analysis)  
- **Contextual chain-of-thought expansion** (observations, synthesis, validation)  
- **Style transfer pipelines** for polished, legal-compliant prose  
- **Cross-model comparison** between OpenAI GPT-4o and Anthropic Claude  

This project demonstrates how GenAI can accelerate intellectual property workflows while maintaining clarity, accuracy, and compliance with patent documentation standards.  

---

## Features
- **Component recognition**: extract and validate entities from claims and figures  
- **Reasoning pipeline**: domain context, pattern recognition, synthesis, and validation  
- **Multimodal fusion**: combine bounding box detection with text-driven analysis  
- **Style transfer**: refine raw AI generations into professional patent prose  
- **Evaluation**: compare outputs across models and reasoning steps  

---

## Getting Started

### Prerequisites
- Python 3.10+
- Install dependencies:

pip install -r requirements.txt

API keys required:
OpenAI API
Claude (Anthropic)

- **Setup**
Clone the repo and move into the project directory:
https://github.com/ap461719/Patent.git
cd PatentAutomate

- **Example Workflow**
Place patent claims into the claims/ folder and figures into Image_Experiment/.
Run preprocessing to extract raw component lists.
Expand into contextual observations, analyses, and synthesis.
Use generation scripts (OpenAI or Claude) to create draft descriptions.
Apply style transfer for final patent-style prose.


- **Research Significance**
PatentAutomate combines structured reasoning, multimodal grounding, and style transfer to reduce manual drafting time by automating critical stages of the patent writing process.
The framework highlights how GenAI can be used for specialized technical writing tasks, balancing automation with interpretability and compliance