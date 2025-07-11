import requests
import json
import os

# Claude API configuration
CLAUDE_API_KEY = "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"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

def chatgpt_chatbot(messages, max_tokens=4000):
    """
    Send messages to Claude API and return the response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens in response
    
    Returns:
        String response from Claude
    """
    
    # Convert OpenAI format to Claude format
    claude_messages = []
    system_message = None
    
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        else:
            claude_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    # Prepare the request payload
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": claude_messages
    }
    
    # Add system message if present
    if system_message:
        payload["system"] = system_message
    
    # Prepare headers
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        
        # Extract the content from Claude's response format
        if 'content' in response_data and len(response_data['content']) > 0:
            return response_data['content'][0]['text']
        else:
            raise Exception("Unexpected response format from Claude API")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {e}")
    except Exception as e:
        raise Exception(f"Error calling Claude API: {e}")

# Legacy variable for compatibility (if used elsewhere in your code)
EMBODIMENT_PROMPT_IN_CONTEXT = ""