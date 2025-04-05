from typing import Optional, Dict, Any # Added Dict, Any type hints
import google.generativeai as genai
import os
import json
import click
import logging
import textwrap
from dotenv import load_dotenv
import re

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(name)-15s : %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("GeminiDemo")
# Silence overly verbose google client libraries if needed
logging.getLogger("google.api_core").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# --- Global Variable for the Model ---
# Define globally so it can be accessed by query_gemini after initialization
model: Optional[genai.GenerativeModel] = None
SELECTED_MODEL_NAME: str = 'models/gemini-1.5-flash-latest' # Set default model name here

# --- Load API Key and Configure API ---
try:
    load_dotenv() # Load variables from .env file
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        logger.critical("Fatal: GOOGLE_API_KEY not found in environment variables or .env file.")
        logger.critical("Please get an API key from Google AI Studio (https://aistudio.google.com/)")
        logger.critical("and set it as GOOGLE_API_KEY in your environment or a .env file.")
        exit(1) # Exit if key is missing

    genai.configure(api_key=API_KEY)
    logger.info("Gemini API configured with key.")

    # --- Initialize the Model ---
    # Moved initialization here, immediately after configuration
    logger.info(f"Attempting to initialize model: {SELECTED_MODEL_NAME}")
    model = genai.GenerativeModel(SELECTED_MODEL_NAME) # Assign to global 'model'
    logger.info(f"Gemini Model '{SELECTED_MODEL_NAME}' initialized successfully.")

    # --- Optional: List Models (for information/debugging only) ---
    try:
        logger.info("Listing available models...")
        print("="*20 + " Available Models Supporting generateContent " + "="*20)
        model_found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                if m.name == SELECTED_MODEL_NAME:
                    model_found = True
        print("="*70)
        if not model_found:
             logger.warning(f"Currently selected model '{SELECTED_MODEL_NAME}' was not found in the list of models supporting generateContent for your API key/region. API calls might fail.")
        logger.info("Finished listing models.")
    except Exception as list_e:
        logger.error(f"Could not list available models (continuing with selected model): {list_e}")

except Exception as config_e:
    logger.critical(f"Fatal: Failed during Gemini API configuration or model initialization: {config_e}", exc_info=True)
    exit(1)


# --- Prompt Engineering ---
def create_gemini_prompt(features: dict) -> str:
    """Creates the prompt for Gemini, asking for JSON output."""
    # Filter out None values and format features nicely
    feature_lines = []
    key_order = ["timestamp", "source_ip", "destination_ip", "protocol", "source_port", "destination_port", "tcp_flags", "icmp_type", "icmp_code", "dns_query_name", "dns_response_ips", "packet_length", "payload_snippet"]
    present_features = {k: features.get(k) for k in key_order if features.get(k) is not None and features.get(k) != ''}
    for key, value in features.items():
         if key not in present_features and value is not None and value != '':
              present_features[key] = value

    for key, value in present_features.items():
        # Simple formatting for readability
        key_str = key.replace('_', ' ').title()
        value_str = str(value)
        if isinstance(value, list):
             value_str = ", ".join(map(str, value))
        # Truncate very long values for prompt brevity
        if len(value_str) > 150:
            value_str = value_str[:150] + "..."
        feature_lines.append(f"- {key_str}: {value_str}")

    if not feature_lines:
        feature_text = "No specific packet features provided."
    else:
        feature_text = "\n".join(feature_lines)

    # The core prompt instructing the model
    prompt = f"""You are an expert network security analyst. Your task is to analyze the provided network packet features and determine if they indicate a potential security vulnerability, focusing on common threats like SQL Injection, XSS, Buffer Overflow patterns, DoS/DDoS indicators, or Man-in-the-Middle indicators.

Analyze the following features:
{feature_text}

Based *only* on these features, respond with a valid JSON object containing your analysis. The JSON object MUST have the following structure:

{{
  "is_vulnerable": boolean, // true if a potential vulnerability is detected, false otherwise
  "vulnerability_type": "string | null", // Concise name (e.g., "Potential SQL Injection Attempt", "SYN Flood Indicator"). Null if not vulnerable.
  "severity": "string | null", // Estimated severity ("Critical", "High", "Medium", "Low", "Informational"). Null if not vulnerable.
  "description": "string | null", // Brief explanation of why the features suggest this vulnerability. Null if not vulnerable.
  "mitigation_steps": [ string ] | null // List of 2-4 actionable mitigation steps relevant to the detection. Null if not vulnerable.
}}

Respond ONLY with the JSON object described above. Do not include any explanatory text before or after the JSON block. If no clear vulnerability is identified based on the provided features, return:
{{
  "is_vulnerable": false,
  "vulnerability_type": null,
  "severity": null,
  "description": null,
  "mitigation_steps": null
}}

JSON Output:
"""
    logger.debug(f"Generated Gemini Prompt (truncated):\n{prompt[:500]}...")
    return prompt

# --- JSON Extraction ---
def extract_json_output(response_text: str) -> Optional[dict]:
    """Attempts to extract the JSON object from Gemini's response text."""
    logger.debug(f"Attempting to extract JSON from response (first 500 chars): {response_text[:500]}")
    match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_string = next((g for g in match.groups() if g), None)
        if json_string:
            logger.debug(f"Potential JSON string found: {json_string}")
            try:
                parsed_json = json.loads(json_string)
                if "is_vulnerable" not in parsed_json or not isinstance(parsed_json["is_vulnerable"], bool):
                     logger.warning(f"Parsed JSON missing or invalid 'is_vulnerable' key: {parsed_json}")
                     return None
                if parsed_json["is_vulnerable"]:
                     required_if_vuln = ["vulnerability_type", "severity", "description", "mitigation_steps"]
                     if not all(k in parsed_json for k in required_if_vuln): logger.warning(f"Parsed JSON marked vulnerable but missing expected keys: {parsed_json}")
                     mitigation = parsed_json.get("mitigation_steps")
                     if mitigation is not None and not isinstance(mitigation, list):
                           logger.warning(f"Mitigation steps is not a list or null: {mitigation}")
                           if isinstance(mitigation, str): parsed_json["mitigation_steps"] = [mitigation]
                           else: parsed_json["mitigation_steps"] = ["Error: Invalid mitigation format received."]
                elif len(parsed_json) > 1:
                     logger.warning(f"Parsed JSON marked not vulnerable but has extra keys: {parsed_json}")
                     return {"is_vulnerable": False}
                logger.info("JSON parsed and validated successfully.")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from extracted string: {e}\nString was: {json_string}")
                return None
        else: logger.error("Regex matched, but no JSON content found in groups."); return None
    else: logger.error(f"Could not find JSON block in response: {response_text[:500]}..."); return None


# --- Gemini API Call ---
def query_gemini(prompt: str) -> Optional[dict]:
    """Sends prompt to Gemini and attempts to parse JSON response."""
    global model # Explicitly use the global model variable initialized earlier
    if model is None:
         logger.error("Gemini model was not initialized successfully. Cannot query API.")
         return {"is_vulnerable": False, "error": "Gemini model not initialized."}

    logger.info("Querying Gemini API...")
    try:
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # Be slightly more permissive for dangerous content in security context, but still block none is risky
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        generation_config = genai.types.GenerationConfig(
             temperature=0.1,
             max_output_tokens=1024
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if not response.candidates:
             feedback = response.prompt_feedback
             block_reason = feedback.block_reason.name if feedback and feedback.block_reason else 'Unknown'
             logger.error(f"Gemini response blocked or empty. Reason: {block_reason}")
             # Check safety ratings for details if available
             if feedback and feedback.safety_ratings:
                  for rating in feedback.safety_ratings: logger.error(f"  Safety Rating: {rating.category.name} - {rating.probability.name}")
             return {"is_vulnerable": False, "error": f"Response blocked or empty (Reason: {block_reason})"}

        finish_reason = response.candidates[0].finish_reason.name
        if finish_reason != "STOP":
             logger.warning(f"Gemini generation finished unexpectedly: {finish_reason}")

        response_text = response.text
        logger.debug(f"Raw Gemini Response Text:\n{response_text}")

        parsed_json = extract_json_output(response_text)
        if parsed_json:
            return parsed_json
        else:
            return {"is_vulnerable": False, "error": "Failed to parse valid JSON from LLM response.", "raw_response": response_text[:1000]}

    except Exception as e:
        logger.error(f"Error querying Gemini API: {e}", exc_info=True)
        # Check for specific API errors if needed (e.g., quota, authentication)
        return {"is_vulnerable": False, "error": f"Gemini API query failed: {type(e).__name__} - {e}"}


# --- CLI Definition ---
@click.group()
def cli():
    """
    Demo Tool: Analyzes packet features using the Gemini API
    for potential vulnerability assessment.
    """
    pass

@cli.command('analyze', short_help="Analyze provided packet features.")
@click.option('--src-ip', help='Source IP address (e.g., 192.168.1.100)')
@click.option('--dst-ip', help='Destination IP address (e.g., 10.0.0.5)')
@click.option('--src-port', type=int, help='Source port number (e.g., 49152)')
@click.option('--dst-port', type=int, help='Destination port number (e.g., 80, 443, 445)')
@click.option('--protocol', help='Protocol name (e.g., TCP, UDP, ICMP, DNS)')
@click.option('--tcp-flags', help='TCP flags (e.g., S, SA, PA, R, F)')
@click.option('--payload', help='Snippet of the packet payload (text or description). Limit length.')
@click.option('--length', type=int, help='Packet length in bytes.')
@click.option('--timestamp', help='Packet timestamp (ISO format string recommended).')
def analyze_packet(src_ip, dst_ip, src_port, dst_port, protocol, tcp_flags, payload, length, timestamp):
    """
    Sends packet features to the Gemini API for vulnerability analysis
    and displays the result. Provide as many features as possible.
    """
    click.echo("[*] Preparing packet features for analysis...")
    features = {
        "timestamp": timestamp, "source_ip": src_ip, "destination_ip": dst_ip,
        "protocol": protocol, "source_port": src_port, "destination_port": dst_port,
        "tcp_flags": tcp_flags, "packet_length": length,
        "payload_snippet": payload[:200] if payload else None, # Limit sent payload
    }
    features = {k: v for k, v in features.items() if v is not None} # Remove None values

    if not features:
        click.echo("[ERROR] No packet features provided. Use options like --src-ip, --dst-port, etc.", err=True)
        return

    click.echo("[*] Sending features to Gemini API for analysis...")
    result = query_gemini(create_gemini_prompt(features))

    click.echo("\n" + "="*25 + " Analysis Result " + "="*25)
    if not result:
        click.echo("[ERROR] Failed to get a valid response from the API.")
        return

    if "error" in result:
        click.secho(f"[ERROR] Analysis failed: {result['error']}", fg='red', err=True)
        if "raw_response" in result:
             click.echo("\n--- Raw Response Snippet (Debug) ---", err=True)
             click.echo(textwrap.indent(result['raw_response'], '  '), err=True)
             click.echo("--- End Raw Response ---", err=True)
        return

    is_vulnerable = result.get("is_vulnerable", False)
    if is_vulnerable:
        click.secho("[!] Potential Vulnerability Detected!", fg='red', bold=True)
        click.echo(f"  Type      : {result.get('vulnerability_type', 'N/A')}")
        click.echo(f"  Severity  : {result.get('severity', 'N/A')}")
        click.echo("\n  Description:")
        click.echo(textwrap.indent(result.get('description', 'N/A'), '    '))
        click.echo("\n  Suggested Mitigation Steps:")
        mitigation = result.get('mitigation_steps')
        if mitigation:
            for i, step in enumerate(mitigation, 1): click.echo(textwrap.indent(f"{i}. {step}", '    '))
        else: click.echo("    N/A")
    else:
        click.secho("[+] No specific vulnerability detected based on provided features.", fg='green')
        if result.get('description'):
             click.echo("\n  Analysis Description:")
             click.echo(textwrap.indent(result.get('description'), '    '))
    click.echo("="*67)


# --- Entry Point ---
if __name__ == '__main__':
    # Add check to ensure model was initialized
    if model is None:
         logger.critical("Model initialization failed earlier. Exiting CLI.")
         exit(1)
    cli()