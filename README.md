# LLM-Powered Network Packet Vulnerability Assessment Tool

This project aims to leverage the Mistral 7B Large Language Model (LLM) for assessing vulnerabilities in network packets, simulating an implementation within a large-scale enterprise environment.

## Project Structure

your-project/
├── venv/            # Virtual environment (created by user)
├── data/             # Data files
│   ├── database/      # Stores vulnerability info, potentially fine-tuning data index
│   ├── pcap/          # Sample vulnerable PCAP files for testing/analysis
├── src/              # Source code
│   ├── llm_training/ # LLM fine-tuning scripts
│   │   ├── train.py     # Main training script
│   │   ├── utils.py     # Helper functions for data loading, preprocessing
│   ├── vulnerability_assessment/ # Core assessment logic
│   │   ├── assess.py    # Main assessment functions (live & pcap)
│   │   ├── rules.py     # Defines vulnerability categories, severity mappings, output structures
│   │   ├── feature_extractor.py # Extracts relevant features from packets/flows
│   │   ├── llm_interface.py # Handles interaction with the fine-tuned LLM
│   ├── cli/           # Command-line interface code
│   │   ├── cli.py       # Defines CLI commands and arguments
├── tests/            # Unit tests for modules
│   ├── test_feature_extractor.py
│   ├── test_assessment_logic.py
│   ├── # ... other test files
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
├── .gitignore



## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd your-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some dependencies like `pypcap` might require specific system libraries (e.g., `libpcap-dev` on Debian/Ubuntu). Refer to library documentation.*

## Usage

*(Instructions on how to run the analysis will be added here)*

Example (placeholder):

```bash
# Analyze a PCAP file
python src/cli/cli.py assess pcap --file data/pcap/example.pcap

# Analyze live traffic (requires appropriate permissions)
sudo python src/cli/cli.py assess live --interface eth0
```

## Testing

To run the unit tests:
```bash
python -m unittest discover tests/
```