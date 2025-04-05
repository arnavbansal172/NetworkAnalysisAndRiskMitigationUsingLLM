# VulnAssessLLM: Network Vulnerability Assessment using TinyLlama

This project implements a network packet vulnerability assessment tool leveraging the `TinyLlama-1.1B-Chat-v1.0` model. It analyzes network traffic (from PCAP files or live capture) using a combination of predefined rules and a fine-tuned LLM to identify potential security vulnerabilities and suggest mitigation steps.

## Features

*   **Packet Analysis:** Analyzes IPv4/IPv6 packets from PCAP/PCAPNG files or live network interfaces.
*   **Rule-Based Detection:** Includes a configurable engine for fast detection of common issues (e.g., known bad IPs, insecure protocols, basic signatures).
*   **LLM-Powered Analysis:** Utilizes a fine-tuned `TinyLlama-1.1B-Chat-v1.0` model for deeper analysis and identification of more complex or nuanced vulnerabilities (requires fine-tuning for optimal performance).
*   **Vulnerability Reporting:** Generates structured JSON reports detailing findings, including description, severity, evidence, and mitigation suggestions.
*   **Mitigation Suggestions:** Provides actionable mitigation steps based on rule matches or LLM analysis.
*   **Optimized for Efficiency:** Uses iterative PCAP processing for large files and prioritizes rule checks before engaging the LLM. QLoRA is used for efficient fine-tuning.
*   **Command-Line Interface:** Provides a user-friendly CLI for running analyses and managing the tool.

## Project Structure

```
your-project/
├── venv/
├── data/
│   ├── database/      # Fine-tuning data (e.g., finetuning_data.jsonl)
│   ├── pcap/          # Sample PCAP files for analysis
├── models/            # Recommended location for base model & trained adapters
│   ├── base/
│   │   └── tinyllama/ # Downloaded base TinyLlama model files
│   └── tinyllama_1.1b_chat_vuln_adapter/ # Example output of fine-tuning
├── src/
│   ├── llm_training/  # Scripts for fine-tuning the LLM
│   │   ├── train.py
│   │   └── utils.py
│   ├── vulnerability_assessment/ # Core analysis logic
│   │   ├── __init__.py
│   │   ├── assess.py
│   │   ├── rules.py
│   │   ├── feature_extractor.py
│   │   └── llm_interface.py
│   ├── cli/            # Command-line interface
│   │   ├── cli.py
├── tests/             # Unit and integration tests
│   │   ├── __init__.py
│   │   └── test_placeholder.py # Add test files here
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore configuration
```

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   `pip` and `venv`
    *   System dependencies for Scapy's live capture (e.g., `libpcap-dev` on Debian/Ubuntu, `Npcap` on Windows). Check Scapy documentation for your OS.
    *   Git (for cloning)
    *   If training/running on GPU: NVIDIA drivers, CUDA Toolkit compatible with PyTorch and `bitsandbytes`.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd your-project
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `bitsandbytes` installation might require specific steps depending on your setup. Refer to its documentation if issues arise.)*

5.  **Download Base Model (Optional but Recommended):**
    Download the base `TinyLlama-1.1B-Chat-v1.0` model files locally to avoid repeated downloads during runtime/training. You can use Hugging Face CLI or a script.
    ```bash
    # Recommended: Create models directory
    mkdir -p models/base/tinyllama

    # Option 1: Use Hugging Face CLI (requires login: huggingface-cli login)
    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/base/tinyllama --local-dir-use-symlinks False

    # Option 2: Use a download script (like the 'download_model.py' provided in previous answers)
    # python download_model.py --model_key tinyllama --save_directory models/base/tinyllama
    ```
    *(Update paths in `llm_interface.py` or `train.py` if you download to a different location or want to use the online ID directly).*

## Fine-tuning (Optional but Recommended)

The base TinyLlama model has general knowledge but requires fine-tuning on network security data for optimal vulnerability assessment performance.

1.  **Prepare Data:** Create a `finetuning_data.jsonl` file in `data/database/` with instruction-following examples. Each line should be a JSON object with "instruction", "input" (packet features text), and "output" (target JSON analysis string) keys. See example format in `data/database/finetuning_data.jsonl`.
2.  **Run Training Script:**
    ```bash
    # Activate venv
    # Adjust parameters based on your hardware and dataset size
    python src/llm_training/train.py \
        --data_path data/database/finetuning_data.jsonl \
        --output_dir models/tinyllama_1.1b_chat_vuln_adapter \
        --base_model models/base/tinyllama  # Path to local base model OR Hugging Face ID
        --use_qlora \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation 4 \
        --learning_rate 2e-4 \
        --logging_steps 10 \
        --save_steps 50 \
        --max_seq_length 1024
    ```
    *   The fine-tuned adapter (LoRA weights) will be saved in the `--output_dir`.

## Usage (Command-Line Interface)

Make sure your virtual environment is activated (`source venv/bin/activate`).

**General Help:**
```bash
python src/cli/cli.py --help
python src/cli/cli.py assess --help
```

**Analyze PCAP File:**
```bash
# Rule analysis ONLY
python src/cli/cli.py assess pcap -f data/pcap/example.pcap --no-llm

# Rules + Base LLM (limited capability without fine-tuning)
python src/cli/cli.py assess pcap -f data/pcap/example.pcap

# Rules + Fine-tuned LLM (using saved adapter)
python src/cli/cli.py assess pcap -f data/pcap/example.pcap -m models/tinyllama_1.1b_chat_vuln_adapter

# Save full report to JSON file
python src/cli/cli.py assess pcap -f data/pcap/suspicious.pcap -o reports/suspicious_report.json -m models/tinyllama_1.1b_chat_vuln_adapter
```

**Analyze Live Traffic (Requires Root/Admin):**
```bash
# Rules + Fine-tuned LLM on interface eth0 (capture indefinitely)
sudo python src/cli/cli.py assess live -i eth0 -m models/tinyllama_1.1b_chat_vuln_adapter

# Rule analysis ONLY on wlan0, capture 100 packets
sudo python src/cli/cli.py assess live -i wlan0 -c 100 --no-llm

# Rules + Fine-tuned LLM, log findings to file
sudo python src/cli/cli.py assess live -i eth0 --log-file live_findings.jsonl -m models/tinyllama_1.1b_chat_vuln_adapter
```

**Change Logging Level:**
```bash
python src/cli/cli.py --log-level DEBUG assess pcap -f data/pcap/example.pcap -m models/tinyllama_1.1b_chat_vuln_adapter
```

## Testing

1.  **Install Testing Dependencies:**
    ```bash
    pip install pytest # If not already included in requirements.txt
    ```
2.  **Run Tests:** Navigate to the project root directory (`your-project/`) and run:
    ```bash
    pytest tests/
    ```
    *(This will discover and run all files starting with `test_` in the `tests/` directory. You need to create specific test files like `test_rules.py`, `test_feature_extractor.py`, etc. with actual test cases using `unittest` or `pytest` syntax).*

## TODO / Future Enhancements

*   Implement database storage/retrieval for reports (`reports list` command).
*   Add more sophisticated rules and external feed integration for `RuleEngine`.
*   Implement flow tracking in `feature_extractor.py`.
*   Develop comprehensive unit and integration tests.
*   Explore optimized inference deployment options (Triton, vLLM).
*   Web UI or API for interaction.

## License

(Specify your chosen license, e.g., MIT License, Apache 2.0)
```

---

**VI. Running Unit Tests**

1.  **Create Test Files:** Inside the `tests/` directory, create Python files starting with `test_`, for example:
    *   `tests/test_rules.py`
    *   `tests/test_feature_extractor.py`
    *   `tests/test_assess_logic.py` (might need mocking for LLM parts)

2.  **Write Tests:** Use `unittest` or `pytest` frameworks to write test cases.
    *   **Example `tests/test_rules.py` (using `unittest`):**
        ```python
        import unittest
        import sys
        from pathlib import Path

        # Add src to path to allow importing project modules
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))

        from src.vulnerability_assessment.rules import RuleEngine, SEVERITY_HIGH

        class TestRuleEngine(unittest.TestCase):

            def setUp(self):
                """Set up a RuleEngine instance for each test."""
                self.engine = RuleEngine(load_defaults=True) # Load default rules

            def test_malicious_ip_match(self):
                """Test the MAL_IP_BLOCKLIST rule."""
                features_src = {"src_ip": "198.51.100.1", "dst_ip": "10.0.0.1"}
                features_dst = {"src_ip": "10.0.0.1", "dst_ip": "203.0.113.5"}
                finding_src = self.engine.check_packet(features_src)
                finding_dst = self.engine.check_packet(features_dst)

                self.assertIsNotNone(finding_src)
                self.assertEqual(finding_src['rule_id'], "MAL_IP_BLOCKLIST")
                self.assertEqual(finding_src['severity'], SEVERITY_HIGH)

                self.assertIsNotNone(finding_dst)
                self.assertEqual(finding_dst['rule_id'], "MAL_IP_BLOCKLIST")

            def test_no_match(self):
                """Test a packet that should not match any default rule."""
                features = {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP", "dst_port": 8080}
                finding = self.engine.check_packet(features)
                self.assertIsNone(finding)

            def test_telnet_match(self):
                 """Test the INSECURE_TELNET rule."""
                 features = {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP", "dst_port": 23}
                 finding = self.engine.check_packet(features)
                 self.assertIsNotNone(finding)
                 self.assertEqual(finding['rule_id'], "INSECURE_TELNET")

        if __name__ == '__main__':
            unittest.main()
        ```

3.  **Run from Command Line:** Make sure your virtual environment is activated and you are in the `your-project/` directory.
    *   **Using `unittest`:**
        ```bash
        python -m unittest discover tests/
        ```
    *   **Using `pytest` (if installed):**
        ```bash
        pytest tests/
        ```

---

This consolidated output provides the core Python code, setup instructions, example data, and documentation outline, all tailored for the TinyLlama-1.1B model and optimized for performance as requested. Remember to add specific unit tests and potentially more sophisticated feature extraction or rule logic as needed.