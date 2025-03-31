# src/cli/cli.py

import click
import logging
import os
import sys
import json

# --- Project Setup ---
# Ensure the src directory is in the Python path for relative imports
# This allows running 'python src/cli/cli.py ...' directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Module Imports ---
# Import necessary functions and constants from our modules
try:
    from src.vulnerability_assessment import assess, rules, llm_interface
except ImportError as e:
    print(f"Error importing project modules: {e}", file=sys.stderr)
    print("Please ensure you are running from the project root directory or have installed the package.", file=sys.stderr)
    sys.exit(1)

# --- Logging Configuration ---
# Configure basic logging (can be overridden by command-line options later if needed)
# Using a simple format for CLI output readability
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# Get a specific logger for the CLI module
logger = logging.getLogger(__name__)
# Silence overly verbose loggers from libraries if they become noisy
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("scapy").setLevel(logging.ERROR)


# --- Main CLI Group ---
@click.group()
@click.version_option(package_name='your-project-name') # Add version if you set it up
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False), default='INFO', help='Set the logging level.')
def main(log_level):
    """
    LLM-Powered Network Vulnerability Assessment Tool.

    Uses models like Open Llama 3B to analyze network traffic.
    """
    # Update logging level based on command line argument
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to {log_level}")
    # Any other global setup can go here

# --- Assessment Command Group ---
@main.group(name='assess')
def assess_cmd():
    """Commands for assessing vulnerabilities in network traffic."""
    pass

@assess_cmd.command('pcap')
@click.option('--file', '-f',
              required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help='Path to the PCAP file to analyze.')
@click.option('--output', '-o',
              type=click.Path(dir_okay=False, writable=True),
              help='Path to save the JSON report file. If not provided, prints summary to console.')
@click.option('--model', '-m',
              default=llm_interface.MODEL_NAME, # Use default model from llm_interface
              help=f'LLM model name or path (Hugging Face). Default: {llm_interface.MODEL_NAME}')
def pcap_analysis(file, output, model):
    """Analyze a PCAP file for vulnerabilities using the LLM."""
    logger.info(f"Starting PCAP analysis: File='{file}', Model='{model}', Output='{output or 'Console Summary'}'")
    click.echo(f"Analyzing PCAP file: {file}...")
    try:
        # Call the assessment function from assess.py
        results = assess.assess_pcap(pcap_file_path=file, model_path=model)

        if results:
            if output:
                # Save the full report to the specified file
                assess.save_report(results, output)
                click.echo(f"Analysis complete. Found {len(results)} potential vulnerabilities. Report saved to: {output}")
            else:
                # Print a summary to the console if no output file specified
                assess.print_summary(results)
                click.echo(f"Analysis complete. Found {len(results)} potential vulnerabilities (summary above).")
        else:
             click.echo("Analysis complete. No potential vulnerabilities were identified.")

    except FileNotFoundError:
        logger.error(f"PCAP file not found: {file}")
        click.echo(f"Error: PCAP file not found at '{file}'.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during PCAP analysis: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred during analysis: {e}", err=True)
        sys.exit(1)

@assess_cmd.command('live')
@click.option('--interface', '-i',
              required=True,
              help='Network interface name for live capture (e.g., eth0, en0).')
@click.option('--packet-count', '-c',
              type=int, default=0, show_default=True,
              help='Number of packets to capture (0 means capture indefinitely until interrupted).')
@click.option('--log-file', '-l',
              type=click.Path(dir_okay=False, writable=True),
              help='File path to stream detected vulnerability reports (JSON Lines format).')
@click.option('--model', '-m',
              default=llm_interface.MODEL_NAME, # Use default model from llm_interface
              help=f'LLM model name or path (Hugging Face). Default: {llm_interface.MODEL_NAME}')
def live_analysis(interface, packet_count, log_file, model):
    """Analyze live network traffic using the LLM (requires root/admin)."""
    logger.info(f"Starting live analysis: Interface='{interface}', PacketCount='{packet_count or 'Infinite'}', Model='{model}', LogFile='{log_file or 'Console'}'")

    # --- Permission Check/Warning ---
    is_root = False
    try:
        # Check if running as root (Linux/macOS)
        if os.geteuid() == 0:
            is_root = True
    except AttributeError:
        # os.geteuid doesn't exist on Windows, try checking admin status differently if needed
        # For now, assume non-root/admin and let the capture library fail if permissions are insufficient
        logger.warning("Could not determine user privileges (non-Unix system?). Assuming non-root.")
        pass # Continue, capture library will likely raise PermissionError if needed

    if not is_root:
        click.echo(click.style("Warning: Live packet capture typically requires root or administrator privileges.", fg='yellow'), err=True)
        click.echo("Attempting to capture anyway...", err=True)
    else:
        click.echo(f"Attempting live capture on interface: {interface}")

    if log_file:
        click.echo(f"Detected vulnerabilities will be logged to: {log_file}")
    else:
        click.echo("Detected vulnerabilities will be printed to the console (JSON format).")
        if packet_count == 0:
             click.echo("Capturing indefinitely. Press Ctrl+C to stop.")

    try:
        # Call the live assessment function from assess.py
        assess.assess_live(
            interface=interface,
            packet_count=packet_count,
            log_file=log_file,
            model_path=model
        )
        click.echo("\nLive analysis finished or interrupted by user.")

    except (PermissionError, OSError) as e:
         # Catch common capture errors separately for better messages
        logger.error(f"Capture failed on interface '{interface}': {e}")
        click.echo(click.style(f"\nError: Failed to start capture on interface '{interface}'.", fg='red'), err=True)
        click.echo(f"Reason: {e}", err=True)
        click.echo("Please ensure the interface exists and you have the necessary permissions (run with sudo/admin?).", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during live analysis: {e}", exc_info=True)
        click.echo(f"\nAn unexpected error occurred during live analysis: {e}", err=True)
        sys.exit(1)


# --- Reports Command Group ---
@main.group(name='reports')
def reports_cmd():
    """Commands for managing stored vulnerability reports."""
    pass

@reports_cmd.command('list')
@click.option('--severity', '-s',
              type=click.Choice(list(rules.SEVERITY_LEVELS.keys()), case_sensitive=False),
              help='Filter reports by severity level.')
@click.option('--limit', '-n',
              type=int, default=20, show_default=True,
              help='Maximum number of reports to display.')
@click.option('--db-path', # Example: Option if using SQLite later
              type=click.Path(), default='data/database/vuln_reports.db', show_default=True,
              help='Path to the vulnerability report database.')
def list_reports(severity, limit, db_path):
    """List previously stored vulnerability reports (Placeholder)."""
    logger.info(f"Listing reports: Severity='{severity or 'Any'}', Limit={limit}, DB='{db_path}'")
    click.echo("Listing stored vulnerability reports...")

    # --- Placeholder for actual report listing logic ---
    # This part needs to be implemented when database storage is added.
    # It would involve:
    # 1. Connecting to the database (e.g., SQLite file at db_path).
    # 2. Querying the reports table, applying filters for severity.
    # 3. Limiting the results.
    # 4. Formatting and printing the retrieved reports.
    try:
        # Example call to a function that doesn't exist yet
        # reports_list = assess.get_reports_from_db(db_path, severity_filter=severity, result_limit=limit)
        # for report in reports_list:
        #     print(json.dumps(report, indent=2)) # Or format nicely
        click.echo(click.style("Note: Report listing functionality is not yet fully implemented.", fg='yellow'))
        assess.list_reports_func(severity=severity, limit=limit) # Call the placeholder
    except Exception as e:
         logger.error(f"Failed to list reports: {e}", exc_info=True)
         click.echo(f"Error listing reports: {e}", err=True)
         sys.exit(1)


# --- Entry Point ---
if __name__ == '__main__':
    # This allows running the script directly using `python src/cli/cli.py ...`
    # It's standard practice for Click applications.
    main()


    '''example commands
    python src/cli/cli.py --help
python src/cli/cli.py assess --help
python src/cli/cli.py assess pcap --help
python src/cli/cli.py assess live --help
python src/cli/cli.py reports --help

Analyze pcap
python src/cli/cli.py assess pcap -f data/pcap/your_traffic.pcap

Analyze pcap(save full report)
python src/cli/cli.py assess pcap -f data/pcap/your_traffic.pcap -o reports/pcap_analysis_results.json

analyze pcap specify  different mode
python src/cli/cli.py assess pcap -f data/pcap/your_traffic.pcap -m path/to/your/fine-tuned-model

analyze live traffic
# Capture 50 packets on eth0, print findings to console
sudo python src/cli/cli.py assess live -i eth0 -c 50

# Capture indefinitely on en0, stream findings to a file
sudo python src/cli/cli.py assess live -i en0 --log-file live_detections.jsonl

list reports
python src/cli/cli.py reports list --severity High --limit 10

change log level
python src/cli/cli.py --log-level DEBUG assess pcap -f data/pcap/your_traffic.pcap

'''