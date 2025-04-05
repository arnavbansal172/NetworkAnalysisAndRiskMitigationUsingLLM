import click
import logging
import os
import sys
from pathlib import Path
import time
from scapy.error import Scapy_Exception

# --- Project Setup ---
# Add project root to sys.path to allow imports like 'from src.vulnerability...'
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Module Imports ---
try:
    # Import only necessary top-level functions/classes needed by CLI
    from src.vulnerability_assessment import assess
    from src.vulnerability_assessment.rules import RuleEngine, SEVERITY_LEVELS
except ImportError as e:
    # Provide helpful error message if imports fail
    print(f"[ERROR] Failed to import required modules: {e}", file=sys.stderr)
    print("Please ensure that:", file=sys.stderr)
    print("  1. You are running this command from the project root directory ('your-project/').", file=sys.stderr)
    print("  2. Your virtual environment is activated.", file=sys.stderr)
    print("  3. All dependencies in 'requirements.txt' are installed.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
     print(f"[ERROR] An unexpected error occurred during module import: {e}", file=sys.stderr)
     sys.exit(1)

# --- Logging Configuration ---
# Configure root logger based on CLI option, propagate level
# Use a more detailed format for file/debug logging if needed later
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(name)-22s : %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("cli") # Logger specific to CLI actions

# Silence overly verbose libraries unless log level is DEBUG
def configure_library_logging(level):
    log_level_map = {
        'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING,
        'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL
    }
    effective_level = log_level_map.get(level.upper(), logging.INFO)

    if effective_level <= logging.DEBUG:
        # Show more detail in debug mode
        logging.getLogger("scapy.runtime").setLevel(logging.INFO)
        logging.getLogger("vulnerability_assessment").setLevel(logging.DEBUG)
        logging.getLogger("llm_interface").setLevel(logging.DEBUG) # Assuming this module exists
    elif effective_level <= logging.INFO:
        logging.getLogger("scapy.runtime").setLevel(logging.WARNING)
        logging.getLogger("vulnerability_assessment").setLevel(logging.INFO)
        logging.getLogger("llm_interface").setLevel(logging.INFO)
    else: # WARNING or higher
        logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
        logging.getLogger("vulnerability_assessment").setLevel(logging.WARNING)
        logging.getLogger("llm_interface").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.ERROR) # Usually noisy


# --- Main CLI Group ---
@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--log-level', '-L', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False), default='INFO', help='Set application logging level.')
@click.version_option(version='0.1.0', prog_name='VulnAssessLLM') # Example version
def main(log_level):
    """
    VulnAssessLLM: Network Vulnerability Assessment using TinyLlama.

    Analyzes network traffic (PCAP/Live) using rules & a fine-tuned LLM.
    """
    # Set the effective logging level for all loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    configure_library_logging(log_level.upper())
    logger.info(f"VulnAssessLLM started. Logging level: {log_level.upper()}")

# --- Assessment Command Group ---
@main.group(name='assess', short_help="Analyze network traffic for vulnerabilities.")
def assess_cmd():
    """Commands for assessing vulnerabilities in PCAP files or live traffic."""
    pass

# --- Common Assessment Options ---
common_options = [
    click.option('--model', '-m', 'model_adapter_path',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
              help='Path to the fine-tuned LoRA adapter directory. If omitted, uses the base TinyLlama model (limited capability).'),
    click.option('--no-llm', is_flag=True, default=False, help='Disable LLM analysis (perform rule-based checks only).'),
    click.option('--rules-first/--llm-first', default=True, help='Process rules before LLM (default) or vice versa (not recommended).') # Keep default rules-first
]

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

# --- PCAP Analysis Command ---
@assess_cmd.command('pcap', short_help="Analyze a PCAP file.")
@click.option('--file', '-f', 'pcap_file',
              required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
              help='Path to the network capture file (PCAP/PCAPNG).')
@click.option('--output', '-o',
              type=click.Path(dir_okay=False, writable=True, path_type=Path),
              help='Save detailed findings report to a JSON file.')
@add_options(common_options)
@click.pass_context
def pcap_analysis(ctx, pcap_file, output, model_adapter_path, no_llm, rules_first):
    """
    Analyze a recorded network capture file (PCAP/PCAPNG) using
    predefined rules and/or a fine-tuned TinyLlama model.
    """
    start_time = time.monotonic()
    logger.info(f"Received PCAP analysis request for '{pcap_file.name}'")

    # --- Configuration ---
    rule_engine = RuleEngine() # Loads default rules
    analyze_with_llm = not no_llm
    # Note: rules_first logic is handled within assess.py's _process_single_packet

    adapter_path_str = str(model_adapter_path) if model_adapter_path else None

    click.echo(f"[*] Starting Analysis for: {pcap_file}")
    click.echo(f"    Mode: Rules {'Enabled' if rules_first else 'Enabled (After LLM)'}, LLM {'DISABLED' if no_llm else 'ENABLED'}")
    if analyze_with_llm:
        click.echo(f"    LLM Adapter: {adapter_path_str or 'None (Using Base Model)'}")

    # --- Execute Analysis ---
    try:
        results = assess.assess_pcap(
            pcap_file_path=pcap_file,
            rule_engine=rule_engine,
            model_adapter_path=adapter_path_str, # Pass as string or None
            analyze_with_llm=analyze_with_llm
            # progress_interval can be added as option if needed
        )

        # --- Handle Results ---
        if results:
            if output:
                try:
                    assess.save_report(results, output)
                    click.echo(f"\n[+] Analysis complete. Found {len(results)} findings. Report saved to: {output}")
                except Exception as e:
                     click.echo(f"\n[!] Error saving report: {e}", err=True)
                     assess.print_summary(results) # Print summary if save failed
            else:
                assess.print_summary(results)
                click.echo(f"\n[+] Analysis complete. Found {len(results)} findings (summary above).")
        else:
             click.echo("\n[+] Analysis complete. No potential vulnerabilities identified.")

    except FileNotFoundError:
        logger.error(f"Input PCAP file not found: {pcap_file}")
        click.echo(f"[ERROR] Input file not found: '{pcap_file}'", err=True)
        sys.exit(1)
    except Scapy_Exception as se:
         logger.error(f"Scapy error processing PCAP '{pcap_file.name}': {se}")
         click.echo(f"[ERROR] Failed to read or process PCAP file '{pcap_file.name}': {se}", err=True)
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during PCAP analysis: {e}") # Log full traceback
        click.echo(f"\n[ERROR] An unexpected error occurred: {e}", err=True)
        sys.exit(1)
    finally:
         duration = time.monotonic() - start_time
         logger.info(f"PCAP analysis completed in {duration:.2f} seconds.")
         click.echo(f"[*] Analysis duration: {duration:.2f} seconds.")


# --- Live Analysis Command ---
@assess_cmd.command('live', short_help="Analyze live network traffic (requires root).")
@click.option('--interface', '-i',
              required=True,
              help='Network interface name for live capture (e.g., eth0, enp0s3, Wi-Fi).')
@click.option('--packet-count', '-c',
              type=int, default=0, show_default="unlimited",
              help='Number of packets to capture (0 for unlimited).')
@click.option('--log-file', '-l',
              type=click.Path(dir_okay=False, writable=True, path_type=Path),
              help='Stream findings (JSON Lines) to this file instead of console.')
@add_options(common_options)
@click.pass_context
def live_analysis(ctx, interface, packet_count, log_file, model_adapter_path, no_llm, rules_first):
    """
    Analyze live network traffic from a specified interface using
    rules and/or the LLM. Requires administrator/root privileges.
    """
    start_time = time.monotonic()
    logger.info(f"Received live analysis request for interface '{interface}'")

    # --- Permission Check/Warning ---
    try:
        if os.geteuid() != 0:
             click.echo(click.style("[WARNING] Live capture requires root/administrator privileges. Attempting anyway...", fg='yellow'), err=True)
    except AttributeError: # Windows
        click.echo(click.style("[WARNING] Cannot check privileges on this OS. Capture may fail without admin rights.", fg='yellow'), err=True)
    except Exception as e: # Catch other potential errors during privilege check
         logger.warning(f"Could not reliably check privileges: {e}")
         click.echo(click.style("[WARNING] Could not check user privileges.", fg='yellow'), err=True)


    # --- Configuration ---
    rule_engine = RuleEngine()
    analyze_with_llm = not no_llm
    adapter_path_str = str(model_adapter_path) if model_adapter_path else None

    click.echo(f"[*] Starting Live Analysis on Interface: {interface}")
    click.echo(f"    Packet Count Limit: {'Unlimited' if packet_count == 0 else packet_count}")
    click.echo(f"    Mode: Rules {'Enabled' if rules_first else 'Enabled (After LLM)'}, LLM {'DISABLED' if no_llm else 'ENABLED'}")
    if analyze_with_llm:
        click.echo(f"    LLM Adapter: {adapter_path_str or 'None (Using Base Model)'}")
    if log_file:
        click.echo(f"    Logging findings to: {log_file}")
    else:
        click.echo("    Logging findings to: Console (stdout)")
    if packet_count == 0:
         click.echo("[*] Capturing indefinitely. Press Ctrl+C to stop.")

    # --- Execute Analysis ---
    try:
        assess.assess_live(
            interface=interface,
            rule_engine=rule_engine,
            packet_count=packet_count,
            log_file=log_file,
            model_adapter_path=adapter_path_str,
            analyze_with_llm=analyze_with_llm
        )
        # assess_live handles internal exceptions and logs them, but CLI should know if it finished normally vs interrupted
        click.echo("\n[+] Live analysis concluded.")

    except (PermissionError, OSError, Scapy_Exception, RuntimeError) as e:
         # Catch specific errors re-raised by assess_live for clear CLI feedback
         logger.critical(f"Live analysis failed: {e}")
         click.echo(f"\n[ERROR] Live Analysis Failed: {e}", err=True)
         sys.exit(1)
    except KeyboardInterrupt:
         click.echo("\n[*] Live analysis interrupted by user (Ctrl+C).", err=True) # Use err stream for interrupt message
         sys.exit(0) # Exit gracefully on interrupt
    except Exception as e:
        # Catch any unexpected errors not handled within assess_live
        logger.critical(f"An unexpected error occurred during live analysis setup or termination: {e}", exc_info=True)
        click.echo(f"\n[ERROR] An unexpected error occurred: {e}", err=True)
        sys.exit(1)
    finally:
        duration = time.monotonic() - start_time
        # Avoid printing duration if it was extremely short (e.g., immediate error)
        if duration > 0.5:
             logger.info(f"Live analysis session duration: {duration:.2f} seconds.")
             click.echo(f"[*] Session duration: {duration:.2f} seconds.")


# --- Reports Command Group (Placeholder) ---
@main.group(name='reports', short_help="Manage vulnerability reports (placeholder).")
def reports_cmd():
    """Commands for managing stored vulnerability reports (Requires Database Backend)."""
    pass

@reports_cmd.command('list')
@click.option('--severity', '-s', type=click.Choice(list(SEVERITY_LEVELS.keys()), case_sensitive=False), help='Filter by severity.')
@click.option('--limit', '-n', type=int, default=20, help='Max reports to display.')
def list_reports(severity, limit):
    """List stored vulnerability reports (Not Implemented)."""
    logger.info(f"Listing reports request: Severity='{severity or 'Any'}', Limit={limit}")
    assess.list_reports_func(severity=severity, limit=limit)


# --- Entry Point ---
if __name__ == '__main__':
    # Set environment variable potentially needed by underlying libraries on some systems
    # Example: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # For PyTorch on macOS
    main()