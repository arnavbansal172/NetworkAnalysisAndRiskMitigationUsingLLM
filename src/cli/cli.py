import click
import logging
import os
import sys

# Ensure the src directory is in the Python path
# This allows relative imports like 'from vulnerability_assessment...'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.vulnerability_assessment import assess # Assuming functions are defined here

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def main():
    """LLM-Powered Network Vulnerability Assessment Tool"""
    pass

@main.group()
def assess_cmd():
    """Commands for assessing vulnerabilities."""
    pass

@assess_cmd.command('pcap')
@click.option('--file', '-f', required=True, type=click.Path(exists=True, dir_okay=False), help='Path to the PCAP file.')
@click.option('--output', '-o', type=click.Path(dir_okay=False), help='Path to save the report (JSON format).')
@click.option('--model-path', help='Path to the fine-tuned LLM model (optional).') # Placeholder
def pcap_analysis(file, output, model_path):
    """Analyze a PCAP file for vulnerabilities."""
    logger.info(f"Starting PCAP analysis for: {file}")
    try:
        results = assess.assess_pcap(pcap_file_path=file, model_path=model_path)
        assess.save_report(results, output) # Assuming a function to handle saving
        logger.info(f"Analysis complete. {len(results)} potential vulnerabilities found.")
        if not output:
             # Print a summary if not saving to file
             assess.print_summary(results)
    except Exception as e:
        logger.error(f"Error during PCAP analysis: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)

@assess_cmd.command('live')
@click.option('--interface', '-i', required=True, help='Network interface to capture traffic from.')
@click.option('--packet-count', '-c', type=int, default=0, help='Number of packets to capture (0 for indefinite).')
@click.option('--log-file', '-l', type=click.Path(dir_okay=False), help='File to stream results.')
@click.option('--model-path', help='Path to the fine-tuned LLM model (optional).') # Placeholder
def live_analysis(interface, packet_count, log_file, model_path):
    """Analyze live network traffic for vulnerabilities."""
    # NOTE: Live capture usually requires root/administrator privileges.
    logger.info(f"Starting live analysis on interface: {interface}")
    click.echo(f"Attempting to capture on {interface}. This may require elevated privileges.")
    if os.geteuid() != 0:
         logger.warning("Script not run as root. Live packet capture might fail.")
         # Consider adding a more robust check or warning based on OS
    try:
        assess.assess_live(
            interface=interface,
            packet_count=packet_count,
            log_file=log_file,
            model_path=model_path
        )
        logger.info("Live analysis finished or interrupted.")
    except PermissionError:
        logger.error(f"Permission denied for capturing on interface {interface}. Try running with sudo.")
        click.echo(f"Error: Permission denied for interface {interface}. Run with sudo.", err=True)
    except Exception as e:
        logger.error(f"Error during live analysis: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)


@main.group()
def reports():
    """Commands for managing reports."""
    pass

@reports.command('list')
@click.option('--severity', '-s', type=click.Choice(['Critical', 'High', 'Medium', 'Low', 'Informational'], case_sensitive=False), help='Filter reports by severity.')
@click.option('--limit', '-n', type=int, default=20, help='Maximum number of reports to display.')
def list_reports(severity, limit):
    """List stored vulnerability reports."""
    logger.info(f"Listing reports (Severity: {severity}, Limit: {limit})")
    # Placeholder: Implement logic to query stored reports (e.g., from DB)
    click.echo(f"Listing reports... (Feature not yet implemented)")
    assess.list_reports_func(severity=severity, limit=limit) # Assuming a function exists


if __name__ == '__main__':
    main()