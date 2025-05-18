import os
os.environ["LIGHTGBM_VERBOSITY"] = "0"

import sys
import warnings
import logging
import click
from MLtool import MLCLIPipeline
from Datatool import DataTool
import pandas as pd

os.environ["LIGHTGBM_VERBOSITY"] = "0"
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

@click.group()
@click.version_option("1.0.0")
def cli():
    """Unified CLI: `ml-run` for MLtool, `data-run` for Datatool."""
    pass

@cli.command("start")
def start():
    """Display welcome message and list of available commands and flags."""
    click.secho("Welcome to the ML Track CLI!", fg="bright_blue", bold=True)
    click.echo("Here are the available commands and their options:\n")
    click.secho("ml-run", fg="green", bold=True)
    click.echo("  Run the ML pipeline with detailed output.")
    click.echo("  Usage: python click.py ml-run <data_path> <target_column> [OPTIONS]")
    click.echo("  Options:")
    click.echo("    -k, --api-key TEXT      GROQ API key (or via environment GROQ_API/GROQ_API_KEY)")
    click.echo("    -t, --test-size FLOAT   Test split fraction (default: 0.2)")
    click.echo("    -b, --batch-size INT    Dask batch size (default: 10000)")
    click.echo("    -o, --output-dir TEXT   Directory to save models (default: saved_models)\n")
    click.echo("    -T, --temperature FLOAT (0.00-1.00) strength of LLM (default if 0.7)\n")
    
    click.secho("data-run", fg="green", bold=True)
    click.echo("  Analyze & visualize your dataset with LLM support.")
    click.echo("  Usage: python click.py data-run <data_path> <target_column> [OPTIONS]")
    click.echo("  Options:")
    click.echo("    -k, --api-key TEXT      GROQ API key (required for LLM features)")
    click.echo("    -n, --top-n INT         Number of top features to show (default: 5)")
    click.echo("    -q, --query TEXT        Optional LLM question about the data")
    click.echo("    -i, --interactive       Enable interactive query mode after analysis")
    click.echo("    --no-plots              Skip generating plots")
    click.echo("    --no-llm                Skip LLM summary and query")
    click.echo("\n--------------------------------------------------\n")
    
# ----- MLtool runner with design improvements -----
@cli.command("ml-run")
@click.argument("data", type=click.Path(exists=True))
@click.argument("target", type=str)
@click.option("-k", "--api-key", envvar=["GROQ_API", "GROQ_API_KEY"], help="GROQ API key")
@click.option("-t", "--test-size", default=0.2, help="Test split fraction")
@click.option("-b", "--batch-size", default=10000, help="Dask batch size")
@click.option("-o", "--output-dir", default="saved_models", help="Where to save models")
@click.option("-T", "--temperature", type=float, default=None, help="Temperature for pipeline decision power (range: 0.0 to 1.0)")
def ml_run(data, target, api_key, test_size, batch_size, temperature, output_dir):

    if not api_key:
        raise click.UsageError("API key required for MLtool (use -k or env).")
    os.makedirs(output_dir, exist_ok=True)
    click.secho(
        f"[MLtool] test_size={test_size}, batch_size={batch_size}, temperature={temperature}",
        fg="cyan", bold=True
    )

    # Pass temperature into the pipeline constructor
    pipe = MLCLIPipeline(
        api_key,
        test_size=test_size,
        batch_size=batch_size,
        temperature=temperature
    )
    with click.progressbar(
        length=1,
        label="Running ML pipeline",
        fill_char=click.style('#', fg='green')
    ) as bar:
        res = pipe.run_pipeline(data, target, output_dir)
        bar.update(1)
    
    # ----- Display detailed results in a more colorful style -----
    click.secho("\n" + "="*40, fg="magenta")
    click.secho("Raw Data Summary", fg="magenta", bold=True)
    click.secho("="*40, fg="magenta")
    shape = res['raw_data_info']['shape']
    click.echo(f"Dataset Shape: {shape[0]} rows, {shape[1]} columns")
    click.echo(f"Target Column: {target}")
    
    click.secho("\nMissing Values:", fg="yellow", bold=True)
    for col, count in res['raw_data_info']['missing_values'].items():
        click.echo(f"  - {col}: {count}")
    
    click.secho("\nTarget Distribution:", fg="yellow", bold=True)
    if res['task_type'] == 'classification':
        for cls, cnt in res['raw_data_info']['target_distribution'].items():
            click.echo(f"  - Class {cls}: {cnt} samples")
    else:
        stats = res['raw_data_info']['target_distribution']
        click.echo(f"  - Mean: {stats.get('mean', 0):.2f}")
        click.echo(f"  - Std: {stats.get('std', 0):.2f}")
        click.echo(f"  - Range: {stats.get('min', 0):.2f} to {stats.get('max', 0):.2f}")
    
    click.secho("\n" + "="*40, fg="blue")
    click.secho("Pipeline Execution Details", fg="blue", bold=True)
    click.secho("="*40, fg="blue")
    click.echo(f"- Task Type: {res['task_type'].capitalize()}")
    click.echo(f"- Train/Test Split: {100*(1-pipe.test_size):.0f}%/{100*pipe.test_size:.0f}%")
    
    click.secho("\nPreprocessing Steps:", fg="green", bold=True)
    prep = res['plan']['preprocessing']
    click.echo(f"  - Numeric Features: Imputed with {prep.get('numeric_strategy', 'N/A')}")
    click.echo(f"  - Categorical Features: Imputed with {prep.get('categorical_strategy', 'N/A')} + OneHotEncoded")
    scaling = "Applied" if prep.get('scale_features', False) else "Not Applied"
    click.echo(f"  - Feature Scaling: {scaling}")
    
    fe = prep.get('feature_engineering', {})
    eng_steps = []
    if fe.get('text_features'):
        eng_steps.append("Text vectorization")
    if fe.get('date_features'):
        eng_steps.append("Date feature extraction")
    click.echo(f"  - Feature Engineering: {', '.join(eng_steps) if eng_steps else 'None'}")
    
    click.secho("\nModel Selection:", fg="bright_cyan", bold=True)
    cv_folds = res['plan']['model_selection'].get('cv_folds', 'N/A')
    click.echo(f"- Validation Strategy: {cv_folds}-fold CV")
    click.echo(f"- Selected Model: {res['model'].__class__.__name__}")
    
    click.secho("\nFinal Metrics:", fg="bright_yellow", bold=True)
    for metric, value in res['metrics'].items():
        click.echo(f"  - {metric.upper()}: {value:.4f}")
    
    click.secho(f"\nModel saved to: {res['model_path']}", fg="bright_magenta")

# ----- Datatool runner 
@cli.command("data-run")
@click.argument("data", type=click.Path(exists=True))
@click.argument("target", type=str)
@click.option("-k", "--api-key", envvar=["GROQ_API", "GROQ_API_KEY"], help="GROQ API key")
@click.option("-n", "--top-n", default=5, help="How many top features to show")
@click.option("-q", "--query", default=None, help="Optional LLM question about the data")
@click.option("-i", "--interactive", is_flag=True, help="Enable interactive query mode after analysis")
@click.option("--no-plots", is_flag=True, help="Skip generating plots")
@click.option("--no-llm", is_flag=True, help="Skip LLM summary and query")
def data_run(data, target, api_key, top_n, query, interactive, no_plots, no_llm):
    """Analyze & visualize dataset with comprehensive EDA, plus LLM summary/query."""
    if not api_key and not no_llm:
        raise click.UsageError("API key required for Datatool LLM features (use -k or env).")
    
    dt = DataTool(api_key if not no_llm else None)
    
    with click.progressbar(length=1, label="Loading data", fill_char=click.style('#', fg='green')) as bar:
        df = dt.load_data(data)
        bar.update(1)
    
    with click.progressbar(length=1, label="Analyzing dataset", fill_char=click.style('#', fg='green')) as bar:
        info = dt.analyze_data(df, target)
        bar.update(1)
    
    # Print basic statistics
    click.secho("\n=== Dataset Statistics ===", fg="green", bold=True)
    dt.print_basic_stats(info)
    
    # Compute correlations
    with click.progressbar(length=1, label="Computing correlations", fill_char=click.style('#', fg='green')) as bar:
        corr = dt.compute_correlations(df)
        bar.update(1)
    
    if not no_plots:
        # Show correlation heatmap
        click.secho("\n=== Correlation Heatmap ===", fg="cyan", bold=True)
        dt.print_cli_heatmap(corr)
        
        # Show top scatter plots
        if target in corr.columns:
            click.secho(f"\n=== Top {top_n} Features vs {target} ===", fg="cyan", bold=True)
            dt.print_cli_scatter(df, corr, target, top_n=top_n)
        
        # Show missing data visualization
        if info['missing_values']['total_missing_cells'] > 0:
            click.secho("\n=== Missing Data Visualization ===", fg="cyan", bold=True)
            dt.print_missing_data_chart(df)
        
        # Show distribution of target if categorical
        if target in df.columns and (df[target].dtype == 'object' or df[target].nunique() < 10):
            click.secho(f"\n=== Distribution of Target ({target}) ===", fg="cyan", bold=True)
            dt.print_distribution_bars(df, target)
        elif target in df.columns:
            click.secho(f"\n=== Histogram of Target ({target}) ===", fg="cyan", bold=True)
            dt.print_cli_histogram(df, target)
    
    if not no_llm:
        # LLM summary
        click.secho("\n=== LLM Dataset Summary ===", fg="green", bold=True)
        with click.progressbar(length=1, label="Generating LLM summary", fill_char=click.style('#', fg='green')) as bar:
            summary = dt.get_llm_summary(info, corr, target)
            bar.update(1)
        click.echo(summary)
        
        # LLM query if provided
        if query:
            click.secho("\n=== LLM Answer ===", fg="yellow", bold=True)
            with click.progressbar(length=1, label="Processing query", fill_char=click.style('#', fg='green')) as bar:
                answer = dt.ask_query(query, info, corr, target)
                bar.update(1)
            click.echo(answer)
        
        # Interactive query loop
        if interactive:
            run_query_loop(dt, info, corr, target)

def run_query_loop(dt: DataTool, info: dict, corr: pd.DataFrame, target: str):
    """Interactive query loop for context-aware questions about the dataset."""
    click.secho("\n=== Interactive Query Mode ===", fg="bright_green", bold=True)
    click.echo("Enter questions about the dataset. Type 'exit', 'quit', or press Ctrl+C to exit.")

    conversation_history: List[Tuple[str, str]] = []

    try:
        while True:
            query = click.prompt("\nQuery", type=str).strip()
            if query.lower() in ('exit', 'quit', 'q'):
                click.secho("Exiting query mode.", fg="yellow")
                break
            if not query:
                continue

            click.secho("Processing query...", fg="cyan")
            # Pass conversation history into ask_query
            answer = dt.ask_query(query, conversation_history, info, corr, target)

            # Append to history
            conversation_history.append((query, answer))

            click.secho("\n=== Answer ===", fg="yellow", bold=True)
            click.echo(answer)

    except KeyboardInterrupt:
        click.secho("\nExiting query mode.", fg="yellow")
    click.echo("Thank you for using DataTool!")

if __name__ == "__main__":
    cli()
