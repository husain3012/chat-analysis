import typer
from app.pipeline import cli_pipeline
import pandas as pd

app = typer.Typer()


@app.command()
def analyze_chat(
    file_path: str,
    deep: bool = typer.Option(True, help="Perform deep analysis"),
    sample_size: int = typer.Option(
        None, help="Number of messages to analyze, default value is None (All messages)"
    ),
    batch_size: int = typer.Option(32, help="Batch size for processing messages"),
):
    if not deep and batch_size:
        typer.echo("batch_size is ignored when deep analysis is disabled.")
    typer.echo(f"Processing chat file: {file_path}")
    chat_data = cli_pipeline(file_path, deep, sample_size, batch_size)
    typer.echo("Analysis completed!")
    typer.echo(chat_data)

    # output is dictionary
    # save dictionary to json file


if __name__ == "__main__":
    app()
