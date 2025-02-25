import typer
from app.pipeline import cli_pipeline
import pandas as pd
from typing_extensions import Annotated
from enum import Enum
from typing import List
import os
import shutil
import torch

app = typer.Typer()


class Platform(str, Enum):
    ios = "ios"
    android = "android"

    def __str__(self):
        return self.value


@app.command()
def analyze(
    platform: Annotated[
        Platform, typer.Argument(help="Is the chat from an iOS or Android device?")
    ],
    file_path: str = typer.Argument(..., help="Path to the chat file"),
    deep: bool = typer.Option(True, help="Perform deep analysis"),
    sample_size: int = typer.Option(
        None, help="Number of messages to analyze, default value is None (All messages)"
    ),
    batch_size: int = typer.Option(32, help="Batch size for processing messages"),
):
    if not deep and batch_size:
        typer.echo("batch_size is ignored when deep analysis is disabled.")
    typer.echo(f"Processing chat file: {file_path}")
    chat_data = cli_pipeline(platform, file_path, deep, sample_size, batch_size)
    typer.echo("Analysis completed!")
    typer.echo(chat_data)

    # output is dictionary
    # save dictionary to json file


@app.command()
def setup_cuda():

    if torch.cuda.is_available():
        type.echo("CUDA is available")
    else:
        typer.echo("CUDA is not available")

        if os.name == "nt":
            typer.echo("Setting up CUDA")
            commands = [
                "pip uninstall torch -y",
                "pip install torch  --index-url https://download.pytorch.org/whl/cu121",
            ]
            for command in commands:
                os.system(command)

        else:
            os.system(f"No CUDA setup for {os.name}")

    typer.echo("CUDA setup complete")


if __name__ == "__main__":
    app()
