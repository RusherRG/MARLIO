from typing import Optional
from pathlib import Path
from os import path
import typer

default_output_dir = path.expanduser("~/.MARLIO-runner/")
sample_conf = "./config.yml"


def main(config: Path = typer.Option(sample_conf, exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
         verbose: bool = typer.Option(False, "--verbose/"),
         gui: bool = typer.Option(False, "--gui/--no-gui"),
         tensorboard: bool = typer.Option(False, "--tensorboard"),
         output_dir: Path = typer.Option(default_output_dir,
                                         exists=True, file_okay=False, dir_okay=True, writable=True, resolve_path=True),
         train: Optional[bool] = typer.Option(None, "--train/--test")):

    typer.echo(f"Hello {config} \n {output_dir}")


if __name__ == "__main__":
    typer.run(main)
