from abc import ABC, abstractmethod
import argparse


class CLICommand(ABC):
    """Base class for CLI subcommands

    Contains cli args in common between all subcommands.
    To inherit, subclass should implement register_subcommand and run methods.
    """

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--log_level",
        type=str,
        choices=["info", "warning", "error", "debug"],
        default="info",
        help="Loggin level. Default: info",
    )

    dl_parser = argparse.ArgumentParser(add_help=False)
    dl_parser.add_argument(
        "--tokenizer",
        help="Path to tokenizer config dir.",
        type=str,
        required=True,
    )
    dl_parser.add_argument(
        "--model_path_or_name",
        help="Path to model or model type name.",
        required=True,
        type=str,
    )

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: argparse._SubParsersAction, name: str, help: str):
        """CLI Args specific to subcommand"""

    @abstractmethod
    def run(self):
        """Function to run specific to subcommand"""
