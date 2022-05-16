import argparse
import logging
import transformers

from .tokenizer import TokenizerCommand
from .pretrain_dataset import PreTrainDatasetCommand
from .pretrain import PreTrainCommand
from .finetune_dataset import CreateDatasetCommand
from .finetune import FinetuneCommand
from .test import TestCommand
from .predict import PredictCommand

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s >> %(message)s", datefmt="%y-%m-%d %H:%M:%S", force=True
)
log = logging.getLogger(__name__)


def set_logging(level):
    log_level = logging.getLevelName(level.upper())
    log.setLevel(level.upper())
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()


def main():
    parser = argparse.ArgumentParser("transforkmers")
    commands_parser = parser.add_subparsers(dest="subcommand")

    TokenizerCommand.register_subcommand(commands_parser)
    CreateDatasetCommand.register_subcommand(commands_parser)
    FinetuneCommand.register_subcommand(commands_parser)
    TestCommand.register_subcommand(commands_parser)
    PredictCommand.register_subcommand(commands_parser)
    PreTrainDatasetCommand.register_subcommand(commands_parser)
    PreTrainCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        exit(1)

    set_logging(args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()
