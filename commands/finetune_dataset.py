import logging
import argparse
import os

from .base import CLICommand
from transforkmers.postprocess import write_in_csv

log = logging.getLogger(__name__)


def split(string):
    if string is None:
        return string

    percents = string.split(",")
    if not 2 <= len(percents) <= 3:
        raise argparse.ArgumentTypeError(
            f"Split should take 2 or 3 percentages, e.g. '80,20' or '75,15,10' but '{string}' provided."
        )
    if sum([int(percent) for percent in percents]) != 100:
        raise argparse.ArgumentTypeError("Sum of percentages should be equal to 100.")
    return percents


class CreateDatasetCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "finetune-dataset",
            help="Build datasets for finetuning step.",
            parents=[cls.parent_parser],
        )
        sub_parser.add_argument(
            "--inputs",
            help=(
                "Path to input files. Each fasta is mapped to its number. "
                "e.g --inputs a.fa b.fa will map sequences from a.fa to class 0 and sequences from b.fa to class 1."
            ),
            type=str,
            required=True,
            nargs="+",
        )
        sub_parser.add_argument(
            "--output_dir",
            help="Prefix to save datasets. Default: ./dataset",
            default="dataset",
        )
        sub_parser.add_argument(
            "--max-len", help="Optional max sequence length. Default: None", type=int, default=None
        )
        sub_parser.add_argument(
            "--no-shuffle", help="Don't shuffle dataset. Default: disabled", action="store_true"
        )
        sub_parser.add_argument(
            "--split",
            help="Split input dataset in train/test or train/eval/test. ex: 80,20 or 75,15,10. Default: disabled",
            type=split,
            default=None,
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        import random
        from transforkmers.fasta import Fasta

        #######################################################################
        # Parse fastas
        dataset = []
        if len(args.inputs) < 2:
            raise Exception("Should provide 2 or more inputs.")

        for i, file in enumerate(args.inputs):
            log.info(f"Mapping sequences from {os.path.abspath(file)} to label {i}. ")
            with open(file) as fd:
                for rec in Fasta(fd):
                    dataset.append((rec.id, rec.seq.upper()[: args.max_len], i))
        log.info(f"{len(dataset)} sequences detected.")

        #######################################################################
        # Shuffle dataset
        if not args.no_shuffle:
            random.shuffle(dataset)
            log.warn("Dataset shuffled.")

        #######################################################################
        # Create CSVs
        header = ["id", "sequence", "label"]
        out_dir = args.output_dir

        if args.split is not None:
            if len(args.split) == 2:
                (train, eval) = args.split
                train_th = round(len(dataset) * int(train) / 100)
                write_in_csv(f"{out_dir}_train.csv", zip(*dataset[:train_th]), header=header)
                write_in_csv(f"{out_dir}_test.csv", zip(*dataset[train_th:]), header=header)
            else:
                (train, eval, _) = args.split
                train_th = round(len(dataset) * int(train) / 100)
                eval_th = train_th + round(len(dataset) * int(eval) / 100)
                write_in_csv(f"{out_dir}_train.csv", zip(*dataset[:train_th]), header=header)
                write_in_csv(f"{out_dir}_eval.csv", zip(*dataset[train_th:eval_th]), header=header)
                write_in_csv(f"{out_dir}_test.csv", zip(*dataset[eval_th:]), header=header)
            return
        write_in_csv(f"{out_dir}.csv", zip(*dataset), header=header)
