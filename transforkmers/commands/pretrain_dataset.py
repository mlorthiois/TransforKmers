import argparse
import logging
import tqdm

from .base import CLICommand

log = logging.getLogger(__name__)


class PreTrainDatasetCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "pretrain-dataset",
            help="Create a dataset for the pretraining step.",
            parents=[cls.parent_parser],
        )
        sub_parser.add_argument(
            "--fasta",
            help="Path to fasta containing sequences.",
            type=argparse.FileType("r"),
            required=True,
        )
        sub_parser.add_argument(
            "--num-seq",
            help="Number of random sequences to pick.",
            type=int,
            default=1e6,
        )
        sub_parser.add_argument(
            "--len",
            help="Length of generated sequences. Default: 512.",
            type=int,
            default=512,
        )
        sub_parser.add_argument(
            "--n",
            help="Percentage of 'N' allowed inside sequences. Default: 0.1",
            type=float,
            default=0.1,
        )
        sub_parser.add_argument(
            "--seed",
            help="Seed. Default: None",
            type=int,
            default=None,
        )
        sub_parser.add_argument(
            "--output",
            help="Path where to store dataset in txt.",
            type=argparse.FileType("w"),
            default="dataset.txt",
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transforkmers.fasta import Fasta
        import random

        random.seed(args.seed)
        log.info(f"Parsing {args.fasta.name}...")
        fa = [seq.seq for seq in Fasta(args.fasta) if len(seq.seq) > 10 * args.len]

        log.info(f"Creating sequences...")
        for _ in tqdm.trange(args.num_seq):
            while True:
                try:
                    chr_idx = random.randint(0, len(fa) - 1)
                    chr = fa[chr_idx]
                    start = random.randint(0, len(chr) - args.len)
                except ValueError:
                    continue
                seq = chr[start : start + args.len].upper()

                if (seq.count("N") / len(seq)) < args.n:
                    args.output.write(seq + "\n")
                    break
        log.info(f"{args.output.name} successfully created")
