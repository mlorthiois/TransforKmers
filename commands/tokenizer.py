from .base import CLICommand

import argparse
import logging

log = logging.getLogger(__name__)

ALPHABET = ["A", "T", "C", "G"]


def check_alphabet(string):
    splitted_alphabet = string.split(",")
    if len(splitted_alphabet) == 1:
        raise argparse.ArgumentTypeError(
            f"Alphabet is incorrect. Is {string} but should be formatted like A,T,C,G"
        )
    return splitted_alphabet


class TokenizerCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "tokenizer", help="Build custom tokenizer.", parents=[cls.parent_parser]
        )
        sub_parser.add_argument(
            "--alphabet",
            type=check_alphabet,
            default=",".join(ALPHABET),
            help=f"Alphabet to build kmers on. Default: {','.join(ALPHABET)}.",
        )
        sub_parser.add_argument(
            "--k",
            help=f"K-mer size.",
            type=int,
            required=True,
        )
        sub_parser.add_argument(
            "--max-len", help=f"Size to pad/trim input sequences.", type=int, required=True
        )
        sub_parser.add_argument(
            "--output", help=f"Path directory to save tokenizer.", type=str, default=None
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transforkmers.tokenizer import DNATokenizer

        tokenizer = DNATokenizer(k=args.k, alphabet=args.alphabet, model_max_length=args.max_len)
        tokenizer.save_pretrained(
            args.output if args.output is not None else f"tokenizer_k{args.k}_{args.max_len}"
        )
