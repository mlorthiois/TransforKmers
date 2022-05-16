import argparse
import logging

from .base import CLICommand
from .training_args import add_training_args, parse_training_args

log = logging.getLogger(__name__)


class PreTrainCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "pretrain",
            help="Pretrain from scratch a model.",
            parents=[cls.parent_parser, cls.dl_parser],
        )
        add_training_args(sub_parser)
        sub_parser.add_argument(
            "--dataset",
            help="Path to dataset in txt used for pretraining.",
            type=argparse.FileType("r"),
            required=True,
        )
        sub_parser.add_argument(
            "--overload_config",
            help="Overload default config. Format: --model_config n_embd=10,scale_attn_weights=false",
            type=str,
            default=None,
        )
        sub_parser.add_argument(
            "--resume-from",
            help="Path to checkpoint directory. Default: disabled",
            type=str,
            default=None,
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transformers import (
            AutoModelForMaskedLM,
            AutoConfig,
            Trainer,
            TrainingArguments,
        )

        from transforkmers.tokenizer import DNATokenizer
        from transforkmers.data_collator import DataCollatorForKmerLM
        from transforkmers.dataset import SequenceDataset

        ###############################################################################
        # Loading requirements
        tokenizer = DNATokenizer.from_pretrained(args.tokenizer)
        log.info(f"Max len: {tokenizer.model_max_length}")
        data_collator = DataCollatorForKmerLM(tokenizer=tokenizer)
        config = AutoConfig.for_model(
            args.model_path_or_name,
            vocab_size=len(tokenizer),
            max_position_embeddings=tokenizer.model_max_length,
        )
        if args.overload_config is not None:
            log.warn("Overriding default model configuration.")
            config.update_from_string(args.overload_config)

        model = AutoModelForMaskedLM.from_config(config)
        log.info(f"Number of parameters to train: {model.num_parameters():,}")

        dataset = SequenceDataset.from_txt(
            args.dataset,
            tokenizer=tokenizer,
            block_size=tokenizer.model_max_length,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        ###############################################################################
        # Training
        training_args = TrainingArguments(**parse_training_args(args))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train(resume_from_checkpoint=args.resume_from)
        trainer.save_state()
        trainer.save_model()
