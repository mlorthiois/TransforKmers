import argparse
import logging

from .base import CLICommand
from .training_args import add_training_args, parse_training_args

log = logging.getLogger(__name__)


class FinetuneCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "finetune",
            help="Finetune a model.",
            parents=[cls.parent_parser, cls.dl_parser],
        )
        add_training_args(sub_parser)
        sub_parser.add_argument(
            "--train_dataset",
            help="Path to dataset in csv used for training.",
            type=argparse.FileType("r"),
            required=True,
        )
        sub_parser.add_argument(
            "--eval_dataset",
            help="Path to dataset in csv used for evaluation.",
            type=argparse.FileType("r"),
            default=None,
        )
        sub_parser.add_argument(
            "--split",
            help="Split ratio if no testing dataset provided. Default: 0.8",
            type=float,
            default=0.8,
        )
        sub_parser.add_argument(
            "--num_labels",
            help="Number of labels to predict. Eg. 2 for binary classification.",
            type=int,
            required=True,
        )
        sub_parser.add_argument(
            "--patience",
            help="Number of labels to predict. Eg. 2 for binary classification.",
            type=int,
            required=True,
            default=None,
        )
        sub_parser.add_argument(
            "--resume_from",
            help="Path to checkpoint directory. Default: disabled",
            type=str,
            default=None,
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transforkmers.tokenizer import DNATokenizer
        from transforkmers.dataset import SequenceDataset
        from transforkmers import postprocess

        from torch.utils.data import random_split
        from transformers import AutoModelForSequenceClassification, Trainer
        from transformers.training_args import TrainingArguments
        from transformers.trainer_callback import EarlyStoppingCallback

        ###############################################################################
        # Loading and creating datasets
        tokenizer = DNATokenizer.from_pretrained(args.tokenizer)
        if args.eval_dataset is not None:
            train_dataset = SequenceDataset.from_csv(args.train_dataset, tokenizer)
            eval_dataset = SequenceDataset.from_csv(args.eval_dataset, tokenizer)
        else:
            log.warning("Creating evaluation dataset from training dataset.")
            dataset = SequenceDataset.from_csv(args.train_dataset, tokenizer)
            train_n = round(len(dataset) * args.split)
            train_dataset, eval_dataset = random_split(dataset, (train_n, len(dataset) - train_n))

        ###############################################################################
        # Loading model
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path_or_name)
        log.warn(f"Set the number of predicted labels to {args.num_labels}")
        model.num_labels = args.num_labels

        ###############################################################################
        # Training
        callback = (
            [EarlyStoppingCallback(early_stopping_patience=args.patience)]
            if args.patience is not None
            else None
        )
        training_args = TrainingArguments(**parse_training_args(args))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callback,
        )
        trainer.train(resume_from_checkpoint=args.resume_from)
        trainer.save_state()
        trainer.save_model(output_dir=f"{args.output_dir}/best/")

        ###############################################################################
        # Threshold
        log.info("Evaluating...")
        predictions = trainer.predict(test_dataset=eval_dataset)
        y_pred, y_true = predictions.predictions, predictions.label_ids
        y_pred_softmax = postprocess.softmax(y_pred)
        postprocess.write_in_csv(
            f"{args.output_dir}/validation.csv",
            [eval_dataset.ids, y_true, *y_pred_softmax.T],
            header=["ids", "true_label"] + [str(i) for i in range(y_pred_softmax.shape[1])],
        )
