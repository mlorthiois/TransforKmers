import argparse
import logging

from .base import CLICommand
from .training_args import add_training_args, parse_training_args

log = logging.getLogger(__name__)


class TestCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "test",
            help="Test a model.",
            parents=[cls.parent_parser, cls.dl_parser],
        )
        add_training_args(sub_parser)
        sub_parser.add_argument(
            "--test_dataset",
            help="Path to dataset in csv for testing.",
            type=argparse.FileType("r"),
            required=True,
        )
        sub_parser.add_argument(
            "--quantize-model",
            help="Dynamic quantize model. Default: False",
            type=bool,
            default=False,
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transformers import AutoModelForSequenceClassification, Trainer
        from transformers.training_args import TrainingArguments
        import torch

        from transforkmers.tokenizer import DNATokenizer
        from transforkmers.dataset import SequenceDataset
        from transforkmers import postprocess

        ###############################################################################
        # Loading requirements
        tokenizer = DNATokenizer.from_pretrained(args.tokenizer)
        test_dataset = SequenceDataset.from_csv(args.test_dataset, tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path_or_name)

        # https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
        if args.quantize_model:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        ###############################################################################
        # Testing
        training_args = TrainingArguments(**parse_training_args(args))
        trainer = Trainer(model=model, args=training_args)
        predictions = trainer.predict(test_dataset=test_dataset)

        ###############################################################################
        # Compute and plot metrics
        y_pred, y_true = predictions.predictions, predictions.label_ids
        y_pred_softmax = postprocess.softmax(y_pred)
        postprocess.write_in_csv(
            f"{args.output_dir}/test.csv",
            [test_dataset.ids, y_true, *y_pred_softmax.T],
            header=["ids", "true_label"] + [str(i) for i in range(y_pred_softmax.shape[1])],
        )
