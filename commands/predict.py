import argparse
import logging

from .base import CLICommand

log = logging.getLogger(__name__)


class PredictCommand(CLICommand):
    @classmethod
    def register_subcommand(cls, commands_parser: argparse._SubParsersAction):
        sub_parser = commands_parser.add_parser(
            "predict",
            help="Predict sequences potential.",
            parents=[cls.parent_parser, cls.dl_parser],
        )
        sub_parser.add_argument(
            "--input",
            help="Path to input dataset in fasta.",
            type=argparse.FileType("r"),
            required=True,
        )
        sub_parser.add_argument(
            "--quantize-model",
            help="Dynamic quantize model if inference is on CPU. Default: False",
            action="store_true",
        )
        sub_parser.add_argument(
            "--per_device_eval_batch_size",
            help="Batch size",
            type=int,
            default=24,
        )
        sub_parser.add_argument(
            "--output_dir",
            help="Path to save results. Default: ./output/",
            type=str,
            default="./output/",
        )
        sub_parser.set_defaults(func=cls.run)

    @staticmethod
    def run(args):
        from transforkmers.tokenizer import DNATokenizer
        from transforkmers.dataset import SequenceDataset
        from transforkmers import postprocess

        from transformers import AutoModelForSequenceClassification, Trainer
        from transformers.training_args import TrainingArguments
        import torch

        ###############################################################################
        # Loading requirements
        tokenizer = DNATokenizer.from_pretrained(args.tokenizer)
        dataset = SequenceDataset.from_fasta(args.input, tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path_or_name)

        if args.quantize_model:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        ###############################################################################
        # Predicting
        training_args = TrainingArguments(
            output_dir="./",
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=training_args)
        predictions = trainer.predict(test_dataset=dataset)

        ###############################################################################
        # Compute and plot metrics
        y_pred = predictions.predictions
        y_pred_softmax = postprocess.softmax(y_pred)
        postprocess.write_in_csv(
            f"{args.output_dir}/output.csv",
            [dataset.ids, *y_pred_softmax.T],
            header=["ids"] + [str(i) for i in range(y_pred_softmax.shape[1])],
        )
