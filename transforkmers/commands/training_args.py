from typing import Union, get_type_hints
from argparse import ArgumentParser, _ArgumentGroup
import dataclasses

from transformers import TrainingArguments, HfArgumentParser


def add_training_args(parser: Union[ArgumentParser, _ArgumentGroup]):
    trainer = parser.add_argument_group("Training arguments")
    type_hints = get_type_hints(TrainingArguments)
    for field in dataclasses.fields(TrainingArguments):
        if (
            not field.init
            or field.name in ["do_train", "do_eval", "do_predict", "log_level"]
            or "hub" in field.name
        ):
            continue
        field.type = type_hints[field.name]
        HfArgumentParser._parse_dataclass_field(trainer, field)


def parse_training_args(args):
    ta_fields = set([field.name for field in dataclasses.fields(TrainingArguments)])
    return dict((key, value) for key, value in vars(args).items() if key in ta_fields)
