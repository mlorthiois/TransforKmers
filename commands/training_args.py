from typing import Union, Optional, get_type_hints
from inspect import isclass
from enum import Enum
from argparse import ArgumentParser, _ArgumentGroup, ArgumentTypeError
import dataclasses

from transformers import TrainingArguments


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def parse_dataclass_field(parser: Union[ArgumentParser, _ArgumentGroup], field: dataclasses.Field):
    field_name = f"--{field.name}"
    kwargs = field.metadata.copy()
    if isinstance(field.type, str):
        raise RuntimeError(
            "Unresolved type detected, which should have been done with the help of "
            "`typing.get_type_hints` method by default"
        )

    origin_type = getattr(field.type, "__origin__", field.type)
    if origin_type is Union:
        if len(field.type.__args__) != 2 or type(None) not in field.type.__args__:
            raise ValueError(
                "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union`"
            )
        if bool not in field.type.__args__:
            field.type = (
                field.type.__args__[0]
                if isinstance(None, field.type.__args__[1])
                else field.type.__args__[1]
            )
            origin_type = getattr(field.type, "__origin__", field.type)

    if isinstance(field.type, type) and issubclass(field.type, Enum):
        kwargs["choices"] = [x.value for x in field.type]
        kwargs["type"] = type(kwargs["choices"][0])
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        else:
            kwargs["required"] = True

    elif field.type is bool or field.type is Optional[bool]:
        kwargs["type"] = string_to_bool
        if field.type is bool or (
            field.default is not None and field.default is not dataclasses.MISSING
        ):
            default = False if field.default is dataclasses.MISSING else field.default
            kwargs["default"] = default
            kwargs["nargs"] = "?"
            kwargs["const"] = True

    elif isclass(origin_type) and issubclass(origin_type, list):
        kwargs["type"] = field.type.__args__[0]
        kwargs["nargs"] = "+"
        if field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        elif field.default is dataclasses.MISSING:
            kwargs["required"] = True

    else:
        kwargs["type"] = field.type
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        else:
            kwargs["required"] = True

    parser.add_argument(field_name, **kwargs)


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
        parse_dataclass_field(trainer, field)


def parse_training_args(args):
    ta_fields = set([field.name for field in dataclasses.fields(TrainingArguments)])
    return dict((key, value) for key, value in vars(args).items() if key in ta_fields)
