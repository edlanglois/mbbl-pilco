"""Command line / script utilities."""
# Copyright © 2019 Eric Langlois
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import inspect
import itertools
import logging
import time

import progressbar

__all__ = ["DictLookupAction", "CLIInitializable", "message_progress_bar"]


def filename_datetime():
    """The current time as formatted date-time string useable as a filename."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


class DictLookupAction(argparse.Action):
    """Argparse action that allows only keys from a given dictionary.

    The dictionary should be passed as the argument to `choices`.
    The argument to `default` is used as a key in the dictionary.
    """

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        default=None,
        choices=None,
        required=False,
        help=None,  # pylint: disable=redefined-builtin
        metavar=None,
    ):
        if choices is None:
            raise ValueError("Must set choices to the lookup dict.")
        self.dict = choices
        try:
            default_value = self.dict[default]
        except KeyError:
            if default is None:
                default_value = None
            elif nargs not in (None, "?"):
                default_value = [self.dict[value] for value in default]
            else:
                raise

        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=default_value,
            choices=self.dict.keys(),
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if self.nargs in (None, "?"):
            mapped_values = self.dict[values]
        else:
            mapped_values = [self.dict[v] for v in values]
        setattr(namespace, self.dest, mapped_values)


class LogLevelAction(argparse.Action):
    """Argparse action that set the logging level.

    The value of `const` is the default logging scope (or a list of scopes) to which
    level changes are applied.
    It defaults to the root logger, which applies the level globally.
    Suggested alternatives, assuming loggers created with logging.getLogger(__name__)
        __main__    : The logger for the main script.
        module.name : A module name (with scoping via dots).

    If a value for `default` is given, it is set as the global default log level.
    """

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        const="",
        metavar=("LOG_LEVEL", "SCOPE"),
        help=(
            "Set the logging level for the given logger scope(s). "
            "The log level is an integer or one of "
            "debug, info, warning, error, critical. "
            "Scope is a scope string. "
            "Typically a module name, or empty string for global. "
            "Default: %(const)r"
        ),
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs="+",
            const=const,
            type=str,
            help=help,
            metavar=metavar,
        )
        kwargs = {}
        if default is not argparse.SUPPRESS:
            level = getattr(logging, default.upper())
            kwargs["level"] = level
        logging.basicConfig(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        level_str, *scopes = values
        try:
            level = int(level_str)
        except ValueError:
            try:
                level = getattr(logging, level_str.upper())
            except AttributeError:
                parser.error(f"Invalid log level {level_str}.")

        if not scopes:
            scopes = self.const
        if isinstance(scopes, str):
            scopes = (scopes,)
        for scope in scopes:
            logging.getLogger(scope).setLevel(level)


class CLIInitializable(type):
    """Meta-class making a class command-line initializable.

    The command-line arguments are obtained from the class property _ARGS,
    which is a dict mapping argument name to a dict of `add_argument` keyword
    arguments.
    If _ARGS is not present, it is inferred from the signature of __init__.
    If _CLI_EXCLUDE is present, it defines a set of arguments to __init__
        that should be excluded from _ARGS.
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "_ARGS"):
            init = dct["__init__"]
            help_dict = {}
            if init.__doc__ is not None:
                in_args = False
                for line in init.__doc__.splitlines():
                    if not in_args:
                        if line.strip().startswith("Args:"):
                            in_args = True
                    elif line == "" or line.strip().startswith("Return"):
                        break
                    else:
                        try:
                            name, msg = line.split(":", 1)
                        except ValueError:
                            pass
                        else:
                            help_dict[name.strip()] = msg.strip()

            sig = inspect.signature(dct["__init__"])
            cls._ARGS = {}
            try:
                cli_exclude = cls._CLI_EXCLUDE
            except AttributeError:
                cli_exclude = frozenset()

            for name, parameter in sig.parameters.items():
                if name == "self" or name in cli_exclude:
                    continue
                flags = ["--" + name.replace("_", "-")]
                kwargs = {"dest": name}
                if parameter.default is not parameter.empty:
                    kwargs["default"] = parameter.default
                else:
                    kwargs["required"] = True
                if parameter.annotation is not parameter.empty:
                    kwargs["type"] = parameter.annotation
                else:
                    try:
                        default = kwargs["default"]
                    except KeyError:
                        pass
                    else:
                        if default is False:
                            kwargs["action"] = "store_true"
                        elif default is True:
                            kwargs["action"] = "store_false"
                            flag, = flags
                            flags = ["--no-" + flag[2:]]
                        else:
                            kwargs["type"] = type(default)
                try:
                    kwargs["help"] = help_dict[name]
                except KeyError:
                    pass
                cls._ARGS[name] = (flags, kwargs)

    def _add_argument(cls, parser, name, flags, kwargs):
        """Add an argument to the parser."""
        # pylint: disable=no-value-for-parameter
        flags, kwargs = cls._prepare_argument_args(name, flags, kwargs)
        parser.add_argument(*flags, **kwargs)

    def _prepare_argument_args(cls, name, flags, kwargs):
        """Prepare the arguments for parser.add_argument."""
        del name  # Unused
        return flags, kwargs

    def add_arguments(cls, parser):
        """Add necessary arguments to a parser.

        Args:
            parser: An `ArgumentParser` instance to which arguments are added.
        """
        for name, (flags, kwargs) in cls._ARGS.items():
            # pylint: disable=no-value-for-parameter
            cls._add_argument(parser, name, flags, kwargs)

    def from_args(cls, args, **overrides):
        """Create an instance from parsed args."""
        kwargs = {}
        for name in itertools.chain(cls._ARGS):
            try:
                kwargs[name] = getattr(args, name)
            except AttributeError:
                pass
        kwargs.update(overrides)
        return cls(**kwargs)


def message_progress_bar(names, max_value=None):
    """Creates a ProgressBar with dynamic messages.

    Args:
        names: List of dynamic message names.
        max_value: Optimal maximum progress value.

    Usage:
        with message_progress_bar(['loss'], n) as bar:
            for i in range(n):
                loss = do_something()
                bar.update(i, loss=loss)
    """
    message_widgets = list(
        itertools.chain.from_iterable(
            (progressbar.widgets.DynamicMessage(name), " ") for name in names
        )
    )
    if max_value is None:
        widgets = [
            progressbar.widgets.AnimatedMarker(),
            " ",
            *message_widgets,
            progressbar.widgets.BouncingBar(),
            " ",
            progressbar.widgets.Counter(),
            " ",
            progressbar.widgets.Timer(),
        ]
    else:
        widgets = [
            progressbar.widgets.Percentage(),
            " ",
            progressbar.widgets.SimpleProgress(),
            " ",
            *message_widgets,
            progressbar.widgets.Bar(),
            " ",
            progressbar.widgets.Timer(),
            " ",
            progressbar.widgets.AdaptiveETA(),
        ]
    return progressbar.ProgressBar(widgets=widgets, max_value=max_value)
