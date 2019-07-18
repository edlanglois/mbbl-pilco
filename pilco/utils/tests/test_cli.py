"""Command-line Interface Tests."""
import pytest

from pilco.utils import cli


@pytest.fixture
def parser():
    import argparse

    return argparse.ArgumentParser()


def test_dict_lookup_action_no_args(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}
    )
    args = parser.parse_args([])
    assert args.foo is None


def test_dict_lookup_action_no_args_none_key(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar", None: "baz"}
    )
    args = parser.parse_args([])
    assert args.foo == "baz"


def test_dict_lookup_action_1arg(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}
    )
    args = parser.parse_args(["--foo", "b"])
    assert args.foo == "bar"


def test_dict_lookup_action_1arg_int_value(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}
    )
    args = parser.parse_args(["--foo", "a"])
    assert args.foo == 1


def test_dict_lookup_action_1arg_invalid(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["--foo", "c"])


def test_dict_lookup_action_nargs_opt(parser):
    parser.add_argument(
        "--foo",
        action=cli.DictLookupAction,
        choices={"a": 1, "b": "bar", None: "baz"},
        nargs="?",
    )
    args = parser.parse_args(["--foo", "b"])
    assert args.foo == "bar"
    args = parser.parse_args(["--foo"])
    assert args.foo == "baz"


def test_dict_lookup_action_nargs_many(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}, nargs="*"
    )
    args = parser.parse_args(["--foo"])
    args = parser.parse_args(["--foo", "a", "b", "a"])
    assert args.foo == [1, "bar", 1]
    args = parser.parse_args(["--foo"])
    assert args.foo == []


def test_dict_lookup_action_help(parser):
    parser.add_argument(
        "--foo",
        action=cli.DictLookupAction,
        choices={"a": 1, "b": "bar"},
        help="Some help.",
    )
    help_msg = parser.format_help()
    assert "foo" in help_msg
    assert "Some help." in help_msg


def test_dict_lookup_action_metavar(parser):
    parser.add_argument(
        "--foo",
        action=cli.DictLookupAction,
        choices={"a": 1, "b": "bar"},
        metavar="THEKEY",
    )
    assert "THEKEY" in parser.format_help()


def test_dict_lookup_action_dest(parser):
    parser.add_argument(
        "--foo", action=cli.DictLookupAction, choices={"a": 1, "b": "bar"}, dest="other"
    )
    args = parser.parse_args(["--foo", "b"])
    assert args.other == "bar"
    with pytest.raises(AttributeError):
        args.foo
