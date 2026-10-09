"""Exception classes and the optional-dependency import."""

from __future__ import annotations

import math

import pytest

from room_geometry_inference import (
    ConfigError,
    DatasetError,
    GeometryError,
    InversionError,
    MissingDependencyError,
    ModelFileError,
    require_module,
)


@pytest.mark.parametrize(
    "error", [ConfigError, DatasetError, ModelFileError, GeometryError, InversionError]
)
def test_expected_failures_are_value_errors(error):
    assert issubclass(error, ValueError)


def test_a_missing_module_names_the_extra_to_install_from_the_checkout():
    with pytest.raises(MissingDependencyError) as caught:
        require_module("no_such_module", extra="dl", feature="the test feature")
    assert isinstance(caught.value, ImportError)
    assert str(caught.value) == (
        'the test feature needs no_such_module: run pip install -e ".[dl]" in the project folder'
    )


def test_an_installed_module_is_returned():
    assert require_module("math", extra="dl", feature="the test feature") is math
