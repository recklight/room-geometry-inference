"""Logging setup: one handler of our own, foreign handlers untouched."""

from __future__ import annotations

import logging

import pytest

from room_geometry_inference import configure_logging
from room_geometry_inference.logging_utils import _OWN_HANDLER


@pytest.fixture
def root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def _own(root):
    return [handler for handler in root.handlers if getattr(handler, _OWN_HANDLER, False)]


def test_repeated_calls_do_not_stack_handlers(root_logger):
    configure_logging("INFO")
    configure_logging("DEBUG")
    assert len(_own(root_logger)) == 1
    assert root_logger.level == logging.DEBUG


def test_foreign_handlers_are_left_alone(root_logger):
    foreign = logging.NullHandler()
    root_logger.addHandler(foreign)
    configure_logging("WARNING")
    configure_logging("ERROR")
    assert foreign in root_logger.handlers


def test_a_log_file_receives_the_records(root_logger, tmp_path):
    target = tmp_path / "logs" / "run.log"
    configure_logging(logging.INFO, log_file=target)
    logging.getLogger("room_geometry_inference.test").info("measurement %d of %d", 1, 3)
    for handler in _own(root_logger):
        handler.flush()
    assert "measurement 1 of 3" in target.read_text(encoding="utf-8")
    assert len(_own(root_logger)) == 2


def test_unknown_levels_are_rejected(root_logger):
    with pytest.raises(ValueError, match="unknown logging level 'LOUD'"):
        configure_logging("LOUD")
