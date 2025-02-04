from __future__ import annotations

import logging

from contextlib import contextmanager
from typing_extensions import TYPE_CHECKING
from warbler.constant import LOGS

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing_extensions import Any


@contextmanager
def logger() -> Generator[None, Any, None]:
    try:
        log = logging.getLogger()
        log.setLevel(logging.INFO)

        path = LOGS.joinpath('warbler.log')
        path.open('a', encoding='utf-8').close()

        file_handler = logging.FileHandler(
            filename=path,
            encoding='utf-8'
        )
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[{asctime}] [{levelname}] {name}: {message}',
            '%Y-%m-%d %I:%M:%S %p',
            style='{'
        )

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)

        log.addHandler(file_handler)
        log.addHandler(stream_handler)

        yield
    finally:
        for handler in log.handlers:
            handler.close()
            log.removeHandler(handler)
