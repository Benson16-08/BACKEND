
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class _UTCFormatter(logging.Formatter):

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

    def format(self, record: logging.LogRecord) -> str:
        record.utc_time = self.formatTime(record)
        return super().format(record)


def _build_logger(name: str = "mediassist") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        _UTCFormatter(
            fmt="%(utc_time)s  %(levelname)-8s  %(name)s  %(message)s"
        )
    )
    log.addHandler(console)

    try:
        file_handler = logging.FileHandler("mediassist.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            _UTCFormatter(
                fmt='{"time":"%(utc_time)s","level":"%(levelname)s","msg":"%(message)s"}'
            )
        )
        log.addHandler(file_handler)
    except OSError:
        log.warning("Could not create mediassist.log")

    log.propagate = False
    return log


logger: logging.Logger = _build_logger()