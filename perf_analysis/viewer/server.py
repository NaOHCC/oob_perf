# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve a local browser viewer for performance analysis artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

_ALLOWED_SORTS = frozenset(
    {
        "actual_ms",
        "projected_ms",
        "efficiency",
        "flops",
        "memory_bytes",
        "timestamp_us",
        "order",
    },
)
_ALLOWED_STATUSES = frozenset(
    {"all", "sequence-paired", "actual-only", "projected-only"},
)
_MAX_PAGE_SIZE = 500
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ViewerError(ValueError):
    """Raised when viewer artifacts or query parameters are invalid."""


@dataclass(frozen=True)
class ViewerData:
    """Validated, immutable data served by one viewer process."""

    analysis: dict[str, Any]
    calls: list[dict[str, Any]]

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> ViewerData:
        """Load the reports that comprise a viewer artifact directory.

        Returns:
            Validated data ready for local API requests.

        Raises:
            ViewerError: If either artifact is absent or malformed.
        """
        analysis = _load_object(artifact_dir / "analysis.json")
        calls_data = _load_object(artifact_dir / "calls.json")
        calls = calls_data.get("calls")
        if not isinstance(calls, list) or not all(
            isinstance(call, dict) for call in calls
        ):
            msg = f"calls artifact must contain a calls list: {artifact_dir / 'calls.json'}"
            raise ViewerError(msg)
        operators = analysis.get("operators")
        if not isinstance(operators, list) or not all(
            isinstance(operator, dict) for operator in operators
        ):
            msg = f"analysis artifact must contain an operators list: {artifact_dir / 'analysis.json'}"
            raise ViewerError(msg)
        return cls(analysis=analysis, calls=calls)

    def summary(self) -> dict[str, Any]:
        """Return aggregate analysis enriched with call pairing coverage."""
        coverage: dict[str, Counter[str]] = {}
        for call in self.calls:
            name = call.get("name")
            status = call.get("match_status")
            if isinstance(name, str) and isinstance(status, str):
                coverage.setdefault(name, Counter())[status] += 1
        analysis = {
            key: value for key, value in self.analysis.items() if key != "operators"
        }
        operators = []
        for operator in self.analysis["operators"]:
            name = operator.get("name")
            operator_coverage = coverage.get(name, Counter())
            operators.append(
                {
                    **operator,
                    "pairing": {
                        "sequence_paired": operator_coverage["sequence-paired"],
                        "actual_only": operator_coverage["actual-only"],
                        "projected_only": operator_coverage["projected-only"],
                    },
                },
            )
        return {**analysis, "operators": operators}

    def calls_for(
        self,
        *,
        operator: str,
        status: str,
        sort: str,
        direction: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        """Filter, sort, and paginate one operator's calls.

        Returns:
            One JSON-compatible API page containing the requested calls.

        Raises:
            ViewerError: If a query parameter is invalid.
        """
        known_operators = {
            item.get("name")
            for item in self.analysis["operators"]
            if isinstance(item, dict)
        }
        if operator not in known_operators:
            msg = f"unknown operator: {operator}"
            raise ViewerError(msg)
        if status not in _ALLOWED_STATUSES:
            msg = f"invalid status: {status}"
            raise ViewerError(msg)
        if sort not in _ALLOWED_SORTS:
            msg = f"invalid sort: {sort}"
            raise ViewerError(msg)
        if direction not in {"asc", "desc"}:
            msg = f"invalid direction: {direction}"
            raise ViewerError(msg)
        if offset < 0:
            msg = "offset must not be negative"
            raise ViewerError(msg)
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            msg = f"limit must be between 1 and {_MAX_PAGE_SIZE}"
            raise ViewerError(msg)

        filtered = [call for call in self.calls if call.get("name") == operator]
        if status != "all":
            filtered = [call for call in filtered if call.get("match_status") == status]
        present = [call for call in filtered if _sort_value(call, sort) is not None]
        missing = [call for call in filtered if _sort_value(call, sort) is None]
        present.sort(
            key=lambda call: _present_sort_value(call, sort),
            reverse=direction == "desc",
        )
        ordered = present + missing
        return {
            "operator": operator,
            "status": status,
            "sort": sort,
            "direction": direction,
            "offset": offset,
            "limit": limit,
            "total": len(ordered),
            "calls": ordered[offset : offset + limit],
        }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"invalid viewer artifact: {path}"
        raise ViewerError(msg) from error
    if not isinstance(data, dict):
        msg = f"viewer artifact must contain an object: {path}"
        raise ViewerError(msg)
    return data


def _sort_value(call: dict[str, Any], sort: str) -> int | float | None:
    if sort == "order":
        actual_index = call.get("actual_index")
        projected_index = call.get("projected_index")
        return actual_index if actual_index is not None else projected_index
    value = call.get(sort)
    return value if isinstance(value, int | float) else None


def _present_sort_value(call: dict[str, Any], sort: str) -> int | float:
    """Return a known-present sort value after missing values were separated.

    Raises:
        ViewerError: If the value was unexpectedly absent.
    """
    value = _sort_value(call, sort)
    if value is None:
        msg = f"missing sort value for {sort}"
        raise ViewerError(msg)
    return value


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve fixed static resources and a read-only artifact API."""

    data: ClassVar[ViewerData]

    def do_GET(self) -> None:
        """Handle one static resource or API request."""
        request = urlparse(self.path)
        if request.path == "/api/summary":
            self._send_json(HTTPStatus.OK, self.data.summary())
            return
        if request.path == "/api/calls":
            try:
                self._send_json(HTTPStatus.OK, self._calls_response(request.query))
            except ViewerError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        static_file = _STATIC_FILES.get(request.path)
        if static_file is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        filename, content_type = static_file
        path = Path(__file__).with_name("static") / filename
        self._send_bytes(HTTPStatus.OK, path.read_bytes(), content_type)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the viewer quiet unless its caller adds its own logging."""

    def _calls_response(self, query: str) -> dict[str, Any]:
        parameters = parse_qs(query, keep_blank_values=True)
        operator = _single_parameter(parameters, "operator")
        return self.data.calls_for(
            operator=operator,
            status=_single_parameter(parameters, "status", "all"),
            sort=_single_parameter(parameters, "sort", "actual_ms"),
            direction=_single_parameter(parameters, "direction", "desc"),
            offset=_integer_parameter(parameters, "offset", 0),
            limit=_integer_parameter(parameters, "limit", 100),
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _single_parameter(
    parameters: dict[str, list[str]],
    name: str,
    default: str | None = None,
) -> str:
    values = parameters.get(name)
    if values is None:
        if default is None:
            msg = f"missing parameter: {name}"
            raise ViewerError(msg)
        return default
    if len(values) != 1 or not values[0]:
        msg = f"invalid parameter: {name}"
        raise ViewerError(msg)
    return values[0]


def _integer_parameter(
    parameters: dict[str, list[str]],
    name: str,
    default: int,
) -> int:
    value = _single_parameter(parameters, name, str(default))
    try:
        return int(value)
    except ValueError as error:
        msg = f"invalid integer parameter: {name}"
        raise ViewerError(msg) from error


def create_server(
    artifact_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Create a local-only HTTP server for one artifact directory.

    Returns:
        A server ready to be started by its caller.

    Raises:
        ViewerError: If the requested bind address is not localhost or the
            artifacts are invalid.
    """
    if host != "127.0.0.1":
        msg = "viewer must bind to 127.0.0.1"
        raise ViewerError(msg)
    data = ViewerData.from_artifact_dir(artifact_dir)
    handler = type(
        "ArtifactViewerRequestHandler",
        (ViewerRequestHandler,),
        {"data": data},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    """Start the local viewer server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Directory containing analysis.json and calls.json",
    )
    parser.add_argument("--port", type=int, default=8000, help="Local TCP port")
    args = parser.parse_args()
    try:
        server = create_server(args.artifact_dir, port=args.port)
    except ViewerError as error:
        parser.error(str(error))
    bound_port = server.server_port
    host = "127.0.0.1"
    print(f"Performance viewer: http://{host}:{bound_port}")  # noqa: T201
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
