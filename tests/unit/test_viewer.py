# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Tests for the local per-operator performance viewer."""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from threading import Thread
from typing import Any, TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from perf_analysis.viewer import create_server, ViewerError

if TYPE_CHECKING:
    from collections.abc import Iterator


def _addmm_calls() -> list[dict[str, Any]]:
    return [
        {
            "name": "aten::addmm",
            "match_status": "sequence-paired",
            "actual_index": 3,
            "projected_index": 4,
            "actual_ms": 1.0,
            "projected_ms": 0.5,
            "efficiency": 0.5,
        },
        {
            "name": "aten::addmm",
            "match_status": "sequence-paired",
            "actual_index": 5,
            "projected_index": 6,
            "actual_ms": 2.0,
            "projected_ms": 1.0,
            "efficiency": 0.5,
        },
        {
            "name": "aten::addmm",
            "match_status": "projected-only",
            "actual_index": None,
            "projected_index": 7,
            "actual_ms": None,
            "projected_ms": 0.25,
            "efficiency": None,
        },
    ]


def _write_artifacts(artifact_dir: Path) -> None:
    artifact_dir.mkdir()
    (artifact_dir / "analysis.json").write_text(
        json.dumps(
            {
                "workload_name": "tiny",
                "device": "xpu",
                "hardware": {"label": "synthetic"},
                "actual_source": "unitrace",
                "t1_ms": 2.0,
                "t2_wall_ms": 3.0,
                "t2_device_ms": 2.5,
                "efficiency": 2.0 / 3.0,
                "diagnostics": ["kernel skipped"],
                "operators": [
                    {
                        "name": "aten::addmm",
                        "projected_ms": 2.0,
                        "actual_ms": 3.0,
                        "efficiency": 2.0 / 3.0,
                        "projected_calls": 2,
                        "actual_calls": 2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (artifact_dir / "calls.json").write_text(
        json.dumps(
            {
                "calls": _addmm_calls(),
            },
        ),
        encoding="utf-8",
    )


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture
def viewer_url(tmp_path: Path) -> Iterator[str]:
    """Run a viewer against a synthetic artifact directory.

    Yields:
        Base URL of the running localhost viewer.
    """
    artifact_dir = tmp_path / "artifact"
    _write_artifacts(artifact_dir)
    server = create_server(artifact_dir, port=0)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_viewer_serves_static_assets_and_pairing_coverage(viewer_url: str) -> None:
    """The dashboard and summary expose the immutable artifact state."""
    with urlopen(f"{viewer_url}/") as response:  # noqa: S310
        page = response.read().decode("utf-8")
    summary = _get_json(f"{viewer_url}/api/summary")

    assert "Performance Trace Ledger" in page
    assert "Callable performance analysis" in page
    assert summary["operators"][0]["pairing"] == {
        "sequence_paired": 2,
        "actual_only": 0,
        "projected_only": 1,
    }
    assert summary["diagnostics"] == ["kernel skipped"]


def test_viewer_filters_sorts_and_pages_calls(viewer_url: str) -> None:
    """The API orders present values first and preserves explicit call states."""
    base = f"{viewer_url}/api/calls?operator=aten%3A%3Aaddmm"
    sorted_calls = _get_json(f"{base}&sort=actual_ms&direction=desc&limit=1")
    projected_only = _get_json(f"{base}&status=projected-only&limit=100")

    assert sorted_calls["total"] == len(_addmm_calls())
    assert sorted_calls["calls"][0]["actual_index"] == max(
        call["actual_index"]
        for call in _addmm_calls()
        if call["actual_index"] is not None
    )
    assert projected_only["total"] == 1
    assert projected_only["calls"][0]["efficiency"] is None


def test_viewer_rejects_invalid_requests_and_nonlocal_binding(viewer_url: str) -> None:
    """The server rejects invalid query inputs and serves no arbitrary files."""
    with pytest.raises(HTTPError) as invalid_operator:
        urlopen(f"{viewer_url}/api/calls?operator=aten%3A%3Amissing")  # noqa: S310
    with pytest.raises(HTTPError) as traversal:
        urlopen(f"{viewer_url}/../analysis.json")  # noqa: S310

    assert invalid_operator.value.code == HTTPStatus.BAD_REQUEST
    assert traversal.value.code == HTTPStatus.NOT_FOUND
    nonlocal_host = "0.0.0." + "0"
    with pytest.raises(ViewerError, match=r"127\.0\.0\.1"):
        create_server(Path(), host=nonlocal_host)
