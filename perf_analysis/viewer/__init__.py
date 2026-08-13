# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local browser viewer for callable performance analysis artifacts."""

from perf_analysis.viewer.server import create_server, ViewerError

__all__ = ["ViewerError", "create_server"]
