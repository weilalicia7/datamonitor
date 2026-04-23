"""
Tests for /api/tuning/status + /api/tuning/run (no UI panel).

Confirms the endpoints surface manifest state correctly and that a
'random_search' POST writes a synthetic-channel manifest entry that
the boot path will (correctly) refuse to apply.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("TUNING_MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("SACT_CHANNEL", "synthetic")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import flask_app
    return flask_app.app


def test_status_empty_manifest(app, tmp_path, monkeypatch):
    monkeypatch.setenv("TUNING_MANIFEST_PATH", str(tmp_path / "manifest.json"))
    client = app.test_client()
    resp = client.get('/api/tuning/status')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['success'] is True
    assert body['present'] is False
    assert body['overrides_active'] is False


def test_run_rejects_unknown_tuner(app):
    client = app.test_client()
    resp = client.post('/api/tuning/run', json={'tuner': 'gradient_descent'})
    assert resp.status_code == 400
    assert 'tuner must be one of' in resp.get_json()['error']


def test_run_rejects_unknown_channel(app):
    client = app.test_client()
    resp = client.post('/api/tuning/run', json={
        'tuner': 'random_search', 'channel': 'production',
    })
    assert resp.status_code == 400
    assert 'channel must be' in resp.get_json()['error']


def test_run_random_search_synthetic_writes_manifest(app, tmp_path, monkeypatch):
    """End-to-end: POST a small random-search run on synthetic data,
    verify the manifest landed but overrides remain inactive."""
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setenv("TUNING_MANIFEST_PATH", str(manifest_path))
    client = app.test_client()
    resp = client.post('/api/tuning/run', json={
        'tuner': 'random_search',
        'channel': 'synthetic',
        'n_iter': 3,
        'cv_splits': 3,
    })
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body['success'] is True
    assert body['tuner'] == 'random_search'
    assert body['channel'] == 'synthetic'
    assert body['manifest']['present'] is True
    assert body['manifest']['data_channel'] == 'synthetic'
    # Synthetic-channel manifest never enables overrides.
    assert body['manifest']['overrides_active'] is False
    # Manifest file created at the expected path.
    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert 'noshow_model' in written.get('results', {})
