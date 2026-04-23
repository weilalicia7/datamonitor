"""
Flask integration tests — data-source, patient, regimen + staff viewers.

Covers the read-only data-plane endpoints that the dashboard Gantt chart +
patient browser query.  All return JSON; most serialise directly from the
pandas frames cached on app_state.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    import flask_app
    return flask_app.app.test_client()


class TestDataSourceInventory:
    def test_data_source_lists_channels(self, client):
        resp = client.get("/api/data/source")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "current_channel" in body
        assert body["current_channel"] in ("synthetic", "real")
        assert "channels" in body
        # All three documented channels surface even when only one is loaded.
        for ch in ("synthetic", "real", "nhs_open"):
            assert ch in body["channels"]
            assert "available" in body["channels"][ch]


class TestPatientsList:
    def test_patients_list_reports_count_and_records(self, client):
        resp = client.get("/api/patients/list")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "count" in body and "patients" in body
        # In the default synthetic channel the count matches the in-memory df.
        if body["count"] > 0:
            first = body["patients"][0]
            # Patient_ID is the stable key across sample data.
            assert "Patient_ID" in first or "patient_id" in first


class TestRegimensList:
    def test_regimens_list_enumerates_protocols(self, client):
        resp = client.get("/api/regimens")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "count" in body and "regimens" in body
        assert isinstance(body["regimens"], list)


class TestSitesConfig:
    def test_sites_endpoint_returns_structured_list(self, client):
        resp = client.get("/api/sites")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "sites" in body and "count" in body
        assert isinstance(body["sites"], list)
        # The synthetic channel ships 5 sites.
        if body["count"] > 0:
            site = body["sites"][0]
            assert isinstance(site, dict)


class TestStaffDirectory:
    def test_staff_endpoint_returns_list(self, client):
        resp = client.get("/api/staff")
        assert resp.status_code == 200
        body = resp.get_json()
        # Allow error-shape when the workbook is missing; test only shape.
        if "error" in body:
            pytest.skip(f"staff workbook not present: {body['error']}")
        assert "staff" in body and "count" in body
        assert isinstance(body["staff"], list)
        assert body["count"] == len(body["staff"])
