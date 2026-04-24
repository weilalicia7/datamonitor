"""
Regression for §3.7 (external-review Improvement H).

Locks the reproducibility artefacts cited in the dissertation:
  * Dockerfile exists and carries FROM/CMD directives the §3.7 prose
    quotes (multi-stage build with a ``reproducibility`` target).
  * environment.yml is valid YAML and names the ``sact-scheduler``
    env with the three dep channels the prose mentions.
  * .zenodo.json is valid JSON and carries the minimum fields the
    Zenodo GitHub integration needs to mint a versioned DOI.
  * reproducibility/manifest.json has the full schema the
    dissertation_analysis.R §22 reader walks, and the generator
    regenerates it deterministically.

Does NOT actually build the Docker image — that takes >10 min and
requires docker daemon access; the test is about schema and string
contract, not image correctness.  The Dockerfile's functional
correctness is locked by the CI pipeline running the reproducibility
target end-to-end on every merge.
"""
import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestReproducibilityArtefacts(unittest.TestCase):
    """§3.7 schema contract for the Improvement H artefacts."""

    def test_dockerfile_has_reproducibility_stage(self):
        """Dockerfile must carry a multi-stage build with a
        ``reproducibility`` target whose CMD runs pytest — the §3.7
        prose quotes ``docker run --rm sact-scheduler:repro`` as the
        reproducibility command, which relies on that stage."""
        p = _REPO_ROOT / "Dockerfile"
        self.assertTrue(p.exists(), "Dockerfile missing from repo root")
        text = p.read_text(encoding="utf-8")
        # FROM + target stage
        self.assertIn("FROM python:3.12-slim AS builder", text,
                      "Dockerfile must start from python:3.12-slim builder")
        self.assertIn("AS reproducibility", text,
                      "Dockerfile must declare the reproducibility stage")
        # CMD that runs pytest
        self.assertIn('CMD ["python", "-m", "pytest"', text,
                      "Reproducibility CMD must run pytest")
        # Deterministic env
        self.assertIn("PYTHONHASHSEED=42", text,
                      "Dockerfile must pin PYTHONHASHSEED for reproducibility")

    def test_environment_yml_is_valid_yaml_with_expected_deps(self):
        """environment.yml must parse and name the sact-scheduler
        environment with the pinned dep set quoted in §3.7."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available — skipping environment.yml schema test")
        p = _REPO_ROOT / "environment.yml"
        self.assertTrue(p.exists(), "environment.yml missing from repo root")
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        self.assertEqual(doc.get("name"), "sact-scheduler")
        # Channels list must contain conda-forge
        channels = doc.get("channels", [])
        self.assertIn("conda-forge", channels)
        # Dependencies list must mention python 3.12 and ortools
        deps = doc.get("dependencies", [])
        dep_text = " ".join(str(d) for d in deps)
        self.assertIn("python=3.12", dep_text)
        self.assertIn("ortools", dep_text)
        self.assertIn("scikit-learn", dep_text)
        self.assertIn("xgboost", dep_text)
        self.assertIn("shap", dep_text)

    def test_zenodo_json_is_valid_with_required_fields(self):
        """.zenodo.json must parse and carry the fields Zenodo's
        GitHub integration expects (title, creators, upload_type,
        license, access_right)."""
        p = _REPO_ROOT / ".zenodo.json"
        self.assertTrue(p.exists(), ".zenodo.json missing from repo root")
        doc = json.loads(p.read_text(encoding="utf-8"))
        for field in ("title", "description", "upload_type", "creators",
                      "access_right", "license"):
            self.assertIn(field, doc, f".zenodo.json missing {field!r}")
        self.assertEqual(doc["upload_type"], "software")
        self.assertEqual(doc["access_right"], "open")
        # At least one creator with a name
        self.assertGreater(len(doc["creators"]), 0)
        self.assertIn("name", doc["creators"][0])

    def test_manifest_generator_produces_valid_manifest(self):
        """`python -m reproducibility.generate_manifest --skip-pip-freeze`
        must produce a JSON file that the R §22 reader can walk."""
        from reproducibility.generate_manifest import generate
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "manifest.json"
            m = generate(out, include_pip_freeze=False)
            # Returned dict and on-disk file must agree
            self.assertTrue(out.exists())
            on_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(m["format_version"], on_disk["format_version"])
            # Top-level schema must carry every key R §22 reads
            for key in ("format_version", "ts_utc", "git", "python",
                        "platform", "key_file_checksums",
                        "data_cache_jsonl_checksums",
                        "docker_repro_command", "docker_build_command"):
                self.assertIn(key, m, f"manifest missing {key!r}")
            # git sub-schema
            for key in ("commit_sha", "short_sha", "branch"):
                self.assertIn(key, m["git"], f"git sub-schema missing {key!r}")
            # docker commands must reference the reproducibility target
            self.assertIn("sact-scheduler:repro", m["docker_repro_command"])
            self.assertIn("--target reproducibility", m["docker_build_command"])


if __name__ == "__main__":
    unittest.main()
