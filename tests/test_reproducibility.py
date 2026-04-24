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

    def test_every_citep_key_has_matching_bibitem(self):
        """Regression for external-review point 2 (the Taskesen
        citation was flagged as missing because the `\\bibitem` was
        orphaned out of alphabetical order and the reviewer couldn't
        find it on a visual scan).  Catches the simpler failure mode
        where a `\\citep{key}` exists in main.tex with no matching
        `\\bibitem{key}` at all — which would trigger a `[?]`
        placeholder in the rendered PDF bibliography."""
        import re

        tex_path = _REPO_ROOT.parent / "dissertation" / "main.tex"
        if not tex_path.exists():
            self.skipTest("dissertation/main.tex not present (outside repo)")
        text = tex_path.read_text(encoding="utf-8")

        # Collect every citation key used in the prose
        cited: set = set()
        for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    cited.add(key)

        # Collect every bibliography entry key
        defined: set = set()
        for m in re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", text):
            defined.add(m.group(1).strip())

        missing = sorted(cited - defined)
        self.assertFalse(
            missing,
            f"{len(missing)} \\citep key(s) have no matching "
            f"\\bibitem in main.tex: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}",
        )

        # Sanity: the Taskesen entries flagged by the reviewer must
        # both resolve.  Locks the fix for point 2 so a future edit
        # cannot silently re-orphan them.
        self.assertIn("taskesen2021statistical", defined,
                      "Taskesen FAccT 2021 bibitem key missing")
        self.assertIn("taskesen2021aistats", defined,
                      "Taskesen AISTATS 2021 bibitem key missing "
                      "(added per external-review point 2)")

    def test_every_ref_target_has_matching_label(self):
        """Regression for external-review point 3 (§5.6.5 cited
        §4.5.1 for DRO fairness penalties, but §4.5.1 is about the
        basic CP-SAT implementation).  Catches the failure mode where
        a `\\ref{key}`/`\\autoref{key}`/`\\eqref{key}` exists with no
        matching `\\label{key}` — which would render `??` in the PDF.

        Also asserts specific labels introduced for the §5.6.5 fix
        actually resolve, so a future edit cannot silently revert the
        repair.
        """
        import re

        tex_path = _REPO_ROOT.parent / "dissertation" / "main.tex"
        if not tex_path.exists():
            self.skipTest("dissertation/main.tex not present (outside repo)")
        text = tex_path.read_text(encoding="utf-8")

        # Collect every cross-reference target used in the prose
        referenced: set = set()
        # \ref{x}, \autoref{x}, \eqref{x}, \pageref{x}, \cref{x},
        # \Cref{x} all expand to a label lookup
        ref_pattern = re.compile(
            r"\\(?:ref|autoref|eqref|pageref|cref|Cref)\{([^}]+)\}",
            re.IGNORECASE,
        )
        for m in ref_pattern.finditer(text):
            # Multi-key refs are rare but possible: \cref{a,b,c}
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    referenced.add(key)

        # Collect every \label{key} in the prose
        defined: set = set()
        for m in re.finditer(r"\\label\{([^}]+)\}", text):
            defined.add(m.group(1).strip())

        missing = sorted(referenced - defined)
        self.assertFalse(
            missing,
            f"{len(missing)} \\ref target(s) have no matching "
            f"\\label in main.tex: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}",
        )

        # Specific anchors the §5.6.5 fix relies on — the DRO fairness
        # label must exist and be the target the fairness-mitigation
        # prose cites.
        self.assertIn("sec:dro-fairness", defined,
                      "sec:dro-fairness label missing — §5.6.5 "
                      "fix for external-review point 3 relies on it")
        self.assertIn("sec:fairness-mitigation", defined,
                      "sec:fairness-mitigation label missing — §5.6.5 "
                      "prose cannot self-reference its own subsection")


if __name__ == "__main__":
    unittest.main()
