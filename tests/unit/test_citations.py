"""Unit tests for IEEE Citation Database and Tooltip System."""

import glob
import re

from src.frontend.citations import IEEE_REFS, cite, step


class TestCitationsDatabase:
    """Audit and regression tests for the 54 canonical thesis citations."""

    def test_canonical_reference_count_and_sequence(self):
        """Verify that canonical IDs span exactly 1 to 54 without gaps or duplicates."""
        unique_ids = sorted({v["id"] for v in IEEE_REFS.values()})
        assert len(unique_ids) == 54, f"Expected 54 unique references, got {len(unique_ids)}"
        assert unique_ids == list(range(1, 55)), "Citation IDs must strictly cover 1 to 54"

    def test_all_entries_have_required_metadata(self):
        """Verify that every reference entry contains all required academic fields."""
        required_fields = ["id", "authors", "title", "journal", "year", "used_in", "context"]
        for key, ref in IEEE_REFS.items():
            for field in required_fields:
                assert field in ref, f"Reference '{key}' missing required field '{field}'"
                assert str(ref[field]).strip() != "", f"Reference '{key}' field '{field}' cannot be empty"

    def test_cite_function_output(self):
        """Verify cite() generates correct HTML tooltips for canonical keys."""
        html = cite("who2021")
        assert "[1]" in html
        assert 'class="cite-tooltip"' in html

        html_lgb = cite("ke2017")
        assert "[17]" in html_lgb

        html_gru = cite("cho2014")
        assert "[21]" in html_gru

        html_zhang = cite("zhang2022")
        assert "[25]" in html_zhang

        html_shap = cite("lundberg2017")
        assert "[41]" in html_shap

        html_cqr = cite("romano2019")
        assert "[42]" in html_cqr

        html_arm = cite("armstrong2001")
        assert "[46]" in html_arm

        html_pi = cite("fisher2019")
        assert "[50]" in html_pi

        html_bhardwaj = cite("bhardwaj2023")
        assert "[53]" in html_bhardwaj

        html_aci = cite("gibbs2021")
        assert "[54]" in html_aci

    def test_backward_compatibility_aliases(self):
        """Verify backward compatibility aliases map to their correct canonical targets."""
        assert "[6]" in cite("hyndman2021")
        assert "[7]" in cite("zhang2017")
        assert "[28]" in cite("nguyen2024")
        assert "[49]" in cite("kang2017")

    def test_invalid_key_fallback(self):
        """Verify invalid key returns safe fallback indicator."""
        html = cite("nonexistent_ref_xyz")
        assert "[?nonexistent_ref_xyz]" in html

    def test_all_codebase_citations_exist(self):
        """Verify that every cite('...') call across python files in src/ is valid."""
        invalid_calls = []
        for fpath in glob.glob("src/**/*.py", recursive=True) + ["app.py"]:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            matches = re.findall(r"cite\([\'\"](.*?)[\'\"]\)", content)
            for m in matches:
                if m not in IEEE_REFS:
                    invalid_calls.append((fpath, m))

        assert len(invalid_calls) == 0, f"Found unregistered citation keys in codebase: {invalid_calls}"

    def test_step_function(self):
        """Verify pipeline step circled number formatting."""
        assert "①" in step(1)
        assert "⑤" in step(5)
        assert "⑦" in step(7)

    def test_all_cite_in_markdown_have_unsafe_allow_html(self):
        """Verify that every st.markdown call containing cite(...) sets unsafe_allow_html=True."""
        import ast

        missing = []
        for fpath in glob.glob("src/**/*.py", recursive=True) + ["app.py"]:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            if "cite(" not in content:
                continue
            tree = ast.parse(content, filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    is_st_md = isinstance(node.func, ast.Attribute) and node.func.attr == "markdown"
                    if is_st_md:
                        has_cite = any("cite(" in ast.unparse(arg) for arg in node.args)
                        if has_cite:
                            has_unsafe = any(
                                kw.arg == "unsafe_allow_html" and getattr(kw.value, "value", None) is True
                                for kw in node.keywords
                            )
                            if not has_unsafe:
                                missing.append(f"{fpath}:{node.lineno}")

        assert len(missing) == 0, f"Found st.markdown with cite() missing unsafe_allow_html=True: {missing}"
