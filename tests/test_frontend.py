"""Tests that the frontend is served and only calls API routes that exist."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import STATIC_DIR, app

client = TestClient(app)
APP_JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
STYLE_CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")


def test_the_page_is_served_at_the_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "RepoScout" in response.text
    assert "/static/app.js" in response.text
    assert "/static/style.css" in response.text


@pytest.mark.parametrize(("name", "kind"), [("app.js", "javascript"), ("style.css", "text/css")])
def test_static_files_are_served(name: str, kind: str) -> None:
    response = client.get(f"/static/{name}")

    assert response.status_code == 200
    assert kind in response.headers["content-type"]


def test_files_outside_the_static_folder_are_not_served() -> None:
    assert client.get("/static/../.env").status_code == 404
    assert client.get("/static/%2e%2e/.env").status_code == 404


def test_the_root_page_stays_out_of_the_api_docs() -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_every_cdn_script_is_pinned_and_integrity_checked() -> None:
    tags = re.findall(r"<(?:script|link)[^>]*cdn\.jsdelivr\.net[^>]*>", INDEX_HTML)

    assert len(tags) == 4
    for tag in tags:
        assert re.search(r"@\d+\.\d+\.\d+/", tag)
        assert 'integrity="sha384-' in tag
        assert 'crossorigin="anonymous"' in tag


def test_every_element_the_script_looks_up_is_on_the_page() -> None:
    ids_used = set(re.findall(r"""\$\("([\w-]+)"\)""", APP_JS))
    ids_used |= set(re.findall(r"""setText\("([\w-]+)\"""", APP_JS))
    ids_on_page = set(re.findall(r'\bid="([\w-]+)"', INDEX_HTML))

    assert ids_used
    assert ids_used - ids_on_page == set()


def test_every_api_path_the_script_calls_exists() -> None:
    called = {re.sub(r"\$\{[^}]+\}", "{id}", path).split("?")[0] for path in re.findall(r"""[`"](/api/[^`"]*)[`"]""", APP_JS)}
    routes = {re.sub(r"\{[^}]+\}", "{id}", path) for path in client.get("/openapi.json").json()["paths"]}

    assert len(called) >= 12
    assert called - routes == set()


def test_markdown_is_always_sanitized() -> None:
    assert "DOMPurify.sanitize(marked.parse(text), SANITIZE_OPTIONS)" in APP_JS
    assert APP_JS.count("innerHTML") == 1


def test_the_static_folder_holds_the_three_frontend_files() -> None:
    assert {path.name for path in Path(STATIC_DIR).iterdir()} == {"index.html", "style.css", "app.js"}


def _style_without_reduced_motion() -> str:
    return STYLE_CSS.split("@media (prefers-reduced-motion: reduce)")[0]


def test_only_transform_and_opacity_are_transitioned() -> None:
    transitions = re.findall(r"transition:\s*([^;]+);", _style_without_reduced_motion())

    assert transitions
    for transition in transitions:
        if transition.strip() == "none":
            continue
        for part in transition.split(","):
            assert part.split()[0] in {"transform", "opacity"}, transition


def test_only_transform_and_opacity_are_animated() -> None:
    frames = re.findall(r"@keyframes [\w-]+ \{(.*?)\n\}", STYLE_CSS, re.DOTALL)
    properties = {name for frame in frames for name in re.findall(r"([\w-]+):", frame)}

    assert frames
    assert properties == {"opacity", "transform"}


def test_motion_settings_come_from_the_tokens() -> None:
    for token in ("--ease:", "--fast: 150ms", "--normal: 250ms"):
        assert token in STYLE_CSS
    for rule in re.findall(r"transition:\s*([^;]+);", _style_without_reduced_motion()):
        if rule.strip() != "none":
            assert re.search(r"var\(--(fast|normal|slow)\)", rule), rule
            assert "var(--ease)" in rule, rule


def test_reduced_motion_turns_animation_off() -> None:
    block = STYLE_CSS.split("@media (prefers-reduced-motion: reduce)")[1]

    assert "animation: none !important" in block
    assert "transition: none !important" in block
    assert "prefers-reduced-motion: reduce" in APP_JS


def test_config_tells_the_frontend_about_demo_mode() -> None:
    assert client.get("/api/config").json() == {"demo_mode": False}


def test_enter_in_the_password_box_submits_it_rather_than_cancelling() -> None:
    # Pressing Enter in a form uses its first submit button, so Cancel mustn't be one.
    dialog = INDEX_HTML.split('id="password-dialog"')[1].split("</dialog>")[0]

    assert re.findall(r'<button[^>]*type="submit"[^>]*>([^<]+)<', dialog) == ["Continue"]
