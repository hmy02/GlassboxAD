from __future__ import annotations

import json
from pathlib import Path


def _patch_script_3d(script_text: str) -> str:
    """Replace the fetch('/api/data') flow with reading window.__HYDRA_DATA__."""
    # Minimal, robust patch: replace the start of hydraStartApp.
    needle = "const response = await fetch('/api/data');"
    if needle in script_text:
        script_text = script_text.replace(
            "const response = await fetch('/api/data');\n    data = await response.json();",
            "data = window.__HYDRA_DATA__;\n    if (!data) {\n        console.error('No __HYDRA_DATA__ injected');\n        return;\n    }",
        )

    # Make subsequence rendering robust when subsequences are omitted in the payload.
    subseq_line = "const subsequence = data.subsequences[node.global_idx];"
    if subseq_line in script_text:
        script_text = script_text.replace(
            subseq_line,
            "const subsequence = (data.subsequences && data.subsequences[node.global_idx]) ? data.subsequences[node.global_idx] : data.time_series.slice(node.global_idx, node.global_idx + data.win_size);",
        )
    return script_text


def build_hydra_viewer_html(payload: dict) -> str:
    """Create a self-contained HTML for Streamlit components.html."""
    root = Path(__file__).resolve().parents[1]
    template_path = root / "assets" / "viz_tool" / "templates" / "index.html"
    script_path = root / "assets" / "viz_tool" / "static" / "script_3d.js"

    html = template_path.read_text(encoding="utf-8")
    script = _patch_script_3d(script_path.read_text(encoding="utf-8"))

    # Remove the module script src and inline the patched script.
    html = html.replace('<script type="module" src="/static/script_3d.js"></script>', '')

    inject = f"""
    <script>
      window.__HYDRA_DATA__ = {json.dumps(payload)};
      // Skip landing page; we are in Streamlit
      window.showLoading = function() {{}};
      window.hideLoading = function() {{}};
      window.generateRandom = async function() {{}};
      window.handleFileUpload = async function() {{}};
      // Auto-start when module is ready
      window.__AUTO_START__ = true;
    </script>
    <script type=\"module\">\n{script}\n\n// auto-start
if (window.__AUTO_START__) {{
  if (window.hydraStartApp) {{
    window.hydraStartApp();
  }} else {{
    const t = setInterval(() => {{
      if (window.hydraStartApp) {{ clearInterval(t); window.hydraStartApp(); }}
    }}, 200);
  }}
}}
</script>
    """

    # Insert injection right before closing body
    html = html.replace("</body>", inject + "\n</body>")

    return html
