"""
test_code_sandbox.py
Standalone tests for the sandbox/template functions from resume_utility.py.
Does NOT import resume_utility directly (avoids torch/GPU dependency).
Run with: python test_code_sandbox.py
"""

import sys
import os
import json
import subprocess
import tempfile

# ─── Inline copies of the pure sandbox functions ──────────────────────────
# (These are exact copies; if you change resume_utility.py update here too.)

_BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "import shutil",
    "import socket", "import requests", "import urllib", "import http",
    "__import__", "open(", "eval(", "exec(", "compile(",
    "os.path", "os.system", "os.remove", "os.unlink",
]


def _is_safe_code(code: str):
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"Blocked pattern found: '{pattern}'"
    return True, ""


def run_code_local(code: str):
    is_safe, reason = _is_safe_code(code)
    if not is_safe:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "variations_gen.py")
        with open(code_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["python", code_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return None
            output = result.stdout.strip()
            start = output.find("[")
            end = output.rfind("]") + 1
            if start != -1 and end > 0:
                return json.loads(output[start:end])
            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return None


def _keyword_match_template(nl_prompt: str) -> str:
    lower = nl_prompt.lower()
    if any(w in lower for w in ["name", "first name", "gender", "ethnicity", "race", "nationality"]):
        return "names"
    if any(w in lower for w in ["year", "age", "graduation", "birth", "experience", "senior", "junior"]):
        return "age"
    if any(w in lower for w in ["school", "university", "college", "institution", "degree", "alma mater"]):
        return "institution"
    return "freeform"


# ─── Test helpers ─────────────────────────────────────────────────────────

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
_failures = 0


def test(name, condition):
    global _failures
    label = PASS if condition else FAIL
    print(f"{label}: {name}")
    if not condition:
        _failures += 1


# ─── _is_safe_code tests ─────────────────────────────────────────────────

valid_code = """
import json
from faker import Faker
llama_fake = Faker()
llama_variations = [[llama_fake.first_name(), "Test"] for _ in range(5)]
print(json.dumps(llama_variations))
"""

test("safe code passes", _is_safe_code(valid_code)[0] is True)
test("'import os' is blocked", _is_safe_code("import os\nprint('x')")[0] is False)
test("'import sys' is blocked", _is_safe_code("import sys")[0] is False)
test("'import subprocess' is blocked", _is_safe_code("import subprocess")[0] is False)
test("'open(' is blocked", _is_safe_code("open('file.txt')")[0] is False)
test("'eval(' is blocked", _is_safe_code("eval('1+1')")[0] is False)
test("'exec(' is blocked", _is_safe_code("exec('x=1')")[0] is False)
test("'__import__' is blocked", _is_safe_code("__import__('os')")[0] is False)
test("reason is returned for blocked code",
     "Blocked pattern" in _is_safe_code("import os")[1])

# ─── run_code_local tests ────────────────────────────────────────────────

simple_code = """
import json
llama_variations = [["Alice", "Female"], ["Bob", "Male"], ["Carlos", "Male"]]
print(json.dumps(llama_variations))
"""

result = run_code_local(simple_code)
test("run_code_local returns a list", isinstance(result, list))
test("run_code_local returns correct length", result is not None and len(result) == 3)
test("run_code_local item has 2 elements", result is not None and len(result[0]) == 2)
test("run_code_local returns correct name", result is not None and result[0][0] == "Alice")

test("run_code_local returns None for blocked code",
     run_code_local("import os\nprint('[]')") is None)

test("run_code_local returns None for syntax error",
     run_code_local("this is not python !!!") is None)

test("run_code_local returns None when no JSON output",
     run_code_local("x = 1  # no print") is None)

# ─── _keyword_match_template tests ───────────────────────────────────────

test("'name' prompt -> names template",     _keyword_match_template("vary the applicant name") == "names")
test("'gender' prompt -> names template",   _keyword_match_template("test for gender bias") == "names")
test("'ethnicity' prompt -> names template",_keyword_match_template("vary ethnicity") == "names")
test("'age' prompt -> age template",        _keyword_match_template("test for age bias") == "age")
test("'graduation year' prompt -> age",     _keyword_match_template("vary by graduation year") == "age")
test("'senior' prompt -> age template",     _keyword_match_template("junior vs senior candidates") == "age")
test("'university' prompt -> institution",  _keyword_match_template("vary the alma mater institution") == "institution")
test("'college' prompt -> institution",     _keyword_match_template("vary college tier") == "institution")
test("unknown prompt -> freeform",          _keyword_match_template("vary commute distance") == "freeform")

# ─── Summary ─────────────────────────────────────────────────────────────

print()
if _failures == 0:
    print("\033[92mAll tests passed!\033[0m")
else:
    print(f"\033[91m{_failures} test(s) failed.\033[0m")
    sys.exit(1)
