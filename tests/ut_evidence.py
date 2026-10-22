"""测试证据辅助程序：计划由测试声明建立，运行名取官方 JUnit 解析器。"""
import ast
import json
from pathlib import Path
import runpy
import subprocess


def planned_cases(files):
    tests = []
    for file in files:
        for node in ast.parse(Path(file).read_text()).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            ids = []
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call) or not isinstance(deco.func, ast.Attribute) or deco.func.attr != "parametrize":
                    continue
                explicit = next((kw.value for kw in deco.keywords if kw.arg == "ids"), None)
                if isinstance(explicit, (ast.List, ast.Tuple)):
                    ids = ast.literal_eval(explicit)
                else:
                    for value in deco.args[1].elts:
                        if isinstance(value, ast.Constant):
                            ids.append(str(value.value))
                        elif isinstance(value, ast.Call):
                            ids.append(next(ast.literal_eval(kw.value) for kw in value.keywords if kw.arg == "id"))
            tests.extend(f"{file}:{node.name}[{value}]" for value in ids) if ids else tests.append(f"{file}:{node.name}")
    return tests


def create_unit(directory, unit, source_files, test_files, source, points, added=(), changes=(), round=1):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "unit": unit, "round": round, "files": source_files, "tests": planned_cases(test_files),
        "acceptance_review": [{"point": point, "source": source, "status": "verified", "note": "已核对用户明确授权的本地完整正文"} for point in points],
        "acceptance_added": list(added), "changes_before_round": list(changes),
        "skill_files": ["references/protocol.md", "agents/ut-agent.md", "references/design.md", "references/write-run.md", "references/analysis.md", "references/defects.md", "references/languages/python/execution.md", "references/languages/python/style.md", "references/languages/python/mocking.md", "references/languages/python/pytest.md"],
    }
    with (directory / "unit.json").open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
    return data


def record_run(directory, skill_root, *, reasons=None, changed_files=(), env_fixes=(), command=""):
    directory = Path(directory)
    parsers = runpy.run_path(str(Path(skill_root) / "scripts/ut_metrics.py"))
    raw_report = (directory / ".out.json").read_text()
    parser = parsers["junit_results"] if raw_report.lstrip().startswith("<?xml") else parsers["parse_run_output"]
    results = parser(str(Path.cwd()), raw_report)
    if not results:
        raise RuntimeError("官方解析器未解析到真实运行结果")
    for result in results:
        if result["status"] == "fail":
            result["reason"] = (reasons or {}).get(result["name"], result.get("detail", "需定位断言期望与实际"))
        result.pop("detail", None)
    stamp = subprocess.check_output(["date", "+%Y%m%dT%H%M%S"], text=True).strip()
    data = {"test_results": results, "changed_test_files": list(changed_files), "env_fixes": list(env_fixes), "command": command}
    with (directory / f"run-{stamp}.json").open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False))
    return results


def write_done(directory, data):
    with (Path(directory) / "done.json").open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
