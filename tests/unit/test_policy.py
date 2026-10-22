"""验收依据：技术方案§命令执行与授权；完整 argv、实际 cwd、会话隔离。"""
from coding_agent.policy import Policy


def test_execute_mode_requires_authorization(tmp_path):
    assert Policy().evaluate(["python", "-V"], tmp_path) == "ask"


def test_session_authorization_matches_complete_argv_and_cwd(tmp_path):
    policy = Policy()
    argv = ["python", "-m", "pytest", "-q"]
    policy.approve_for_session(argv, tmp_path)
    assert policy.evaluate(argv, tmp_path) == "allow"
    assert policy.evaluate(argv, tmp_path) == "allow"
    assert policy.evaluate(["python", "-m", "pytest"], tmp_path) == "ask"
    assert policy.evaluate(["python", "-c", "print(1)"], tmp_path) == "ask"
    assert policy.evaluate([*argv, "--collect-only"], tmp_path) == "ask"
    assert policy.evaluate(argv, tmp_path / "other") == "ask"


def test_one_time_approval_is_consumed_only_by_matching_command(tmp_path):
    policy = Policy()
    argv = ["python", "-V"]
    policy.approve_once(argv, tmp_path)
    assert policy.evaluate(["python", "-c", "print(1)"], tmp_path) == "ask"
    assert policy.evaluate(argv, tmp_path / "other") == "ask"
    assert policy.evaluate(argv, tmp_path) == "allow"
    assert policy.evaluate(argv, tmp_path) == "ask"


def test_authorization_uses_resolved_cwd(tmp_path):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(cwd, target_is_directory=True)
    policy = Policy()
    policy.approve_for_session(["python", "-V"], cwd)
    assert policy.evaluate(["python", "-V"], alias) == "allow"


def test_new_session_does_not_inherit_approvals(tmp_path):
    policy = Policy()
    argv = ["python", "-V"]
    policy.approve_once(argv, tmp_path)
    policy.approve_for_session(argv, tmp_path)
    assert Policy().evaluate(argv, tmp_path) == "ask"


def test_analysis_denies_commands_even_after_approval(tmp_path):
    policy = Policy(mode="analysis")
    argv = ["python", "-V"]
    policy.approve_once(argv, tmp_path)
    policy.approve_for_session(argv, tmp_path)
    assert policy.evaluate(argv, tmp_path) == "deny"
