"""Precision-first detection of declared AI involvement."""

from app.stages.ai import agent_account, commit_tool, pr_tool


def test_agent_accounts_with_and_without_bot_suffix():
    assert agent_account("copilot-swe-agent[bot]") == ("copilot", "author")
    assert agent_account("Copilot") == ("copilot", "author")
    assert agent_account("coderabbitai[bot]") == ("coderabbit", "reviewer")
    assert agent_account("JulesScholler") is None and agent_account("devincool") is None and agent_account(None) is None


def test_commit_trailers_need_the_tools_own_address():
    assert commit_tool("Fix bug\n\nCo-Authored-By: Claude <noreply@anthropic.com>") == "claude"
    assert commit_tool("x\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)") == "claude"
    assert commit_tool("x\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>") == "copilot"
    assert commit_tool("x\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>") == "cursor"
    assert commit_tool("x\n\nCo-authored-by: aider (gpt-4o) <noreply@aider.chat>") == "aider"
    # people and mentions are not AI credits
    assert commit_tool("x\n\nCo-authored-by: Jules Scholler <jules@example.org>") is None
    assert commit_tool("Disable copilot suggestions in the editor config") is None
    assert commit_tool("Co-authored-by: Claude Monet <claude@monet.fr>") is None


def test_pr_descriptions_credit_agents_not_summaries():
    assert pr_tool("Adds X.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)") == "claude"
    assert pr_tool("Link to Devin run: https://app.devin.ai/sessions/abc") == "devin"
    assert pr_tool("Copilot summary: this PR updates the docs") is None
    assert pr_tool(None) is None
