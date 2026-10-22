"""Compare PR reviews from REST (one request per PR) with the GraphQL query used by the reviews stage."""
import subprocess, sys, time
import app.pipeline  # noqa: F401
from app.github import GitHubClient
from app.stages.github_items import REVIEWS_QUERY, graphql_user

tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
c = GitHubClient(tokens=(tok,))

def key(number, gh_id, user, state, submitted, body):
    return (number, gh_id, (user or {}).get("login"), (user or {}).get("type"), state, submitted, body or "")

for full in sys.argv[1:]:
    owner, name = full.split("/")
    t0 = time.time(); gql, cursor, requests, prs = set(), None, 0, set()
    while True:
        d = c.graphql(REVIEWS_QUERY, {"owner": owner, "name": name, "cursor": cursor}); requests += 1
        p = d["repository"]["pullRequests"]
        for pr in p["nodes"]:
            prs.add(pr["number"])
            assert not pr["reviews"]["pageInfo"]["hasNextPage"], f"PR {pr['number']} has >100 reviews"
            for rv in pr["reviews"]["nodes"]:
                if rv["submittedAt"] and rv["databaseId"] is not None:
                    gql.add(key(pr["number"], rv["databaseId"], graphql_user(rv["author"]), rv["state"], rv["submittedAt"], rv["body"]))
        cursor = p["pageInfo"]["endCursor"]
        if not p["pageInfo"]["hasNextPage"]: break
    t_gql = time.time() - t0
    t0 = time.time(); rest = set()
    for n in sorted(prs):
        for _, items in c.paginate(f"/repos/{full}/pulls/{n}/reviews", retry_empty=False):
            for it in items:
                if it.get("submitted_at"):
                    u = it.get("user"); u = {"login": u["login"], "type": u["type"]} if u else None
                    rest.add(key(n, it["id"], u, it["state"], it["submitted_at"], it["body"]))
    t_rest = time.time() - t0
    bots = sorted({k[2] for k in rest if k[3] == "Bot"})
    print(f"{full}: {len(prs)} PRs | REST {len(rest)} reviews in {len(prs)} requests, {t_rest:.0f}s | "
          f"GraphQL {len(gql)} reviews in {requests} requests, {t_gql:.1f}s | identical: {rest == gql}")
    if rest != gql:
        print("   only REST:", sorted(rest - gql)[:3]); print("   only GraphQL:", sorted(gql - rest)[:3])
    if bots: print("   bot reviewers:", bots[:5])
