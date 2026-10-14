import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"


def fetch_issues(repo, max_issues=1500):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "closed", "per_page": 100,
              "sort": "created", "direction": "desc"}
    issues, page = [], 1

    while len(issues) < max_issues:
        params["page"] = page
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)

        # Wait out the rate limit instead of crashing
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers["X-RateLimit-Reset"])
            time.sleep(max(reset - time.time(), 0) + 5)
            continue

        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        for it in batch:
            if "pull_request" in it:      # the issues endpoint also returns PRs
                continue
            issues.append({
                "id": it["number"],
                "title": it["title"],
                "body": it["body"] or "",
                "labels": [l["name"] for l in it["labels"]],
                "created_at": it["created_at"],
            })                            # no usernames or emails kept
        page += 1
        time.sleep(0.5)

    return issues[:max_issues]


if __name__ == "__main__":
    repo = sys.argv[1]                    # e.g. scikit-learn/scikit-learn
    data = fetch_issues(repo)
    out = f"data/{repo.replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} issues to {out}")