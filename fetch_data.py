import argparse
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://codeforces.com/api"


def call_api(method, **params):
    url = f"{API}/{method}"

    for attempt in range(1, 4):
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            payload = response.json()
            if payload["status"] == "OK":
                return payload["result"]
            raise RuntimeError(f"{method} failed: {payload.get('comment')}")

        print(f"  attempt {attempt}: HTTP {response.status_code}, retrying...")
        time.sleep(2 * attempt)

    raise RuntimeError(f"{method} failed after 3 attempts")


def contest_division(name):
    lowered = name.lower()

    for label in [
        "div. 1 + div. 2",
        "div. 1",
        "div. 2",
        "div. 3",
        "div. 4"
    ]:
        if label in lowered:
            return label

    if "educational" in lowered:
        return "educational"

    if "global" in lowered:
        return "global"

    return "other"


def index_to_number(index):
    first_letter = index[0].upper()

    if not first_letter.isalpha():
        return None

    return ord(first_letter) - ord("A") + 1


def fetch_problems():
    print("Fetching problemset...")
    result = call_api("problemset.problems")
    problems = result["problems"]
    statistics = result["problemStatistics"]

    solved_lookup = {
        (s["contestId"], s["index"]): s["solvedCount"]
        for s in statistics
    }

    print("Fetching contest list...")
    contests = call_api("contest.list")
    contest_lookup = {c["id"]: c for c in contests}

    rows = []

    for problem in problems:
        contest_id = problem.get("contestId")

        if contest_id is None:
            continue

        contest = contest_lookup.get(contest_id, {})
        start_seconds = contest.get("startTimeSeconds")

        rows.append(
            {
                "contest_id": contest_id,
                "index": problem["index"],
                "index_num": index_to_number(problem["index"]),
                "name": problem["name"],
                "rating": problem.get("rating"),
                "points": problem.get("points"),
                "tags": "|".join(problem.get("tags", [])),
                "n_tags": len(problem.get("tags", [])),
                "solved_count": solved_lookup.get(
                    (contest_id, problem["index"])
                ),
                "contest_name": contest.get("name", ""),
                "division": contest_division(contest.get("name", "")),
                "year": (
                    time.gmtime(start_seconds).tm_year
                    if start_seconds
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def fetch_submissions(handle):
    print(f"Fetching submissions for '{handle}'...")
    result = call_api("user.status", handle=handle)

    rows = []

    for submission in result:
        problem = submission["problem"]

        if problem.get("contestId") is None:
            continue

        rows.append(
            {
                "contest_id": problem["contestId"],
                "index": problem["index"],
                "name": problem["name"],
                "rating": problem.get("rating"),
                "tags": "|".join(problem.get("tags", [])),
                "verdict": submission.get("verdict"),
                "language": submission.get("programmingLanguage"),
                "time": submission["creationTimeSeconds"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Download Codeforces data."
    )

    parser.add_argument(
        "--handle",
        default=None,
        help="Your Codeforces handle. Needed for recommend.py, optional otherwise.",
    )

    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    problems = fetch_problems()
    problems.to_csv("data/problems.csv", index=False)

    rated = problems["rating"].notna().sum()

    print("\nSaved data/problems.csv")
    print(f"  {len(problems):,} problems total")
    print(
        f"  {rated:,} of them have a difficulty rating "
        f"(these train the model)"
    )
    print(
        f"  difficulty ranges from {problems['rating'].min():.0f} "
        f"to {problems['rating'].max():.0f}"
    )

    if args.handle:
        submissions = fetch_submissions(args.handle)
        submissions.to_csv("data/submissions.csv", index=False)

        solved = submissions[submissions["verdict"] == "OK"]
        unique_solved = solved.drop_duplicates(
            subset=["contest_id", "index"]
        )

        print("\nSaved data/submissions.csv")
        print(f"  {len(submissions):,} submissions")
        print(f"  {len(unique_solved):,} distinct problems solved")
    else:
        print("\nNo --handle given, so we skipped your submission history.")
        print(
            "Re-run with --handle YOUR_HANDLE to enable recommend.py"
        )


if __name__ == "__main__":
    main()