"""Manual check of the GitHub client: print the top repos for a search and the start of the first README."""

import argparse
import asyncio
import sys
from pathlib import Path

# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.finder.github import GitHubClient, GitHubError  # noqa: E402

README_PREVIEW_CHARS = 300


async def run(query: str, language: str | None) -> None:
    """Search GitHub and print the results."""
    async with GitHubClient() as github:
        repos = await github.search_repos(query, language=language)
        for repo in repos:
            pushed = repo.pushed_at.date() if repo.pushed_at else "unknown"
            print(f"{repo.stars:>7}  {repo.full_name}  [{repo.language}, {repo.license}, pushed {pushed}]")
        if repos:
            readme = await github.get_readme(repos[0].full_name)
            print(f"\nREADME of {repos[0].full_name} ({len(readme)} chars):\n{readme[:README_PREVIEW_CHARS]}")


def main() -> None:
    """Read the search from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="pdf table extraction")
    parser.add_argument("--language")
    args = parser.parse_args()
    # READMEs often have emoji, which a Windows console's encoding can't print and would crash on.
    sys.stdout.reconfigure(errors="replace")
    try:
        asyncio.run(run(args.query, args.language))
    except GitHubError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
