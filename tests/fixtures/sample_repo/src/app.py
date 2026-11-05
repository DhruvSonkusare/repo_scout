"""A small program for the RepoGuide tests."""


def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


def shout(name: str) -> str:
    """Return a loud greeting."""
    return greet(name).upper()


def repeat(name: str, times: int) -> list[str]:
    """Return the same greeting several times."""
    return [greet(name) for _ in range(times)]


if __name__ == "__main__":
    print(greet("world"))
