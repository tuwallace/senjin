"""The job's requirements as a file a person can edit, and quick ways to steer them.

senjin-review writes `<candidate-dir>/requirements.yaml` the first time and reads it back after
that, so a run does not depend on what the model happened to list this time. Edit it to drop a
requirement (`skip: true`), change `kind`, or raise a `weight`. `--focus` does the same from the
command line for one run.
"""

from pathlib import Path

import yaml

from senjin.review.generate import ReviewError
from senjin.review.schema import Requirement
from senjin.review.skills import weight_of, words

FOCUS_WEIGHT = 6.0  # three times a must-have's default of 2

HEADER = """\
# The job's requirements, written by senjin-review from job_description.md. Edit freely:
#   skip: true     ignore a requirement (for example one no question can test)
#   kind           must_have or nice_to_have
#   weight         how much it matters when choosing and writing questions (must_have 2, other 1)
# `senjin-review` reads this file instead of asking the model again; use --refresh-requirements to
# list them afresh. `senjin-review --score` checks questions.yaml against it without any model.
"""


def save_requirements(
    path: Path, requirements: list[Requirement], weights: dict[str, float]
) -> None:
    items = [
        {
            "name": r.name,
            "kind": r.kind,
            "weight": weight_of(r, weights),
            "evidence": r.evidence,
            "skip": False,
        }
        for r in requirements
    ]
    path.write_text(HEADER + yaml.safe_dump(items, sort_keys=False, width=100), encoding="utf-8")


def load_requirements(path: Path) -> tuple[list[Requirement], dict[str, float]]:
    """The requirements in the file, without skipped ones, and their weights."""
    try:
        items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        found: list[Requirement] = []
        weights: dict[str, float] = {}
        for item in items:
            if item.get("skip"):
                continue
            requirement = Requirement(
                name=str(item["name"]).strip(),
                kind=item.get("kind", "must_have"),
                evidence=str(item.get("evidence", "")),
            )
            found.append(requirement)
            if "weight" in item:
                weights[requirement.name] = float(item["weight"])
    except (KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError) as exc:
        raise ReviewError(f"{path} is not a valid requirements file: {exc}") from None
    if not found:
        raise ReviewError(f"{path} has no requirements (all skipped?)")
    return found, weights


def apply_focus(
    requirements: list[Requirement], weights: dict[str, float], terms: list[str]
) -> tuple[list[Requirement], dict[str, float], dict[str, list[str]]]:
    """Make requirements that match a --focus term must-haves with triple weight.

    A term matches a requirement if all its words are in the requirement's name. A term that
    matches nothing is an error, so a typo does not silently do nothing.
    """
    result = list(requirements)
    weights = dict(weights)
    matched: dict[str, list[str]] = {}
    for term in terms:
        wanted = set(words(term))
        hits = [
            i
            for i, r in enumerate(result)
            if (wanted and wanted <= set(words(r.name))) or term.strip().lower() in r.name.lower()
        ]
        if not hits:
            names = "; ".join(r.name for r in requirements)
            raise ReviewError(f"--focus {term!r} matches no requirement. They are: {names}")
        for i in hits:
            result[i] = result[i].model_copy(update={"kind": "must_have"})
            weights[result[i].name] = max(weights.get(result[i].name, 0.0), FOCUS_WEIGHT)
        matched[term] = [result[i].name for i in hits]
    return result, weights, matched
