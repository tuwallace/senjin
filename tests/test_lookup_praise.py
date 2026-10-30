import pytest

from senjin.interview.lookup import relevant_lines
from senjin.interview.praise import strip_praise

JD = """### **Key Responsibilities**
- **DevSecOps:** Maintain CI/CD pipelines and cloud infrastructure (AWS/Azure GovCloud).
- **Accessibility:** Champion Section 508 and WCAG 2.1 AA compliance.
- Competitive base salary ($140,000 - $185,000, dependent on experience).
"""


def test_relevant_lines_finds_the_cloud_line():
    assert "AWS/Azure GovCloud" in relevant_lines(JD, "is the solution Google-based or AWS.")[0]


def test_relevant_lines_finds_salary_and_ignores_filler_questions():
    assert "$140,000" in relevant_lines(JD, "What is the salary?")[0]
    assert relevant_lines(JD, "Do you have anything else?") == []


@pytest.mark.parametrize(
    ("said", "kept"),
    [
        ("Great! Let's start with your background.", "Let's start with your background."),
        (
            "Thank you for your detailed explanation. The technical part is done.",
            "The technical part is done.",
        ),
        ("Great insights. How would you scale it?", "How would you scale it?"),
        ("That's a really good point. What about retries?", "What about retries?"),
        ("Great, let's move on to testing.", "Let's move on to testing."),
        (
            "Thank you for your time today. The team will follow up.",
            "Thank you for your time today. The team will follow up.",
        ),
        ("Good morning. I'm Senjin.", "Good morning. I'm Senjin."),
        ("Understood. How would you scale it?", "Understood. How would you scale it?"),
        ("Great!", "Great!"),
    ],
)
def test_strip_praise(said, kept):
    assert strip_praise(said) == kept


def test_markdown_is_not_spoken():
    from senjin.interview.praise import PraiseFilter

    f = PraiseFilter()
    said = f.feed("It mentions **AWS GovCloud** or `Azure` here. ## Next, more.") + f.flush()
    assert "*" not in said and "`" not in said and "#" not in said
    assert "AWS GovCloud" in said


def test_vocabulary_hint_keeps_names_and_drops_ordinary_capitalised_words():
    from senjin.audio.vocab import vocabulary_hint

    text = "The lead worked at Paratus Solutions with the team, using Next.js on Databricks."
    hint = vocabulary_hint(text)
    assert "Paratus" in hint and "Databricks" in hint and "Next.js" in hint
    assert "The" not in hint and "using" not in hint.lower()  # ordinary words that start sentences
    assert vocabulary_hint("nothing special here") == ""
