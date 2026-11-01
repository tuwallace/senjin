import json

from senjin.llm.decide import OllamaDecider, decider_for


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a):
        return json.dumps(self.payload).encode()


def test_no_model_means_no_decider():
    assert decider_for("") is None and decider_for("  ") is None
    assert isinstance(decider_for("nimble"), OllamaDecider)


def test_reads_the_yes_no_probability(monkeypatch):
    sent = {}

    def fake_open(request, timeout):
        sent.update(json.loads(request.data), url=request.full_url)
        return Response({"answers": {"done": {"type": "noul", "noul": 0.93}}})

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    decider = OllamaDecider("nimble", host="http://h:1/")
    assert decider._call("done", {"candidate_said": "no"}, "Done?") == 0.93
    assert sent["url"] == "http://h:1/v1/systemone" and sent["model"] == "nimble"
    assert sent["questions"]["done"] == {"type": "noul", "instructions": "Done?"}


def test_any_failure_returns_none(monkeypatch):
    def refuse(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert OllamaDecider("nimble")._call("done", {}, "x") is None
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda r, timeout: Response({"answers": {"done": {"noul": 7}}})
    )
    assert OllamaDecider("nimble")._call("done", {}, "x") is None  # not a probability
    monkeypatch.setattr("urllib.request.urlopen", lambda r, timeout: Response({"error": "x"}))
    assert OllamaDecider("nimble")._call("done", {}, "x") is None


def test_config_reads_the_decision_model():
    from senjin.config import InterviewConfig

    assert InterviewConfig.from_env({}).decision_model == ""
    assert (
        InterviewConfig.from_env({"SENJIN_DECISION_MODEL": " nimble "}).decision_model == "nimble"
    )
