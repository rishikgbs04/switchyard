from switchyard.breaker import CircuitBreaker
from switchyard.embeddings import HashEmbedder
from switchyard.ratelimit import RateLimiter
from switchyard.router import score_difficulty
from switchyard.schemas import Message


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_and_half_opens():
    clk = Clock()
    b = CircuitBreaker(failure_threshold=2, cooldown_seconds=10, clock=clk)
    b.record_failure("m")
    assert b.allow("m")
    b.record_failure("m")
    assert not b.allow("m")
    clk.t = 10
    assert b.allow("m")  # probe allowed after cooldown
    b.record_success("m")
    assert b.allow("m")


def test_token_bucket_refills():
    clk = Clock()
    rl = RateLimiter(clock=clk)
    assert rl.allow("k", 60)[0]
    for _ in range(59):
        rl.allow("k", 60)
    ok, wait = rl.allow("k", 60)
    assert not ok and 0 < wait <= 1.0
    clk.t = 1.0
    assert rl.allow("k", 60)[0]


def test_embedder_paraphrase_closer_than_unrelated():
    e = HashEmbedder()
    a = e.embed("How do I reverse a list in Python?")
    b = e.embed("how to reverse a python list")
    c = e.embed("Best pizza places in Brooklyn")
    assert a @ b > 0.6
    assert a @ c < 0.3


def test_difficulty_monotonic():
    easy = score_difficulty([Message(role="user", content="hi")])
    hard = score_difficulty([Message(role="user", content="Prove and analyze the trade-offs step by step")])
    assert easy.score < hard.score
