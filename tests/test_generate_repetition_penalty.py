"""generate()'s repetition_penalty: HF-style logit adjustment for tokens already
in the sequence (prompt + generated so far) -- positive logits divided by the
penalty, negative ones multiplied by it, both making the token less likely."""
from __future__ import annotations

import torch

from food_slm.eval.generate import generate


class _ConstantLogitsModel(torch.nn.Module):
    """Returns the SAME fixed logit vector every step, regardless of input --
    isolates the repetition-penalty math from any real learned behaviour."""

    def __init__(self, logits_row: torch.Tensor, max_seq_len: int = 64) -> None:
        super().__init__()
        self.logits_row = logits_row
        self.cfg = type("Cfg", (), {"max_seq_len": max_seq_len})()
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # generate() needs next(model.parameters())

    def forward(self, input_ids: torch.Tensor):
        b, s = input_ids.shape
        logits = self.logits_row.expand(b, s, -1).clone()
        return type("Out", (), {"logits": logits})()


def test_repetition_penalty_suppresses_the_seen_argmax_token():
    # token 2 has the highest logit (positive) -- greedy would pick it forever
    # without a penalty, since this model always returns the same distribution.
    logits_row = torch.tensor([0.0, 1.0, 5.0, 0.5])
    model = _ConstantLogitsModel(logits_row)
    prompt = torch.tensor([0])

    no_penalty = generate(model, prompt, max_new_tokens=5, temperature=0.0,
                          repetition_penalty=1.0)
    # greedy, no penalty: token 2 chosen every single step
    assert no_penalty[0, 1:].tolist() == [2, 2, 2, 2, 2]

    # This fake model returns the SAME raw logits every step (no real state),
    # so the penalty must be strong enough to flip the ranking in a single
    # division, not just nudge it: 5.0/10.0=0.5 < token1's unpenalized 1.0.
    with_penalty = generate(model, prompt, max_new_tokens=5, temperature=0.0,
                            repetition_penalty=10.0)
    generated = with_penalty[0, 1:].tolist()
    assert generated != [2, 2, 2, 2, 2]
    assert generated[0] == 2  # first pick is unaffected (not yet "seen")
    assert generated[1] != 2  # second pick must move off the now-penalized token


def test_repetition_penalty_one_is_a_no_op():
    logits_row = torch.tensor([0.0, 1.0, 5.0, 0.5])
    model = _ConstantLogitsModel(logits_row)
    prompt = torch.tensor([0])
    out_default = generate(model, prompt, max_new_tokens=5, temperature=0.0)
    out_explicit = generate(model, prompt, max_new_tokens=5, temperature=0.0,
                            repetition_penalty=1.0)
    assert out_default.tolist() == out_explicit.tolist()


def test_repetition_penalty_handles_negative_logits_correctly():
    # token 1 has a NEGATIVE logit -- penalizing it must make it MORE negative
    # (multiply, not divide), i.e. even less likely, not accidentally flipped
    # toward zero/more likely.
    logits_row = torch.tensor([0.0, -5.0, -1.0, -2.0])
    model = _ConstantLogitsModel(logits_row)
    prompt = torch.tensor([1])  # token 1 (logit -5.0) is already "seen" via the prompt

    out = generate(model, prompt, max_new_tokens=1, temperature=0.0, repetition_penalty=2.0)
    # after penalty: token 1's logit -5.0*2.0=-10.0 (even less likely than
    # before); token 0 stays 0.0 (unseen, untouched) and is now the clear max
    # -- greedy must pick token 0, not re-pick the already-seen token 1.
    assert out[0, -1].item() == 0
