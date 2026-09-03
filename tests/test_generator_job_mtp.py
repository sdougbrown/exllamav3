from types import SimpleNamespace

import torch

from exllamav3.generator import job as job_module
from exllamav3.generator.job import Job
from exllamav3.util.tensor import SeqTensor


def test_invalidate_mtp_carry_clears_job_and_all_sequences():
    job = Job(
        [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])],
        max_new_tokens = 2,
    )
    job.mtp_last_hidden = torch.ones((1, 1, 4))
    for sequence in job.sequences:
        sequence.mtp_carry_hidden = torch.ones((1, 1, 4))

    job._invalidate_mtp_carry()

    assert job.mtp_last_hidden is None
    assert all(sequence.mtp_carry_hidden is None for sequence in job.sequences)


def test_banned_string_rewind_invalidates_mtp_carry(monkeypatch):
    job = Job(
        torch.tensor([[1, 1]]), max_new_tokens = 8,
        banned_strings = ["bad"], token_healing = False,
    )
    job.generator = SimpleNamespace(
        num_draft_tokens = 3,
        tokenizer = SimpleNamespace(get_id_to_piece_list = lambda _decode: ["", "ok", "bad"]),
    )
    job.max_rq_tokens = 100
    job.held_text = ""
    job.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
    job.held_k_tokens = SeqTensor((1, 0, 0), dtype = torch.long, seq_dim = 1)
    job.held_k_probs = SeqTensor((1, 0, 0), dtype = torch.float, seq_dim = 1)
    job.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
    job.held_logits = SeqTensor((1, 0, 0), dtype = torch.float, seq_dim = 1)
    job.mtp_last_hidden = torch.ones((1, 1, 4))
    page = SimpleNamespace(kv_position = 1, can_revert = False)
    for sequence in job.sequences:
        sequence.kv_position = 1
        sequence.allocated_pages = [page]
        sequence.mtp_carry_hidden = torch.ones((1, 1, 4))
    monkeypatch.setattr(job_module.ext, "partial_strings_match", lambda *_args: 0)

    results = []
    job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)

    assert job.checkpoint_rewound
    assert job.mtp_last_hidden is None
    assert all(sequence.mtp_carry_hidden is None for sequence in job.sequences)
    assert results[0]["suppressed_text"] == "bad"
