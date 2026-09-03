import torch

from exllamav3.generator.job import Job


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
