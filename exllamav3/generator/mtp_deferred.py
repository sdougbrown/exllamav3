"""Default-off, single-request target-first MTP prefill experiment."""
import torch


def validate_deferred_job(generator, job):
    g, j = generator, job
    supported = (
        g.mtp_draft and type(g.draft_model).__name__ == "Qwen4ExpMTPModel"
        and not g.model.loaded_tp and not g.draft_model.loaded_tp
        and g.max_batch_size == 1 and not g.active_jobs and not g.pending_jobs
        and g.pagetable.cpu_tier is None and not g.recurrent_cache
        and all(p.kv_position == 0 and p.ref_count == 0 for p in g.pagetable.all_pages)
        and len(j.sequences) == 1
        and all(s.kv_position == 0 and s.mtp_carry_hidden is None for s in j.sequences)
        and j.mtp_last_hidden is None
        and j.generator is None and j._mtp_deferred is None
        and not (j.embeddings or j.banned_strings or j.filters or j.is_requeued)
        and j.prefix_token is None
        and j.orig_max_rq_tokens is None
    )
    if not supported:
        raise ValueError("deferred MTP requires a fresh uncached single-sequence Qwen4Exp job, "
                         "batch1, layer split, no CPU cache, multimodal, filters, healing or requeue")


class DeferredMTPPrefill:
    """Own full exported stream stacks until ordered, same-chunk draft replay."""
    def __init__(self, prompt_end):
        if prompt_end < 1:
            raise ValueError("deferred MTP requires at least two prompt tokens")
        self.prompt_end = prompt_end
        self.target_end = self.draft_end = 0
        self.chunks = []
        self.complete = self.failed = False
        self.retained_peak_bytes = 0

    def append(self, start, end, hidden):
        if (self.failed or self.complete or start != self.target_end or not start < end <= self.prompt_end
                or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != end - start):
            raise RuntimeError("invalid deferred MTP target chunk/position")
        # Exports may alias a module's transient output; retain independent device storage.
        owned = hidden.clone()
        self.chunks.append((start, end, owned))
        self.retained_peak_bytes += owned.numel() * owned.element_size()
        self.target_end = end

    def drain(self, prefill):
        if self.failed or self.complete or self.target_end != self.prompt_end:
            raise RuntimeError("deferred MTP requires complete target history before draft replay")
        carry = None
        try:
            for start, end, hidden in self.chunks:
                if start != self.draft_end:
                    raise RuntimeError("invalid deferred MTP draft position")
                if carry is None:
                    carry = torch.zeros_like(hidden[:, :1, :])
                shifted = torch.cat((carry, hidden[:, :-1, :]), dim=1)
                prefill(start, end, shifted)
                carry = hidden[:, -1:, :].clone()
                self.draft_end = end
            self.complete = True
            self.chunks.clear()
            return carry
        except BaseException:
            self.abort()
            raise

    def abort(self):
        self.chunks.clear()
        self.failed = True
        self.complete = False
