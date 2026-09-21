"""Conservative per-GPU admission for measured multi-process training settings."""
def reservation_gb(algorithm, peaks):
    # CUDA context, allocator fragmentation and headroom above smoke peak.
    return peaks.get(algorithm, 24.0) * 1.25 + 1.5


def can_admit(algorithm, active, config, peaks):
    caps=config.get('jobs_per_gpu_by_algorithm',{})
    limit=config.get('jobs_per_gpu',1)
    limit=min([limit,caps.get(algorithm,limit)]+[caps.get(a,limit) for a in active])
    return (len(active)<limit and
        sum(reservation_gb(a,peaks) for a in active)+reservation_gb(algorithm,peaks)
        <=config.get('gpu_memory_budget_gb',112.0))
