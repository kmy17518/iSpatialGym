from scripts.campaign_20261110.infra_dispatch import backend_capacity


def test_unused_account_reports_full_verified_partition_quota():
    rows = [('minyeong', 'svl', 'RUNNING', 'gpu:titanrtx:40')]
    assert backend_capacity('shiyuc_svl', rows) == 48
    assert backend_capacity('shiyuc_viscam', rows) == 30
    assert backend_capacity('minyeong_svl', rows) == 8


def test_cpu_services_and_dual_partition_pending_retain_real_limits():
    rows = [('minyeong', 'svl', 'RUNNING', 'N/A')] * 54
    rows += [('minyeong', 'viscam,svl', 'PENDING', 'gpu:titanrtx:1')] * 3
    assert backend_capacity('minyeong_svl', rows) == 2
    assert backend_capacity('minyeong_viscam', rows) == 27
    rows += [('minyeong', 'svl', 'COMPLETING', 'N/A')] * 2
    assert backend_capacity('minyeong_svl', rows) == 0


def test_submitted_job_quota_and_gpu_quota_still_bound_admission():
    rows = [('shiyuc', 'svl', 'PENDING', 'N/A')] * 99
    assert backend_capacity('shiyuc_svl', rows) == 1
    rows.append(('shiyuc', 'svl', 'RUNNING', 'gpu:titanrtx:48'))
    assert backend_capacity('shiyuc_svl', rows) == 0
