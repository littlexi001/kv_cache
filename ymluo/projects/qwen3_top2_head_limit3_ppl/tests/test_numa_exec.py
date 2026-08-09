from numa_exec_20260716 import parse_cpu_list


def test_parse_cpu_list() -> None:
    assert parse_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
