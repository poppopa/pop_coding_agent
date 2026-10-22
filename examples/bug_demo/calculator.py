"""演示用业务模块：空列表应返回 0，当前实现故意保留缺陷。"""
def average(values: list[float]) -> float:
    return sum(values) / len(values)
