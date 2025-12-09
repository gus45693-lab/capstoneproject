# src/sentiment/score_map.py
from dataclasses import dataclass
from typing import Literal

Label5 = Literal["강한부정", "부정", "중립", "긍정", "강한긍정"]

@dataclass
class Thresholds:
    t1: float = 0.55
    t2: float = 0.75

def score_to_label(s: float, th: Thresholds) -> Label5:
    if s <= -th.t2: return "강한부정"
    if s <= -th.t1: return "부정"
    if -th.t1 < s < th.t1: return "중립"
    if s < th.t2: return "긍정"
    return "강한긍정"
