"""Horizontal scaling: route each card to one scorer partition.

Card-keyed state (velocity windows, decayed card counters) is then local to one
partition and needs no coordination. Cross-entity state is NOT partitionable by
card: the decayed counters, risk tables and peer stats take a shared
``KVBackend`` (e.g. ``RedisKV``), while the entity graph and sequence store are
in-process here and would move to a shared graph/feature service in production.
"""

from __future__ import annotations

from collections.abc import Callable

from .kv import partition_for
from .schema import AccountEvent, LabelEvent, Transaction


class PartitionedScorer:
    def __init__(self, n_partitions: int, factory: Callable[[int], object]) -> None:
        self.n = n_partitions
        self.partitions = [factory(i) for i in range(n_partitions)]

    def partition_of(self, card_id: str) -> int:
        return partition_for(card_id, self.n)

    def score(self, t: Transaction):
        return self.partitions[self.partition_of(t.card_id)].score(t)

    def broadcast_label(self, ev: LabelEvent) -> None:
        for p in self.partitions:
            p.on_label(ev)

    def broadcast_account_event(self, ev: AccountEvent) -> None:
        for p in self.partitions:
            p.on_account_event(ev)
