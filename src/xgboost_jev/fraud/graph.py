"""Incremental entity graph for fraud-ring and money-mule features.

Nodes are namespaced entities (``user:``, ``device:``, ``ip:``, ``payee:``).
Edges come from transactions and account events. A union-find keeps connected
components with their size and the number of entities confirmed as fraud, so
"how many confirmed-fraud entities share infrastructure with this payment" is
an O(alpha) lookup.

Hub protection: an entity whose degree exceeds ``hub_degree`` (a corporate
NAT IP, a shared kiosk) stops merging components; otherwise one public IP
would glue the whole customer base into a single component.

Fraud flags arrive through :meth:`mark_fraud` when labels become known, so the
graph is point-in-time correct during replay and updates reputation online
without retraining.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque


class EntityGraph:
    def __init__(
        self, hub_degree: int = 25, max_neighbors: int = 200, fan_in_window_s: float = 7 * 86400
    ) -> None:
        self.hub_degree = hub_degree
        self.max_neighbors = max_neighbors
        self.fan_in_window_s = fan_in_window_s
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}
        self._fraud_count: dict[str, int] = {}
        self._fraud: set[str] = set()
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._payee_in: dict[str, deque] = defaultdict(deque)
        self._payee_first: dict[str, float] = {}
        self._user_payees: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    # -- union-find ------------------------------------------------------------
    def _find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1
            self._fraud_count[x] = 1 if x in self._fraud else 0
            return x
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        self._fraud_count[ra] += self._fraud_count[rb]

    def link(self, a: str, b: str) -> None:
        with self._lock:
            new_edge = b not in self._adj[a]
            if new_edge:
                if len(self._adj[a]) < self.max_neighbors:
                    self._adj[a].add(b)
                if len(self._adj[b]) < self.max_neighbors:
                    self._adj[b].add(a)
            if len(self._adj[a]) <= self.hub_degree and len(self._adj[b]) <= self.hub_degree:
                self._union(a, b)
            else:
                self._find(a)
                self._find(b)

    # -- updates ---------------------------------------------------------------
    def observe_txn(
        self, ts: float, user: str, card: str, device: str | None, ip: str | None, payee: str | None
    ) -> None:
        with self._lock:
            u = f"user:{user}"
            self.link(u, f"card:{card}")
            if device:
                self.link(u, f"device:{device}")
            if ip:
                self.link(u, f"ip:{ip}")
            if payee:
                p = f"payee:{payee}"
                # A payee account is the same account as user:<payee> when it is ours.
                self.link(p, f"user:{payee}")
                q = self._payee_in[p]
                q.append((ts, user))
                while q and ts - q[0][0] > self.fan_in_window_s:
                    q.popleft()
                self._payee_first.setdefault(p, ts)
                self._user_payees[u].add(p)

    def observe_login(self, user: str, device: str | None, ip: str | None) -> None:
        if device:
            self.link(f"user:{user}", f"device:{device}")
        if ip:
            self.link(f"user:{user}", f"ip:{ip}")

    def mark_fraud(self, entity: str) -> None:
        with self._lock:
            if entity in self._fraud:
                return
            self._fraud.add(entity)
            if entity in self._parent:
                self._fraud_count[self._find(entity)] += 1

    # -- reads -----------------------------------------------------------------
    def is_fraud(self, entity: str) -> bool:
        return entity in self._fraud

    def component(self, entity: str) -> tuple[int, int]:
        """(size, confirmed-fraud entities) of the entity's component; (0, 0) if unseen."""
        with self._lock:
            if entity not in self._parent:
                return 0, 0
            r = self._find(entity)
            return self._size[r], self._fraud_count[r]

    def components(self, entities: list[str]) -> tuple[int, int]:
        """(largest size, confirmed fraud summed over DISTINCT components) for several entities."""
        with self._lock:
            roots = {self._find(e) for e in entities if e in self._parent}
            if not roots:
                return 0, 0
            return max(self._size[r] for r in roots), sum(self._fraud_count[r] for r in roots)

    def degree(self, entity: str, prefix: str = "") -> int:
        with self._lock:
            return sum(1 for n in self._adj.get(entity, ()) if n.startswith(prefix))

    def two_hop_fraud(self, entity: str) -> int:
        """Confirmed-fraud entities exactly two hops away (e.g. device -> user -> device)."""
        with self._lock:
            seen, hits = {entity}, 0
            for n1 in self._adj.get(entity, ()):
                for n2 in self._adj.get(n1, ()):
                    if n2 not in seen:
                        seen.add(n2)
                        hits += n2 in self._fraud
            return hits

    def payee_stats(self, payee: str, user: str, now: float) -> dict[str, float]:
        with self._lock:
            p = f"payee:{payee}"
            q = self._payee_in.get(p, ())
            senders = {u for ts, u in q if now - ts <= self.fan_in_window_s}
            first = self._payee_first.get(p)
            return {
                "payee_fan_in_7d": float(len(senders)),
                "payee_age_days": (now - first) / 86400 if first is not None else 0.0,
                "payee_new_for_user": float(p not in self._user_payees.get(f"user:{user}", ())),
            }
