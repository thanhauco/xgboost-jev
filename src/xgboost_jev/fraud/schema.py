from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    timestamp: float  # unix seconds
    user_id: str
    card_id: str
    amount: float
    merchant_id: str
    mcc: str = "5999"
    channel: str = "card_present"  # card_present | ecommerce | p2p | atm
    device_id: str | None = None
    ip: str | None = None
    lat: float | None = None
    lon: float | None = None
    merchant_descriptor: str = ""
    memo: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Transaction:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)
