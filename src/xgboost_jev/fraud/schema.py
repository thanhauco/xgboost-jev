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
    # P2P / push payments
    payee_id: str | None = None
    payee_name: str = ""  # name the payer typed
    payee_account_name: str = ""  # name on the receiving account (confirmation of payee)
    # In-session free text: support chat, "reason for payment" answers
    session_text: str = ""
    # Device signals from a client SDK (None = not reported)
    is_emulator: bool | None = None
    is_rooted: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Transaction:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


# Account lifecycle events that precede payments (used by sequence features).
EVENT_KINDS = (
    "login",
    "login_new_device",
    "password_reset",
    "mfa_change",
    "new_payee",
    "profile_change",
    "txn",
)


@dataclass(frozen=True)
class AccountEvent:
    timestamp: float
    user_id: str
    kind: str
    device_id: str | None = None
    ip: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class LabelEvent:
    """A label that becomes KNOWN at ``timestamp`` (chargeback, victim report, analyst)."""

    timestamp: float
    txn_id: str
    label: int
    source: str = "chargeback"  # chargeback | report | analyst
    fraud_type: str = ""  # reason code, e.g. "mule_ring" (used by recall monitoring)
