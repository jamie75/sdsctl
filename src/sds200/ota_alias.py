"""Ephemeral best-effort scanner alias validation, never a UID directory."""
from __future__ import annotations

import re
from time import monotonic

from .models import ScannerInfo


class OtaAliasValidator:
    def __init__(self) -> None:
        self.key: tuple[str | None, str | None, str | None] | None = None
        self.previous: tuple[str | None, str | None, str] | None = None
        self.name: str | None = None
        self.count = 0
        self.since = 0.0
        self.last = 0.0
        self.accepted = False

    def observe(self, info: ScannerInfo, now: float | None = None) -> tuple[str | None, str | None]:
        now = monotonic() if now is None else now
        nodes = info.records_by_tag('UnitID')
        key = (info.system, info.talkgroup_id, info.unit_id)
        valid = (len(nodes) == 1 and nodes[0].get('U_Id') == info.unit_id
                 and info.mute == 'Unmute' and all(key)
                 and re.fullmatch(r'UID:\d+', info.unit_id or '') is not None)
        if key != self.key or not valid or now - self.last >= 1:
            if self.name and self.key:
                self.previous = (self.key[0], self.key[2], self.name)
            self.key, self.name, self.count, self.accepted = key, None, 0, False
        self.last = now
        value = nodes[0].get('Name') if len(nodes) == 1 else None
        value = value.strip() if value and len(value) <= 256 else None
        if (not valid or not value or value.casefold() in {
                'none', 'unknown', 'uid none', 'uid:none', 'uid: none', 'n/a'}
                or re.fullmatch(r'(?:uid\s*:?\s*)?\d+', value, re.I)
                or any(ord(c) < 32 for c in value)
                or (self.previous and self.previous[:2] != (key[0], key[2])
                    and self.previous[2] == value)):
            if self.name and self.key:
                self.previous = (self.key[0], self.key[2], self.name)
            self.name, self.count, self.accepted = None, 0, False
            return None, None
        if value != self.name:
            self.name, self.count, self.since, self.accepted = value, 1, now, False
        else:
            self.count += 1
        if self.count < 3 or now - self.since < .75:
            return None, None
        event = None if self.accepted else info.received_at.isoformat()
        self.accepted = True
        return value, event
