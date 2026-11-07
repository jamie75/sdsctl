from sds200.ota_alias import OtaAliasValidator
from sds200.xml_protocol import ScannerInfoParser


def frame(uid='123', name='Alpha', system='A', mute='Unmute'):
    return ScannerInfoParser().parse('PSI', f'<ScannerInfo><System Name="{system}"/>'
        f'<TGID TGID="1"/><UnitID U_Id="UID:{uid}" Name="{name}"/>'
        f'<Property Mute="{mute}"/></ScannerInfo>')


def test_repeat_and_identity_reset():
    v = OtaAliasValidator()
    assert v.observe(frame(), 0) == (None, None)
    assert v.observe(frame(), .4) == (None, None)
    alias, event = v.observe(frame(), .8)
    assert alias == 'Alpha' and event
    assert v.observe(frame(), .9) == ('Alpha', None)
    assert v.observe(frame(uid='456'), 1)[0] is None
    assert v.observe(frame(uid='456'), 1.8)[0] is None


def test_delayed_name_and_numeric_echo():
    v = OtaAliasValidator()
    assert v.observe(frame(name='UID:999'), 0)[0] is None
    assert v.observe(frame(), .3)[0] is None
    assert v.observe(frame(), .7)[0] is None
    assert v.observe(frame(), 1.1)[0] == 'Alpha'
    assert v.observe(frame(system='B'), 1.2)[0] is None
    assert v.observe(frame(mute='Mute'), 1.3)[0] is None


def test_new_name_requires_new_validation_and_emits_once():
    v = OtaAliasValidator()
    for t in (0, .4, .8):
        result = v.observe(frame(), t)
    assert result[0] == 'Alpha' and result[1]
    for t in (1, 1.4):
        assert v.observe(frame(name='Beta'), t) == (None, None)
    assert v.observe(frame(name='Beta'), 1.8)[0] == 'Beta'
    assert v.observe(frame(name='Beta'), 1.9) == ('Beta', None)


def test_gap_and_mute_require_revalidation():
    v = OtaAliasValidator()
    for t in (0, .4, .8):
        v.observe(frame(), t)
    assert v.observe(frame(), 2) == (None, None)
    assert v.observe(frame(mute='Mute'), 2.1) == (None, None)
    assert v.observe(frame(), 2.2) == (None, None)


def test_placeholder_before_uid_change_does_not_forget_previous_name():
    v = OtaAliasValidator()
    for t in (0, .4, .8):
        v.observe(frame(), t)
    v.observe(frame(name='None'), .9)
    for t in (1, 1.4, 1.8):
        assert v.observe(frame(uid='456'), t) == (None, None)


def test_radio_snapshot_carries_validated_alias_and_clears_independently(monkeypatch):
    from dataclasses import asdict

    import sds200.ota_alias as module
    from sds200.state import RadioState
    state = RadioState()
    for t in (0, .4, .8):
        monkeypatch.setattr(module, 'monotonic', lambda t=t: t)
        state.update(frame())
    snapshot = state.snapshot
    assert asdict(snapshot)['ota_alias'] == 'Alpha'
    assert snapshot.ota_event
    monkeypatch.setattr(module, 'monotonic', lambda: .9)
    state.update(frame(uid='456'))
    assert state.snapshot.unit_id == 'UID:456'
    assert state.snapshot.ota_alias is None and state.snapshot.ota_event is None


def test_placeholder_duplicate_and_fallback_uid_rejected():
    for name in ('None', 'UID None', 'UID:None', 'UID: None', 'n/a', '', '123', 'UID:999'):
        v = OtaAliasValidator()
        for t in (0, .4, .8):
            assert v.observe(frame(name=name), t) == (None, None)
    for nodes in (
        '<UnitID Name="Alpha"/><TGID U_Id="UID:123" TGID="1"/>',
        '<UnitID Name="Alpha" U_Id="UID:123"/><UnitID Name="Beta" U_Id="UID:123"/>',
    ):
        info = ScannerInfoParser().parse('PSI', '<ScannerInfo><System Name="A"/>'
            + nodes + '<Property Mute="Unmute"/></ScannerInfo>')
        v = OtaAliasValidator()
        for t in (0, .4, .8):
            assert v.observe(info, t) == (None, None)
