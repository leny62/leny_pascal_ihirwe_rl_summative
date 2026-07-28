from environment.traceability import GENESIS, BlockLedger, FieldEvent


def make_ledger() -> BlockLedger:
    ledger = BlockLedger(block_id="KG-NYA-004", season="2026A")
    ledger.append(FieldEvent(day=12, action="apply_n_split", zone=3, quantity=30.0, unit="kg_ha"))
    ledger.append(FieldEvent(day=19, action="irrigate_medium", zone=3, quantity=14.0, unit="mm"))
    ledger.append(FieldEvent(day=41, action="spray_biopesticide", zone=7, note="neem"))
    return ledger


def test_empty_ledger_head_is_genesis():
    assert BlockLedger(block_id="X", season="2026A").head == GENESIS


def test_chain_verifies():
    assert make_ledger().verify()


def test_each_event_links_to_the_previous_hash():
    ledger = make_ledger()
    assert ledger.events[0]["prev_hash"] == GENESIS
    for earlier, later in zip(ledger.events, ledger.events[1:], strict=False):
        assert later["prev_hash"] == earlier["hash"]


def test_tampering_with_a_quantity_breaks_verification():
    ledger = make_ledger()
    ledger.events[0]["quantity"] = 5.0
    assert not ledger.verify()


def test_tampering_with_a_hash_breaks_verification():
    ledger = make_ledger()
    ledger.events[1]["hash"] = "0" * 64
    assert not ledger.verify()


def test_identical_events_hash_identically():
    a, b = BlockLedger(block_id="X", season="2026A"), BlockLedger(block_id="X", season="2026A")
    event = FieldEvent(day=1, action="scout")
    assert a.append(event) == b.append(event)
