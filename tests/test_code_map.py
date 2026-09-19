import pandas as pd
import pytest

from mrisk.data.code_map import UnresolvedSymbolError, resolve_instrument_id

D = pd.to_datetime

INSTRUMENTS = pd.DataFrame(
    {"instrument_id": ["BJ920682", "SH600000"], "exchange": ["BJ", "SH"], "symbol": ["920682", "600000"]}
)
CODE_MAP = pd.DataFrame(
    {"instrument_id": ["BJ920682"], "old_symbol": ["834682"], "new_symbol": ["920682"], "effective_date": D(["2025-05-06"])}
)


def test_old_code_before_switch_and_new_code_after_map_to_same_id():
    pos = pd.DataFrame(
        {
            "exchange": ["BJ", "BJ", "SH"],
            "symbol": ["834682", "920682", "600000"],
            "trade_date": D(["2025-04-30", "2025-05-06", "2025-04-30"]),
            "quantity": [1000, 1000, 500],
        }
    )
    out = resolve_instrument_id(pos, INSTRUMENTS, CODE_MAP)
    assert out["instrument_id"].tolist() == ["BJ920682", "BJ920682", "SH600000"]
    assert out["quantity"].tolist() == [1000, 1000, 500]
    assert out.groupby("instrument_id").size()["BJ920682"] == 2


def test_new_code_resolves_before_switch_too():
    df = pd.DataFrame({"exchange": ["BJ"], "symbol": ["920682"], "trade_date": D(["2021-01-04"])})
    assert resolve_instrument_id(df, INSTRUMENTS, CODE_MAP)["instrument_id"].tolist() == ["BJ920682"]


def test_old_code_on_or_after_switch_is_rejected():
    df = pd.DataFrame({"exchange": ["BJ"], "symbol": ["834682"], "trade_date": D(["2025-05-06"])})
    with pytest.raises(UnresolvedSymbolError):
        resolve_instrument_id(df, INSTRUMENTS, CODE_MAP)


def test_old_code_on_wrong_exchange_is_rejected():
    df = pd.DataFrame({"exchange": ["SZ"], "symbol": ["834682"], "trade_date": D(["2025-04-30"])})
    with pytest.raises(UnresolvedSymbolError):
        resolve_instrument_id(df, INSTRUMENTS, CODE_MAP)


def test_unknown_symbol_rejected():
    df = pd.DataFrame({"exchange": ["SH"], "symbol": ["600001"], "trade_date": D(["2025-04-30"])})
    with pytest.raises(UnresolvedSymbolError) as e:
        resolve_instrument_id(df, INSTRUMENTS, CODE_MAP)
    assert e.value.rows["symbol"].tolist() == ["600001"]
