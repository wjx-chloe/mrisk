import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from mrisk.db import schema as s
from mrisk.db.repo import get_engine, init_db, read_frame, replace_partition, upsert


@pytest.fixture
def engine(tmp_path):
    eng = get_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    init_db(eng)
    return eng


def _instrument():
    return pd.DataFrame(
        {
            "instrument_id": ["SH600000", "BJ920682"],
            "exchange": ["SH", "BJ"],
            "symbol": ["600000", "920682"],
            "name": ["浦发银行", "球冠电缆"],
            "industry_l1": ["银行", "电力设备"],
            "board": ["main", "bj"],
            "list_date": pd.to_datetime(["1999-11-10", "2020-07-27"]),
            "status": ["active", "active"],
        }
    )


def _prices(close):
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "instrument_id": ["SH600000", "SH600000"],
            "close": close,
            "adj_close": [c * 10 for c in close],
            "volume": [100, 200],
            "is_suspended": [False, True],
            "ingested_at": pd.to_datetime(["2025-01-02 16:31", "2025-01-03 16:31"]),
            "load_type": ["daily", "daily"],
        }
    )


def test_all_tables_created(engine):
    from sqlalchemy import inspect

    names = set(inspect(engine).get_table_names())
    assert names == set(s.metadata.tables)


def test_upsert_roundtrip_and_idempotent(engine):
    upsert(engine, s.instrument, _instrument())
    upsert(engine, s.daily_price, _prices([10.0, 10.5]))
    upsert(engine, s.daily_price, _prices([10.0, 10.5]))
    df = read_frame(engine, s.daily_price)
    assert len(df) == 2
    assert df["close"].tolist() == [10.0, 10.5]
    assert df["is_suspended"].tolist() == [False, True]
    assert df["trade_date"].dt.year.tolist() == [2025, 2025]


def test_upsert_updates_on_conflict(engine):
    upsert(engine, s.instrument, _instrument())
    upsert(engine, s.daily_price, _prices([10.0, 10.5]))
    upsert(engine, s.daily_price, _prices([11.0, 12.0]))
    df = read_frame(engine, s.daily_price, start="2025-01-03")
    assert df["close"].tolist() == [12.0]


def test_foreign_key_enforced(engine):
    with pytest.raises(IntegrityError):
        upsert(engine, s.daily_price, _prices([10.0, 10.5]))


def test_position_rejects_non_positive_quantity(engine):
    upsert(engine, s.instrument, _instrument())
    upsert(engine, s.portfolio, pd.DataFrame({"portfolio_id": ["P1"], "name": ["x"], "base_currency": ["CNY"]}))
    bad = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2025-01-02"]), "portfolio_id": ["P1"], "instrument_id": ["SH600000"], "quantity": [-100.0]}
    )
    with pytest.raises(IntegrityError):
        upsert(engine, s.position, bad)


def test_replace_partition_drops_stale_rows(engine):
    upsert(engine, s.instrument, _instrument())
    upsert(engine, s.portfolio, pd.DataFrame({"portfolio_id": ["P1"], "name": ["x"], "base_currency": ["CNY"]}))
    pos = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "portfolio_id": ["P1", "P1"],
            "instrument_id": ["SH600000", "BJ920682"],
            "quantity": [100.0, 200.0],
        }
    )
    upsert(engine, s.position, pos)
    replace_partition(engine, s.position, pos.iloc[:1], portfolio_id="P1")
    assert read_frame(engine, s.position)["instrument_id"].tolist() == ["SH600000"]


def test_json_column_roundtrip(engine):
    upsert(
        engine,
        s.scenario,
        pd.DataFrame({"scenario_id": ["mkt-5"], "type": ["hypothetical"], "definition": [{"market": -0.05}]}),
    )
    assert read_frame(engine, s.scenario)["definition"].iloc[0] == {"market": -0.05}


def test_read_frame_filters(engine):
    upsert(engine, s.instrument, _instrument())
    df = read_frame(engine, s.instrument, date_col="list_date", exchange=["BJ"])
    assert df["instrument_id"].tolist() == ["BJ920682"]
