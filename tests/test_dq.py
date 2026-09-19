import pandas as pd
import pytest

from mrisk.data.dq import RULES, DQContext, blocked_dates, run_rules
from mrisk.data.dq import rules as r

D = pd.to_datetime
DAYS = D(["2025-04-28", "2025-04-29", "2025-04-30", "2025-05-06", "2025-05-07"])
PARAMS = dict(
    arrival_deadline="17:00", mv_change_max=0.30, row_count_change_max=0.05,
    price_limit_tolerance_pct=0.001, price_limit_tolerance_abs=0.01, new_listing_exempt_days=5,
)
IDS = ["SH600000", "SZ300059", "BJ920682"]


def _instruments():
    return pd.DataFrame(
        {
            "instrument_id": IDS,
            "exchange": ["SH", "SZ", "BJ"],
            "symbol": ["600000", "300059", "920682"],
            "name": ["浦发银行", "东方财富", "球冠电缆"],
            "industry_l1": ["银行", "非银金融", "电力设备"],
            "board": ["main", "chinext", "bj"],
            "list_date": D(["1999-11-10", "2010-03-19", "2020-07-27"]),
            "status": ["active"] * 3,
        }
    )


def _prices():
    closes = {"SH600000": [10.0, 10.2, 10.1, 10.3, 10.4], "SZ300059": [20.0, 21.0, 20.5, 20.8, 21.0], "BJ920682": [11.95, 11.95, 11.95, 12.63, 12.44]}
    rows = []
    for s, cs in closes.items():
        for d, c in zip(DAYS, cs):
            rows.append({"trade_date": d, "instrument_id": s, "close": c, "adj_close": c * 2, "volume": 100, "is_suspended": False,
                         "ingested_at": d + pd.Timedelta("16:35:00"), "load_type": "daily"})
    return pd.DataFrame(rows)


def _positions():
    return pd.DataFrame(
        [{"trade_date": d, "portfolio_id": "P1", "instrument_id": s, "quantity": 1000.0} for d in DAYS for s in IDS]
    )


def _code_map():
    return pd.DataFrame({"instrument_id": ["BJ920682"], "old_symbol": ["834682"], "new_symbol": ["920682"], "effective_date": D(["2025-05-06"])})


def ctx(**over):
    base = dict(
        check_dates=pd.DatetimeIndex(DAYS), calendar=pd.Series(DAYS), instruments=_instruments(), code_map=_code_map(),
        prices=_prices(), positions=_positions(), params=PARAMS,
    )
    base.update(over)
    return DQContext(**base)


def _status(res, rule_id, day=None):
    sub = res[res["rule_id"] == rule_id]
    if day is not None:
        sub = sub[sub["check_date"] == D(day)]
    return sub["status"].tolist()


def test_clean_data_passes_all_rules():
    res = run_rules(ctx(), RULES, blocking=set())
    assert set(res["rule_id"]) == {f"DQ{i:02d}" for i in range(1, 13)}
    bad = res[res["status"] == "fail"]
    assert bad.empty, bad
    # first day has no previous day for DQ11/DQ12 -> skip, everything else passes
    first = res[res["check_date"] == DAYS[0]].set_index("rule_id")["status"]
    assert first["DQ11"] == "skip" and first["DQ12"] == "skip"


def test_dq01_missing_quote():
    px = _prices()
    px = px[~((px["instrument_id"] == "SH600000") & (px["trade_date"] == DAYS[2]))]
    out = r.dq01_price_complete(ctx(prices=px))
    assert out.failures[["check_date", "instrument_id"]].values.tolist() == [[DAYS[2], "SH600000"]]
    assert out.checked[DAYS[2]] == 3


def test_dq01_ignores_dates_before_listing():
    inst = _instruments()
    inst.loc[2, "list_date"] = DAYS[3]
    px = _prices()
    px = px[~((px["instrument_id"] == "BJ920682") & (px["trade_date"] < DAYS[3]))]
    out = r.dq01_price_complete(ctx(instruments=inst, prices=px))
    assert out.failures.empty
    assert out.checked[DAYS[0]] == 2


def test_dq02_position_without_price():
    px = _prices()
    px = px[~((px["instrument_id"] == "SZ300059") & (px["trade_date"] == DAYS[4]))]
    out = r.dq02_position_has_price(ctx(prices=px))
    assert out.failures["instrument_id"].tolist() == ["SZ300059"]


def test_dq03_non_positive_close():
    px = _prices()
    px.loc[0, "close"] = 0.0
    assert len(r.dq03_close_positive(ctx(prices=px)).failures) == 1


def test_price_limit_table():
    board = pd.Series(["main", "main", "chinext", "chinext", "star", "bj"])
    st = pd.Series([False, True, False, False, False, False])
    dates = pd.Series(D(["2025-01-02", "2025-01-02", "2020-08-21", "2020-08-24", "2025-01-02", "2025-01-02"]))
    assert r.price_limit(board, st, dates).tolist() == [0.10, 0.05, 0.10, 0.20, 0.20, 0.30]


def _two_day_prices(closes, factors, board="main", name="X"):
    inst = pd.DataFrame({"instrument_id": ["A"], "exchange": ["SH"], "symbol": ["600001"], "name": [name], "industry_l1": ["银行"],
                         "board": [board], "list_date": D(["2000-01-04"]), "status": ["active"]})
    px = pd.DataFrame({"trade_date": DAYS[:2], "instrument_id": "A", "close": closes, "adj_close": [c * f for c, f in zip(closes, factors)],
                       "volume": 1, "is_suspended": False, "ingested_at": DAYS[:2], "load_type": "backfill"})
    return dict(instruments=inst, prices=px, positions=_positions().iloc[:0])


@pytest.mark.parametrize(
    "closes,factors,board,name,expect_fail",
    [
        ([10.00, 11.01], [1, 1], "main", "X", False),   # 10*1.1=11.00, tol 0.011 -> 11.011
        ([10.00, 11.02], [1, 1], "main", "X", True),
        ([1.05, 1.16], [1, 1], "main", "X", False),     # tick rounding: 1.155 -> 1.16, tol 0.01
        ([10.00, 10.60], [1, 1], "main", "*ST某", True),  # ST 5%
        ([10.00, 12.00], [1, 1], "chinext", "X", False),  # 2025 chinext 20%
        ([10.00, 7.10], [1, 1], "bj", "X", False),        # BJ -29%
        ([10.00, 10.00], [1, 1.1], "main", "X", False),   # ex-date: ref 9.0909, upper 10.0
        ([10.00, 10.50], [1, 1.1], "main", "X", True),    # +5% raw but +15.5% vs ex-ref
    ],
)
def test_dq04_hand_cases(closes, factors, board, name, expect_fail):
    out = r.dq04_within_price_limit(ctx(**_two_day_prices(closes, factors, board, name), check_dates=pd.DatetimeIndex(DAYS[:2])))
    assert out.checked[DAYS[1]] == 1
    assert (len(out.failures) == 1) is expect_fail


def test_dq04_exempts_new_listing_and_suspended():
    kw = _two_day_prices([10.0, 14.4], [1, 1])
    kw["instruments"].loc[0, "list_date"] = DAYS[0]
    out = r.dq04_within_price_limit(ctx(**kw, check_dates=pd.DatetimeIndex(DAYS[:2])))
    assert out.checked.sum() == 0


def test_dq05_factor_drop():
    px = _prices()
    px.loc[(px["instrument_id"] == "SH600000") & (px["trade_date"] == DAYS[3]), "adj_close"] = 10.3 * 1.5
    out = r.dq05_adj_factor_monotone(ctx(prices=px))
    assert out.failures["check_date"].tolist() == [DAYS[3]]


def test_dq06_lot_size_bj_allows_odd_lots():
    pos = _positions()
    pos.loc[(pos["instrument_id"] == "BJ920682"), "quantity"] = 1234.0
    pos.loc[(pos["instrument_id"] == "SH600000") & (pos["trade_date"] == DAYS[0]), "quantity"] = 1050.0
    pos.loc[(pos["instrument_id"] == "SZ300059") & (pos["trade_date"] == DAYS[1]), "quantity"] = 99.0
    out = r.dq06_lot_size(ctx(positions=pos))
    assert sorted(out.failures["instrument_id"]) == ["SH600000", "SZ300059"]


def test_dq06_bj_below_minimum():
    pos = _positions()
    pos.loc[(pos["instrument_id"] == "BJ920682") & (pos["trade_date"] == DAYS[0]), "quantity"] = 50.0
    assert r.dq06_lot_size(ctx(positions=pos)).failures["instrument_id"].tolist() == ["BJ920682"]


def test_dq07_delisted_holding():
    inst = _instruments()
    inst.loc[0, "status"] = "delisted"
    out = r.dq07_no_delisted_holdings(ctx(instruments=inst))
    assert len(out.failures) == 5


def test_dq08_duplicate_symbol():
    inst = pd.concat([_instruments(), _instruments().iloc[[0]].assign(instrument_id="SH600000X")], ignore_index=True)
    out = r.dq08_symbol_unique(ctx(instruments=inst))
    assert len(out.failures) == 2 * len(DAYS)
    assert (out.checked == 4).all()


def test_dq09_old_symbol_collides_with_current_symbol():
    cm = pd.concat([_code_map(), pd.DataFrame({"instrument_id": ["SH600000"], "old_symbol": ["600000"], "new_symbol": ["600001"], "effective_date": D(["2025-01-01"])})])
    out = r.dq09_code_map_disjoint(ctx(code_map=cm))
    assert set(out.failures["old_symbol"]) == {"600000"}


def test_dq09_old_equals_another_new():
    cm = pd.DataFrame({"instrument_id": ["BJ920682", "BJ920682"], "old_symbol": ["834682", "920683"], "new_symbol": ["920683", "920682"],
                       "effective_date": D(["2025-05-06", "2025-06-01"])})
    out = r.dq09_code_map_disjoint(ctx(code_map=cm))
    assert set(out.failures["old_symbol"]) == {"920683"}


def test_dq10_late_daily_fails_backfill_skipped():
    px = _prices()
    px.loc[px["trade_date"] == DAYS[4], "ingested_at"] = DAYS[4] + pd.Timedelta("17:00:01")
    px.loc[px["trade_date"] == DAYS[0], "load_type"] = "backfill"
    px.loc[px["trade_date"] == DAYS[0], "ingested_at"] = D("2026-09-19 10:00")
    res = run_rules(ctx(prices=px), [x for x in RULES if x.rule_id == "DQ10"], blocking=set())
    assert res["status"].tolist() == ["skip", "pass", "pass", "pass", "fail"]


def test_dq11_mv_jump():
    pos = _positions()
    pos.loc[(pos["instrument_id"] == "SZ300059") & (pos["trade_date"] == DAYS[3]), "quantity"] = 3000.0
    # MV 04-30: 10.1k+20.5k+11.95k=42.55k; 05-06: 10.3k+62.4k+12.63k=85.33k -> +100%
    out = r.dq11_mv_change(ctx(positions=pos))
    assert out.failures["check_date"].tolist() == [DAYS[3], DAYS[4]]
    assert out.failures["change"].iloc[0] == pytest.approx(85330 / 42550 - 1)


def test_dq12_row_count_drop():
    px = _prices()
    px = px[~((px["instrument_id"] == "SH600000") & (px["trade_date"] == DAYS[4]))]
    out = r.dq12_row_count_change(ctx(prices=px))
    assert out.failures["check_date"].tolist() == [DAYS[4]]
    assert out.failures["change"].iloc[0] == pytest.approx(-1 / 3)


def test_blocking_failure_blocks_only_its_date():
    px = _prices()
    px.loc[(px["instrument_id"] == "SH600000") & (px["trade_date"] == DAYS[2]), "close"] = -1.0
    res = run_rules(ctx(prices=px), RULES, blocking={"DQ03"})
    assert blocked_dates(res).tolist() == [DAYS[2]]
    detail = res[(res["rule_id"] == "DQ03") & (res["status"] == "fail")]["detail_json"].iloc[0]
    assert detail["n_failed"] == 1 and detail["sample"][0]["instrument_id"] == "SH600000"


def test_warn_failure_does_not_block():
    px = _prices()
    px = px[~((px["instrument_id"] == "SH600000") & (px["trade_date"] == DAYS[4]))]
    res = run_rules(ctx(prices=px), RULES, blocking={"DQ03"})
    assert "fail" in _status(res, "DQ12", "2025-05-07")
    assert len(blocked_dates(res)) == 0
