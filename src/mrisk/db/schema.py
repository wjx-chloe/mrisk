from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

JsonType = JSON().with_variant(JSONB(), "postgresql")

trade_calendar = Table(
    "trade_calendar",
    metadata,
    Column("trade_date", Date, primary_key=True),
)

instrument = Table(
    "instrument",
    metadata,
    Column("instrument_id", Text, primary_key=True),
    Column("exchange", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("industry_l1", Text),
    Column("board", Text, nullable=False),
    Column("list_date", Date),
    Column("status", Text, nullable=False),
    CheckConstraint("exchange IN ('SH','SZ','BJ')", name="ck_instrument_exchange"),
    CheckConstraint("status IN ('active','delisted')", name="ck_instrument_status"),
)

instrument_code_map = Table(
    "instrument_code_map",
    metadata,
    Column("instrument_id", Text, ForeignKey("instrument.instrument_id"), nullable=False),
    Column("old_symbol", Text, nullable=False),
    Column("new_symbol", Text, nullable=False),
    Column("effective_date", Date, nullable=False),
    PrimaryKeyConstraint("instrument_id", "old_symbol"),
)

daily_price = Table(
    "daily_price",
    metadata,
    Column("trade_date", Date, nullable=False),
    Column("instrument_id", Text, ForeignKey("instrument.instrument_id"), nullable=False),
    Column("close", Float, nullable=False),
    # 后复权收盘价（决策：用后复权，历史值不随新的除权事件改变）
    Column("adj_close", Float, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("is_suspended", Boolean, nullable=False),
    Column("ingested_at", DateTime, nullable=False),
    Column("load_type", Text, nullable=False),
    PrimaryKeyConstraint("trade_date", "instrument_id"),
    CheckConstraint("load_type IN ('backfill','daily')", name="ck_price_load_type"),
)

portfolio = Table(
    "portfolio",
    metadata,
    Column("portfolio_id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("base_currency", Text, nullable=False, default="CNY"),
)

position = Table(
    "position",
    metadata,
    Column("trade_date", Date, nullable=False),
    Column("portfolio_id", Text, ForeignKey("portfolio.portfolio_id"), nullable=False),
    Column("instrument_id", Text, ForeignKey("instrument.instrument_id"), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("cost_price", Float),
    PrimaryKeyConstraint("trade_date", "portfolio_id", "instrument_id"),
    # 只做多头（决策 5）
    CheckConstraint("quantity > 0", name="ck_position_long_only"),
)

risk_result = Table(
    "risk_result",
    metadata,
    Column("trade_date", Date, nullable=False),
    Column("portfolio_id", Text, nullable=False),
    Column("level", Text, nullable=False),
    Column("level_key", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("market_value", Float),
    Column("var_hs", Float),
    Column("var_param", Float),
    Column("es_hs", Float),
    Column("run_id", Text, nullable=False),
    PrimaryKeyConstraint("trade_date", "portfolio_id", "level", "level_key", "confidence"),
    CheckConstraint("level IN ('portfolio','industry','instrument')", name="ck_risk_level"),
)

limit_rule = Table(
    "limit_rule",
    metadata,
    Column("rule_id", Text, primary_key=True),
    Column("portfolio_id", Text, nullable=False),
    Column("metric", Text, nullable=False),
    Column("scope", Text, nullable=False),
    Column("operator", Text, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("threshold_type", Text, nullable=False),
    Column("severity", Text, nullable=False),
    Column("is_active", Boolean, nullable=False),
    CheckConstraint("operator IN ('<=','<')", name="ck_limit_operator"),
    CheckConstraint("threshold_type IN ('absolute','pct_of_mv')", name="ck_limit_threshold_type"),
    CheckConstraint("severity IN ('warn','breach')", name="ck_limit_severity"),
)

limit_check = Table(
    "limit_check",
    metadata,
    Column("trade_date", Date, nullable=False),
    Column("rule_id", Text, ForeignKey("limit_rule.rule_id"), nullable=False),
    Column("actual_value", Float),
    Column("threshold", Float),
    Column("status", Text, nullable=False),
    PrimaryKeyConstraint("trade_date", "rule_id"),
    CheckConstraint("status IN ('ok','warn','breach')", name="ck_limit_check_status"),
)

dq_rule = Table(
    "dq_rule",
    metadata,
    Column("rule_id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("category", Text, nullable=False),
    Column("severity", Text, nullable=False),
    Column("description", Text),
    Column("is_active", Boolean, nullable=False),
    CheckConstraint("severity IN ('blocking','warn')", name="ck_dq_rule_severity"),
)

dq_result = Table(
    "dq_result",
    metadata,
    Column("run_id", Text, nullable=False),
    Column("check_date", Date, nullable=False),
    Column("rule_id", Text, ForeignKey("dq_rule.rule_id"), nullable=False),
    Column("status", Text, nullable=False),
    Column("n_checked", Integer, nullable=False),
    Column("n_failed", Integer, nullable=False),
    Column("detail_json", JsonType),
    Column("created_at", DateTime, nullable=False),
    PrimaryKeyConstraint("run_id", "check_date", "rule_id"),
    CheckConstraint("status IN ('pass','fail','skip')", name="ck_dq_result_status"),
)

scenario = Table(
    "scenario",
    metadata,
    Column("scenario_id", Text, primary_key=True),
    Column("type", Text, nullable=False),
    Column("definition", JsonType, nullable=False),
    CheckConstraint("type IN ('historical','hypothetical')", name="ck_scenario_type"),
)


pnl_hypo = Table(
    "pnl_hypo",
    metadata,
    Column("position_date", Date, nullable=False),
    Column("pnl_date", Date, nullable=False),
    Column("portfolio_id", Text, nullable=False),
    Column("level", Text, nullable=False),
    Column("level_key", Text, nullable=False),
    Column("market_value", Float, nullable=False),
    Column("pnl", Float, nullable=False),
    Column("run_id", Text, nullable=False),
    PrimaryKeyConstraint("position_date", "portfolio_id", "level", "level_key"),
    CheckConstraint("level IN ('portfolio','industry','instrument')", name="ck_pnl_level"),
)

backtest_result = Table(
    "backtest_result",
    metadata,
    Column("asof_date", Date, nullable=False),
    Column("portfolio_id", Text, nullable=False),
    Column("method", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("window_days", Integer, nullable=False),
    Column("n_obs", Integer, nullable=False),
    Column("n_exceptions", Integer, nullable=False),
    Column("lr_pof", Float, nullable=False),
    Column("p_value_pof", Float, nullable=False),
    Column("lr_ind", Float, nullable=False),
    Column("p_value_ind", Float, nullable=False),
    Column("zone", Text),
    Column("run_id", Text, nullable=False),
    PrimaryKeyConstraint("asof_date", "portfolio_id", "method", "confidence", "window_days"),
    CheckConstraint("method IN ('hs','param')", name="ck_backtest_method"),
    CheckConstraint("zone IS NULL OR zone IN ('green','yellow','red')", name="ck_backtest_zone"),
)

var_attribution = Table(
    "var_attribution",
    metadata,
    Column("trade_date", Date, nullable=False),
    Column("portfolio_id", Text, nullable=False),
    Column("metric", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("effect", Text, nullable=False),
    Column("level", Text, nullable=False),
    Column("level_key", Text, nullable=False),
    Column("value", Float, nullable=False),
    Column("window_dropped", Date),
    Column("run_id", Text, nullable=False),
    PrimaryKeyConstraint("trade_date", "portfolio_id", "metric", "confidence", "effect", "level", "level_key"),
    CheckConstraint("metric IN ('var_hs','var_param','es_hs')", name="ck_attr_metric"),
    CheckConstraint("effect IN ('quantity','revaluation','window','total')", name="ck_attr_effect"),
    CheckConstraint("level IN ('portfolio','instrument')", name="ck_attr_level"),
)
