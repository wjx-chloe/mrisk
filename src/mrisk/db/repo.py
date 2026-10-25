import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import Date, DateTime, Engine, Table, create_engine, delete, event, inspect, select, text
from sqlalchemy.dialects import postgresql, sqlite

from mrisk.db.schema import metadata


def get_engine(url: str) -> Engine:
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def readonly_engine(url: str) -> Engine:
    """SQLite is opened read-only (mode=ro), so the reader cannot write; other databases as given."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and url != "sqlite:///:memory:":
        return create_engine(f"{prefix}file:{url[len(prefix):]}?mode=ro&uri=true")
    return get_engine(url)


def init_db(engine: Engine) -> list[tuple[str, str]]:
    """Creates missing tables and adds the columns a database made by an older version lacks (upgrade_db).
    -> the columns added."""
    metadata.create_all(engine)
    return upgrade_db(engine)


def missing_columns(engine: Engine) -> list[tuple[str, str]]:
    """(table, column) of the schema that an existing table of the database lacks: what upgrade_db would add."""
    insp = inspect(engine)
    have = set(insp.get_table_names())
    out = []
    for name, table in metadata.tables.items():
        if name in have:
            cols = {c["name"] for c in insp.get_columns(name)}
            out += [(name, c.name) for c in table.columns if c.name not in cols]
    return out


def upgrade_db(engine: Engine) -> list[tuple[str, str]]:
    """Adds the missing columns in place (ALTER TABLE ... ADD COLUMN), so a database made by an older version is not
    rebuilt. Only nullable, non-key columns can be added this way; the rows get NULL, which the job that owns the
    column fills (jobs.risk.fill_es_contrib for risk_result.es_hs_contrib). -> the columns added."""
    gaps = missing_columns(engine)
    q = engine.dialect.identifier_preparer.quote
    with engine.begin() as conn:
        for name, col in gaps:
            c = metadata.tables[name].c[col]
            if not c.nullable or c.primary_key:
                raise ValueError(f"{name}.{col} cannot be added to an existing table: not nullable or part of the key")
            conn.execute(text(f"ALTER TABLE {q(name)} ADD COLUMN {q(col)} {c.type.compile(dialect=engine.dialect)}"))
    return gaps


def _to_records(table: Table, df: pd.DataFrame) -> list[dict]:
    unknown = set(df.columns) - set(table.columns.keys())
    if unknown:
        raise ValueError(f"{table.name}: unknown columns {sorted(unknown)}")
    out = df.copy()
    for col in out.columns:
        ctype = table.columns[col].type
        if isinstance(ctype, DateTime):
            out[col] = [d.to_pydatetime() if pd.notna(d) else None for d in pd.to_datetime(out[col])]
        elif isinstance(ctype, Date):
            out[col] = [d.date() if pd.notna(d) else None for d in pd.to_datetime(out[col])]
    out = out.astype(object).where(pd.notna(out), None)
    records = out.to_dict("records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, np.generic):
                r[k] = v.item()
    return records


def _insert_stmt(engine: Engine, table: Table):
    if engine.dialect.name == "sqlite":
        return sqlite.insert(table)
    if engine.dialect.name == "postgresql":
        return postgresql.insert(table)
    raise NotImplementedError(engine.dialect.name)


def upsert(engine: Engine, table: Table, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = _to_records(table, df)
    pk = [c.name for c in table.primary_key.columns]
    stmt = _insert_stmt(engine, table)
    non_pk = [c for c in df.columns if c not in pk]
    if non_pk:
        stmt = stmt.on_conflict_do_update(
            index_elements=pk, set_={c: stmt.excluded[c] for c in non_pk}
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=pk)
    with engine.begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def replace_partition(engine: Engine, table: Table, df: pd.DataFrame, **eq) -> int:
    """Delete every row matching the equality filters, then insert df, in one transaction."""
    records = _to_records(table, df)
    with engine.begin() as conn:
        conn.execute(delete(table).where(*[table.c[k] == v for k, v in eq.items()]))
        if records:
            conn.execute(table.insert(), records)
    return len(records)


def replace_dates(engine: Engine, table: Table, df: pd.DataFrame, date_col: str, dates: Iterable, **eq) -> int:
    """Delete every row on `dates` matching the equality filters, then insert df, in one transaction."""
    days = [pd.Timestamp(d).date() for d in dates]
    records = _to_records(table, df)
    with engine.begin() as conn:
        if days:
            conn.execute(delete(table).where(table.c[date_col].in_(days), *[table.c[k] == v for k, v in eq.items()]))
        if records:
            conn.execute(table.insert(), records)
    return len(records)


def read_frame(
    engine: Engine,
    table: Table,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    date_col: str = "trade_date",
    columns: Iterable[str] | None = None,
    **eq,
) -> pd.DataFrame:
    cols = [table.c[c] for c in columns] if columns else [table]
    stmt = select(*cols)
    if start is not None:
        stmt = stmt.where(table.c[date_col] >= pd.Timestamp(start).date())
    if end is not None:
        stmt = stmt.where(table.c[date_col] <= pd.Timestamp(end).date())
    for k, v in eq.items():
        stmt = stmt.where(table.c[k].in_(v) if isinstance(v, (list, tuple, set)) else table.c[k] == v)
    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)
    for col in df.columns:
        if isinstance(table.columns[col].type, (Date, DateTime)):
            df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df
