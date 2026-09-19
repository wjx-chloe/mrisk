import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import Date, DateTime, Engine, Table, create_engine, delete, event, select
from sqlalchemy.dialects import postgresql, sqlite

from mrisk.db.schema import metadata


def get_engine(url: str) -> Engine:
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


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
