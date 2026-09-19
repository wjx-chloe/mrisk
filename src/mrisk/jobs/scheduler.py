import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import Engine

from mrisk.jobs.daily import daily

log = logging.getLogger(__name__)


def build_trigger(daily_time: str) -> CronTrigger:
    hour, minute = daily_time.split(":")
    return CronTrigger(day_of_week="mon-fri", hour=int(hour), minute=int(minute), timezone="Asia/Shanghai")


def run_forever(engine: Engine, cfg) -> None:
    sched = BlockingScheduler()

    def job():
        try:
            daily(engine, cfg)
        except Exception:
            log.exception("daily job failed")

    sched.add_job(job, build_trigger(cfg["scheduler"]["daily_time"]), id="daily", misfire_grace_time=3600, coalesce=True)
    log.info("scheduler started: weekdays at %s Asia/Shanghai", cfg["scheduler"]["daily_time"])
    sched.start()
