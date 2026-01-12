import logging
import time

import pytz
from apscheduler.schedulers.background import BackgroundScheduler


def start_process():
    project_name = 'CILL'
    logging.info("Starting the git tracking for => " + project_name)


scheduler = BackgroundScheduler(timezone=pytz.utc)
# scheduler.add_job(cron_job_function, "interval", hours=5)
scheduler.add_job(start_process, "interval", seconds=15)
scheduler.start()

while True:
    time.sleep(1)