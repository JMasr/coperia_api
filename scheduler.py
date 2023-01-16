from main import update_data
from apscheduler.schedulers.blocking import BlockingScheduler


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(update_data('dataset'), 'interval', hours=24)
    scheduler.start()
