from fastlite import Database
from pathlib import Path
import time
import threading, time
from concurrent.futures import ThreadPoolExecutor
import asyncio, time

#-------------- 1. Using SQL as a queue --------------
Path("queue.db").unlink(missing_ok=True)
db = Database("queue.db")

class QueueItem:
    id: int
    data: str
    expire: int 

queue = db.create(QueueItem, pk="id")

def enqueue(data): 
    return queue.insert(data=data, expire=0)

def dequeue():
    items = queue(where="expire=0", limit=1)  # look for unlocked jobs
    if not items: return None
    item = items[0]
    lock_time = int(time.time()) + 60  # lock job for 1 min
    updated = queue.update(id=item.id, expire=lock_time)
    if updated.expire == lock_time:  # check we got the lock
        return updated
    return dequeue()  # retry

# example
enqueue("task1"); enqueue("task2")
while (item := dequeue()):
    print("processing", item.data)

#--------- 2. threads for background tasks----------------

def work(item):
    print(f"start {item}")
    time.sleep(2)  # simulate network call
    print(f"done {item}")

def run_background():
    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.map(work, range(5))

t = threading.Thread(target=run_background)
t.start()

print("main keeps running while work is in background")

# ---------------------3. async processsing------------------


async def task(i):
    print(f"task {i} started")
    await asyncio.sleep(2)  # non-blocking sleep
    print(f"task {i} finished")

async def main():
    start = time.time()
    # run tasks concurrently
    await asyncio.gather(*(task(i) for i in range(5)))
    print("elapsed:", time.time() - start)

# run
asyncio.run(main())

# -----------------------------------------------------------