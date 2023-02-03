import time

current_time = int(time.time())
print(current_time)
# sleep for 1 seconds
time.sleep(1)
# integer current_time
current_time2 = int(time.time())
print(current_time2)

print(current_time2 - current_time)
