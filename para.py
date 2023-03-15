import parallel
import time
p = parallel.Parallel() # open LPT1 or /dev/parport0
while True:
	p.setData(0xFF)
	time.sleep(3)
	p.setData(0x00)
	time.sleep(3)
