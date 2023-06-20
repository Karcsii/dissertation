import subprocess
import re
import pandas as pd
import os
import parallel
import time

block_size = [1, 2, 4, 8, 16, 32, 64]
parallel_tasks = [1, 2, 4, 8, 16, 32, 64]
mode = "write"
name = "probawriteu"
lat_unit = "usec"
thr_unit = "MiB/s"

LatencyBlock = {}
LatencyUnitBlock = {}
IOPSBlock = {}
ThroughputNumberBlock = {}
ThroughputUnitBlock = {}

num_of_tests = len(block_size)*len(parallel_tasks)
p = parallel.Parallel()
outdir = "./{name}".format(name = name)
if not os.path.exists(outdir):
    os.mkdir(outdir)
p.setData(0x00)

for x, num_of_blocks in enumerate(block_size):
	LatencyParallel = {}
	LatencyUnitParallel = {}
	IOPSParallel = {}
	ThroughputNumberParallel = {}
	ThroughputUnitParallel = {}
	for y, num_of_parallel in enumerate(parallel_tasks):
		time.sleep(10)
		command = "sudo fio --name={name} --ioengine=posixaio --rw=write --numjobs={parallel} --size=1g --iodepth={parallel} --runtime=60 --time_based --end_fsync=1 --bs={block}k --group_reporting".format(name = name, parallel = num_of_parallel, block = num_of_blocks)
		print("running test", (x)*len(parallel_tasks)+(y+1), '/', num_of_tests, "block size:", num_of_blocks, "parallel tasks:", num_of_parallel)
		#result = subprocess.run(command.split(), stdout=subprocess.PIPE, cwd="{parent}/{name}/".format(name=name, parent=os.path.dirname(os.path.realpath(__file__))))
		p.setData(0xFF)
		result = subprocess.run(command.split(), stdout=subprocess.PIPE, cwd="./{name}/".format(name=name))
		p.setData(0x00)
		output = result.stdout.decode('utf-8')
		

		lines = output.split('\n')
		i = 0
		while (i<len(lines)):
			item = lines[i]
			i+=1
			if item.startswith("     lat"):
				latency = item.strip()
				break
		while (i<len(lines)):
			item = lines[i]
			i+=1	
			if item.startswith("   iops"):
				iops = item.strip()
				break
		while (i<len(lines)):
			item = lines[i]
			i+=1
			if item.startswith("   READ") or item.startswith("  WRITE"):
				throughput = item.strip()
				break
				
		unit = latency[latency.find('(') + 1 : latency.find(')')]

		start = latency.find('avg=') + 4
		end = latency.find(',', start)
		latency = latency[start : end]

		start = iops.find('avg=') + 4
		end = iops.find(',', start)
		iops = iops[start : end]

		throughputnumber = re.search(r'\d*\.?\d+', throughput).group()

		start = throughput.find(throughputnumber) + len(throughputnumber)
		end = throughput.find(' ', start)
		throughputunit = throughput[start : end]

		
		print("latency:", latency, "unit:", unit, "iops:", iops, "throughputunit:", throughputunit, "throughputnumber:", throughputnumber)
		
		if (unit == lat_unit):
			LatencyParallel[num_of_parallel] = latency
		elif (unit == "msec"):
			LatencyParallel[num_of_parallel] = float(latency) * 1000
		elif (unit == "sec"):
			LatencyParallel[num_of_parallel] = float(latency) * 1e6
		elif (unit == "nsec"):
			LatencyParallel[num_of_parallel] = float(latency) / 1000
		else:
			assert False, "Oh no! This assertion failed!"
		
		
		
		LatencyUnitParallel[num_of_parallel] = unit
		IOPSParallel[num_of_parallel] = iops
		
		if (throughputunit == thr_unit):
			ThroughputNumberParallel[num_of_parallel] = throughputnumber
		elif (throughputunit == "KiB/s"):
			ThroughputNumberParallel[num_of_parallel] = float(throughputnumber) / 1000
		elif (throughputunit == "GiB/s"):
			ThroughputNumberParallel[num_of_parallel] = float(throughputnumber) * 1000
		elif (throughputunit == "B/s"):
			ThroughputNumberParallel[num_of_parallel] = float(throughputnumber) / 1e-6
		else:
			assert False, "Oh no! This assertion failed!"
		
		
		ThroughputUnitParallel[num_of_parallel] = throughputunit
		
	LatencyBlock[num_of_blocks] = LatencyParallel
	LatencyUnitBlock[num_of_blocks] = LatencyUnitParallel
	IOPSBlock[num_of_blocks] = IOPSParallel
	ThroughputNumberBlock[num_of_blocks] = ThroughputNumberParallel
	ThroughputUnitBlock[num_of_blocks] = ThroughputUnitParallel
	


   


latency = pd.DataFrame(LatencyBlock, columns = block_size, index = parallel_tasks)
latency.to_csv("./{name}/latency.csv".format(name = name), index=True, header=True, decimal=',', sep='\t')
unit = pd.DataFrame(LatencyUnitBlock, columns = block_size, index = parallel_tasks)
unit.to_csv("./{name}/unit.csv".format(name = name), sep='\t')
iops = pd.DataFrame(IOPSBlock, columns = block_size, index = parallel_tasks)
iops.to_csv("./{name}/iops.csv".format(name = name), sep='\t')
throughputnumber = pd.DataFrame(ThroughputNumberBlock, columns = block_size, index = parallel_tasks)
throughputnumber.to_csv("./{name}/throughputnumber.csv".format(name = name), sep='\t')
throughputunit = pd.DataFrame(ThroughputUnitBlock, columns = block_size, index = parallel_tasks)
throughputunit.to_csv("./{name}/throughputunit.csv".format(name = name), sep='\t')

print("latency:", latency)
print("unit:", unit)
print("iops:", iops)
print("throughputnumber:", throughputnumber)
print("throughputunit:", throughputunit)


