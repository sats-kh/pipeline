import csv
import time
import psutil
from pynvml import *

# GPU 사용량 측정 함수
def get_gpu_usage():
    nvmlInit()
    gpu_usage = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu_usage.append({
            'gpu': i,
            'memory_used_MB': mem_info.used / 1024 / 1024,
            'memory_total_MB': mem_info.total / 1024 / 1024,
            'utilization_percent': util.gpu
        })
    nvmlShutdown()
    return gpu_usage

# CPU 및 메모리 사용량 측정 함수
def get_system_usage():
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_MB': psutil.virtual_memory().used / 1024 / 1024,
        'memory_total_MB': psutil.virtual_memory().total / 1024 / 1024
    }

# 메인 함수
def monitor_resource_usage(output_file="resource_usage.csv"):
    # CSV 파일 초기화
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'timestamp', 'cpu_percent', 'memory_percent',
            'memory_used_MB', 'memory_total_MB',
            'gpu_id', 'gpu_utilization_percent', 'gpu_memory_used_MB', 'gpu_memory_total_MB'
        ])
    
    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            system_usage = get_system_usage()
            gpu_usage = get_gpu_usage()
            
            # GPU별로 데이터를 기록
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for gpu in gpu_usage:
                    writer.writerow([
                        timestamp, system_usage['cpu_percent'], system_usage['memory_percent'],
                        system_usage['memory_used_MB'], system_usage['memory_total_MB'],
                        gpu['gpu'], gpu['utilization_percent'], gpu['memory_used_MB'], gpu['memory_total_MB']
                    ])
            
            time.sleep(1)  # 1초마다 측정
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == '__main__':
    monitor_resource_usage()
