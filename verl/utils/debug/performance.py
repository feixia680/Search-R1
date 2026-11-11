# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
import logging
import os


def _memory_probe(head):
    if os.environ.get('SEARCH_R1_MEMORY_DEBUG') != '1':
        return
    import psutil
    import threading
    import time
    import faulthandler
    process = psutil.Process()
    def emit(stage):
        rss = process.memory_info().rss / 1024**3
        gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_initialized() else 0
        rank = dist.get_rank() if dist.is_initialized() else -1
        line = f'MEMPROBE t={time.time():.3f} pid={os.getpid()} rank={rank} rss_GiB={rss:.3f} gpu_GiB={gpu:.3f} threads={process.num_threads()} torch_threads={torch.get_num_threads()} stage={stage}'
        print(line, flush=True)
        path = os.environ.get('SEARCH_R1_MEMORY_DIR')
        if path:
            with open(os.path.join(path, f'memory_{os.getpid()}.log'), 'a') as stream:
                stream.write(line + '\n')
        return rss
    emit(head)
    if not getattr(_memory_probe, 'started', False):
        _memory_probe.started = True
        faulthandler.enable()
        if os.environ.get('SEARCH_R1_DUMP_STACKS') == '1':
            faulthandler.dump_traceback_later(120, repeat=True)
        def watch():
            while True:
                time.sleep(10)
                rss = emit('periodic')
                if rss > float(os.environ.get('SEARCH_R1_MEMORY_LIMIT_GB', '96')):
                    print('MEMPROBE memory safety limit exceeded; terminating this worker', flush=True)
                    faulthandler.dump_traceback(all_threads=True)
                    os._exit(86)
        threading.Thread(target=watch, daemon=True, name='memory-probe').start()


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    _memory_probe(head)
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)
