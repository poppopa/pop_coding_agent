"""单进程测试启动器：4 GiB 地址空间上限随每条测试命令生效。"""
import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
os.execv(sys.executable, [sys.executable, "-m", "pytest", *sys.argv[1:]])
