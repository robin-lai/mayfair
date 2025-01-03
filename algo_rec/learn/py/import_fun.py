import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from algo_rec.utils.util import add_job_monitor
if __name__ == '__main__':
    add_job_monitor(1,1,1)
    pass