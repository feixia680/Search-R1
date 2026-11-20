import subprocess,json
from pathlib import Path
result=subprocess.run(['bash', 'run_engineering_benchmark_2gpu.sh'],cwd='/g22538014wej/cityins1122/cityins')
Path('/g22538014wej/cityins1122/cityins/analysis/engineering_profile_20261119_091328/exit_code.json').write_text(json.dumps({"exit_code":result.returncode})+"\n")
raise SystemExit(result.returncode)
