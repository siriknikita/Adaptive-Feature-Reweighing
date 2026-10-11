
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json")
    args = p.parse_args()
    data = json.loads(Path(args.results_json).read_text())
    print(data["summary"])

if __name__ == "__main__":
    main()
