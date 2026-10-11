
#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("results_json")
    a=p.parse_args()
    d=json.loads(Path(a.results_json).read_text())
    print(d["summary"])

if __name__=="__main__":
    main()
