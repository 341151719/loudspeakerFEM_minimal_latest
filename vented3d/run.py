"""Run the isolated vented enclosure route from a source checkout."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from vented3d.cli import main
if __name__=='__main__':raise SystemExit(main())
