"""Write the example lawns (lawns/*.json) and the default mower (mower.json)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mowersim import Mower, example_lawns  # noqa: E402

for name, lawn in example_lawns().items():
    path = ROOT / "lawns" / f"{name}.json"
    path.write_text(json.dumps(lawn.to_dict()) + "\n")
    print("wrote", path, f"({lawn.area:.1f} m²)")
(ROOT / "mower.json").write_text(json.dumps(Mower().to_dict(), indent=1) + "\n")
print("wrote", ROOT / "mower.json")
