import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tail_repair_qm import parse_args


class TestRunTailRepairQmArgs:
    def test_default_args(self):
        args = parse_args([])
        assert args.variant == "same_period"
        assert args.out_csv == Path("results/tail_repair_qm.csv")

    def test_custom_variant(self):
        args = parse_args(["--variant", "climatology"])
        assert args.variant == "climatology"
