"""Generation-only QA/MSG entry point with mixed-task batch support."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import data.datasets.city3dqa1_msg_generation_mixed_wrapper  # noqa: F401, E402
import hydra  # noqa: E402
from run_city3dqa1_qa_urban_coca_decoder_msg_caption import main as _original_main  # noqa: E402


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "configs" / "finetune"),
    config_name="city3dqa1_qa_urban_coca_decoder_pt",
)
def main(cfg):
    # Resolve configs relative to this entry point rather than treating the
    # imported original entry's relative path as a Python config package.
    return _original_main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
