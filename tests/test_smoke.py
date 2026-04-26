import torch
import curvature_dpo
from curvature_dpo.utils.seed import set_seed
from curvature_dpo.data.splits import get_splits
from curvature_dpo.models.policy import load_policy
from curvature_dpo.training.functional import dpo_loss, curvature_loss

def test_imports():
    print(f"curvature_dpo version: {curvature_dpo.__version__}")
    assert curvature_dpo.__version__ == "0.1.0"

def test_cuda():
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    test_imports()
    test_cuda()
    print("All smoke tests passed.")
