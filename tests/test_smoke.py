import torch
import dpocurv
from dpocurv.utils.seed import set_seed
from dpocurv.data.splits import get_splits
from dpocurv.models.policy import load_policy
from dpocurv.training.losses import dpo_loss, curvature_loss

def test_imports():
    print(f"dpocurv version: {dpocurv.__version__}")
    assert dpocurv.__version__ == "0.1.0"

def test_cuda():
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    test_imports()
    test_cuda()
    print("All smoke tests passed.")
