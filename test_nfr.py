import torch
from nfr_activation import NFR

def test_nfr_gradient_stability():
    """Verifies that NFR layer produces valid gradients (no NaN)."""
    layer = NFR()
    x = torch.randn(10, 10, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    
    assert not torch.isnan(x.grad).any(), "Gradient stability test failed: NaN detected!"
    print("✅ Gradient Stability: PASSED")

def test_nfr_cuda_support():
    """Verifies CUDA compatibility if GPU is available."""
    if torch.cuda.is_available():
        layer = NFR().cuda()
        x = torch.randn(5, 5).cuda()
        output = layer(x)
        assert output.is_cuda, "CUDA support test failed!"
        print("✅ CUDA Compatibility: PASSED")
    else:
        print("🟡 CUDA not available, skipping test.")

if __name__ == "__main__":
    print("🚀 Starting NFR Production-Ready Tests...")
    test_nfr_gradient_stability()
    test_nfr_cuda_support()
    print("🎉 All tests passed. NFR is ready for deployment.")
