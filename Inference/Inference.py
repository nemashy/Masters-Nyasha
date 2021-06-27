import sys
sys.path.append('C:/Users/nyasha/Desktop/Masters-Nyasha/Models')
from CNN import ErnNet
import torch
import trtorch


network = ErnNet()
checkpoint = torch.load('checkpoint.pt')
network.load_state_dict(checkpoint)

network.cuda().eval()
script_model = torch.jit.script(network)

spec = {
    "forward":
        trtorch.TensorRTCompileSpec({
            "input_shapes": [[16, 1, 128, 45]],
            "op_precision": torch.half,
            "refit": False,
            "debug": False,
            "strict_types": False,
            "device": {
                "device_type": trtorch.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True
            },
            "capability": trtorch.EngineCapability.default,
            "num_min_timing_iters": 2,
            "num_avg_timing_iters": 1,
            "max_batch_size": 0,
        })
    }