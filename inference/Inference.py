import time
import torch
import argparse

from torch2trt import torch2trt

from CNN import ErnNet
from utils import get_pytorch_model

class ModelPerformance:
    """The ModelPerformance class create an object that can be used
    for measuring the performance of a model"""

    def __init__(self, unoptim_model, input_data_batch):
        self.unoptim_model = unoptim_model
        self.input_data_batch = input_data_batch

    def get_trt_model(self, precision):
        """Reurns a tensorrt model based on the input precision"""

        assert precision in ['FP32', 'FP16', 'INT8'], "Only FP32, FP16 and INT8 precisions are supported."

        print(f"Generating a TensorRT optimised model in {precision} mode . . . .")

        fp16_mode = False
        int8_mode = False

        if precision == "FP16":
            fp16_mode = True
        elif precision == "INT8":
            int8_mode = True

        return torch2trt(
            self.unoptim_model,
            [self.input_data_batch],
            max_batch_size=len(self.input_data_batch),
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
        )

    def get_total_inference_time(self, model, num_loops):
        """Returns the total inference time on all the loops"""
        # Wait for all kernels in all streams on the CUDA device to complete.
        torch.cuda.current_stream().synchronize()

        t0 = time.time()
        for _ in range(num_loops):
            _ = model(self.input_data_batch)
            torch.cuda.current_stream().synchronize()
        t1 = time.time()

        return t1 - t0


    def get_throughput(self, num_loops, model):
        """Calculates the average number of images processed per second"""

        num_imgs = len(self.input_data_batch)
        total_inference_time_s = self.get_total_inference_time(model, num_loops)
        avg_img_inference_time_s = total_inference_time_s / (num_loops * num_imgs)

        print(f"Time elapsed per image {avg_img_inference_time_s} seconds.")

        throughput_images_per_second = 1 / avg_img_inference_time_s
        print(
            f"The throughput is {int(throughput_images_per_second)} images per second."
        )

        return throughput_images_per_second

    @staticmethod
    def get_throughput_boost(initial_throughput, final_throughput):
        """Return boost in the throughput"""
        throughput_boost = (final_throughput - initial_throughput) / initial_throughput
        print(f"\nThe throughput boost is x{round(throughput_boost, 2)}.\n")
        return throughput_boost


class PerfComparison(ModelPerformance):
    """Compares performance of models before and after optimisation by TensorRT"""

    def __init__(
        self, unoptim_model, input_data_batch, precisions=["FP32", "FP16", "INT8"]
    ):
        super().__init__(unoptim_model, input_data_batch)
        self.precisions = precisions

    def compare_throughput_boosts(self, num_loops):
        """Compare the throughput boost for each TensorRT model against the Pytorch model"""
        throughput_unoptim_model_imgs_per_sec = self.get_throughput(
            num_loops, self.unoptim_model
        )

        for precision in self.precisions:
            print(f"\n+++++++ {precision} Mode +++++++\n")

            trt_model = self.get_trt_model(precision)
            model_throughput_imgs_per_sec = self.get_throughput(num_loops, trt_model)
            self.get_throughput_boost(
                throughput_unoptim_model_imgs_per_sec, model_throughput_imgs_per_sec
            )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing e.g 16, 32, 64 and 128')
    parser.add_argument('--num_loops', type=int, default=1000, help='Number of loops used to calculate the average static')
    parser.add_argument('--model_object', help='Model object used for initiating the model')
    parser.add_argument('--weights', type=str, help='weigths.pt file path')
    parser.add_argument('--img_width', type=int, default=45, help='The width of images used for testing')
    parser.add_argument('--img_height', type=int, default=128, help='The height of images used for testing')
    return parser.parse_args()


def main(opt):
    
    batch_size = opt.batch_size
    num_loops = opt.num_loops
    dummy_input_batch = torch.rand((batch_size, 1, opt.img_height, opt.img_width)).cuda()

    # Load model and send to gpu
    unoptim_model = get_pytorch_model(opt.weights, ErnNet())
    unoptim_model.eval().cuda()

    # Compare model performances
    compare_perf = PerfComparison(unoptim_model, dummy_input_batch)
    compare_perf.compare_throughput_boosts(num_loops)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

