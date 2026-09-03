import argparse
import os
import onnx
from onnxconverter_common import float16

def fix_cast_nodes(model):
    # the converter retypes tensors to float16 but leaves Cast 'to' attributes
    # at float32, which makes onnxruntime reject the model on load
    graph = model.graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    for v in list(graph.input) + list(graph.output):
        types[v.name] = v.type.tensor_type.elem_type
    for node in graph.node:
        if node.op_type != 'Cast':
            continue
        expected = types.get(node.output[0])
        for attr in node.attribute:
            if attr.name == 'to' and expected is not None and attr.i != expected:
                attr.i = expected


def convert_to_float16(model_path):
    model = onnx.load(model_path)
    result_file = os.path.splitext(model_path)[0]
    result_extension = os.path.splitext(model_path)[1]
    model_converted = result_file + "_fp16" + result_extension
    model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False, disable_shape_infer=False, op_block_list=None, node_block_list=None)
    fix_cast_nodes(model_fp16)
    onnx.save(model_fp16, model_converted)
    return model_converted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an ONNX model to float16.')
    parser.add_argument('model_path', help='path to the .onnx model to convert')
    args = parser.parse_args()
    print(convert_to_float16(args.model_path))