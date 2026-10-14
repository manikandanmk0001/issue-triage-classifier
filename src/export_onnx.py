import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from common import MODELS

model = joblib.load(MODELS / "model.joblib")
classifier = model.steps[-1][1]


onnx_model = convert_sklearn(
    model,
    initial_types=[("text", StringTensorType([None, 1]))],
    options={id(classifier): {"zipmap": False}},   # return a plain array of probabilities
)
(MODELS / "model.onnx").write_bytes(onnx_model.SerializeToString())
print("Saved models/model.onnx")