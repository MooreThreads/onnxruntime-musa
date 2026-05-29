import onnxruntime as ort
import onnxruntime_musa as musa_ep

ort.register_execution_provider_library(musa_ep.get_ep_name(), musa_ep.get_library_path())
print(f"registered {musa_ep.get_ep_name()} from {musa_ep.get_library_path()}")
