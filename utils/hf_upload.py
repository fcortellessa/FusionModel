from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="./Data",
    repo_id="master-flavia/GripFusionNet-Data",
    repo_type="dataset",
    token="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"
)