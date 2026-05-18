from .dataloader import FileDataloader
from .writer import FileWriter


def create_dataloader_and_writer(
    infer_args, tokenizer, template, processor=None, model_dtype=None, num_return=1, output_type=None
):
    if infer_args.load_from == "file":
        return FileDataloader(infer_args, tokenizer, template, processor, model_dtype, output_type), FileWriter(
            infer_args.outputs
        )
    else:
        raise ValueError(f"Unknown load_from: {infer_args.load_from}")
