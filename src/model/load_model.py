import transformers


def custom_from_pretrained(
    model_class,
    model_name_or_path,
    cache_dir=None,
    **kwargs,
):
    print(f'loading model: {model_name_or_path}')
    transformers.logging.set_verbosity_error()
    model, loading_info = model_class.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        output_loading_info=True,
        **kwargs,
    )
    transformers.logging.set_verbosity_warning()
    for k, v in loading_info.items():
        if v:
            print(f'{k}: {v}')
    print('load successfully')
    print('='*30)

    return model