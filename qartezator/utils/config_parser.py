import pydoc


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop('type')
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def render_config(config, settings):
    """Render configuration."""
    if config is None:
        return config
    elif isinstance(config, str):
        return config.format(**settings)
    elif isinstance(config, list) or isinstance(config, tuple):
        return [render_config(item, settings) for item in config]
    elif isinstance(config, dict):
        return {render_config(key, settings): render_config(value,  settings)
                for key, value in config.items()}
    else:
        return config
