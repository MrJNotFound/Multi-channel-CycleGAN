"""Model factory for explicit compositions, legacy names and custom plugins.

Implementations are grouped by architecture; independent losses live in losses/.
See registry.py and README.md for naming and examples/template.py for the model
interface. Old models/<name>_model imports remain lazily compatible.
"""

import importlib
from models.base_model import BaseModel
from .registry import IMPLEMENTATIONS, MODEL_SPECS, composition_errors, legacy_name
from .compat import install_legacy_imports

install_legacy_imports()


def find_model_using_name(model_name: str):
    """Resolve a registered composition/alias, then try the old plugin convention."""
    implementation = IMPLEMENTATIONS.get(legacy_name(model_name))
    if implementation:
        module_path, class_name = implementation
        return getattr(importlib.import_module(module_path), class_name)

    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and isinstance(cls, type) and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise ImportError(f"{model_filename} must define a BaseModel subclass named {target_model_name}.")

    return model


def get_option_setter(model_name: str):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        return model_class.modify_commandline_options

    def configure(parser, is_train=True):
        parser = model_class.modify_commandline_options(parser, is_train)
        defaults = {"output_nc": 3, **spec.defaults}
        defaults["dataset_mode"] = (
            "colorization" if spec.adapter == "colorization" else
            "dual_channel" if spec.modalities == "dual" else
            "aligned" if spec.architecture == "pix2pix" else "unaligned"
        )
        # Only set options registered for this phase (loss flags are train-only).
        destinations = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in defaults.items() if key in destinations})
        return parser

    return configure


def create_model(opt):
    """Create a model given the option."""
    name = getattr(opt, "source_model", "") if opt.model == "test" else opt.model
    errors = composition_errors(name, vars(opt))
    if errors:
        raise ValueError("\n".join(errors))
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance
