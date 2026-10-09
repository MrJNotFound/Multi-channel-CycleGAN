"""Model factory for explicitly registered compositions.

Implementations use architecture-prefixed filenames and contain their own losses.
See registry.py and README.md for naming and template_model.py for the model
interface. Runtime implementations are never exposed as command-line aliases.
"""

import importlib
from .registry import IMPLEMENTATIONS, composition_errors, model_spec


def find_model_using_name(model_name: str):
    """Resolve a registered composition or the generator-only inference runner."""
    if model_name == "test":
        from .test_model import TestModel
        return TestModel
    spec = model_spec(model_name)
    module_path, class_name = IMPLEMENTATIONS[spec.implementation]
    return getattr(importlib.import_module(module_path), class_name)


def get_option_setter(model_name: str):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    if model_name == "test":
        return model_class.modify_commandline_options
    spec = model_spec(model_name)

    def configure(parser, is_train=True):
        parser = model_class.modify_commandline_options(parser, is_train)
        if spec.fusion == "fusion":
            parser.add_argument("--spif_version", choices=("v1", "v2"), default="v2",
                                help="SPIF v1: channel fusion; v2: channel-spatial fusion")
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
