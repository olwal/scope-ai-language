from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import TextLogPipeline

    register(TextLogPipeline)
