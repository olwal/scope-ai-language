from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import VLMOllamaPipeline, VLMOllamaPrePipeline, VLMOllamaPostPipeline

    register(VLMOllamaPipeline)
    register(VLMOllamaPrePipeline)
    register(VLMOllamaPostPipeline)
