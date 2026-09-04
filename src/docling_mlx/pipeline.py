# SPDX-License-Identifier: Apache-2.0

"""The one small bridge for Docling stages that have no factory hook."""

from __future__ import annotations

from typing import Any, cast

from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def configure(pipeline: StandardPdfPipeline) -> None:
    """Replace Docling's non-pluggable MLX enrichment stages in-place."""

    from docling.models.base_model import GenericEnrichmentModel
    from docling.models.stages.picture_classifier.document_picture_classifier import (
        DocumentPictureClassifier,
    )

    from docling_mlx.stages.chart_extraction import (
        MlxChartExtractionModelOptions,
        MlxGraniteVisionChartExtractionModel,
    )
    from docling_mlx.stages.picture_classification import (
        MlxDocumentPictureClassifier,
        MlxDocumentPictureClassifierOptions,
    )

    options = pipeline.pipeline_options
    picture_options = options.picture_classification_options
    chart_options = options.chart_extraction_options

    if isinstance(picture_options, MlxDocumentPictureClassifierOptions):
        replacement = MlxDocumentPictureClassifier(
            enabled=options.do_picture_classification or options.do_chart_extraction,
            artifacts_path=pipeline.artifacts_path,
            options=picture_options,
            accelerator_options=options.accelerator_options,
            enable_remote_services=options.enable_remote_services,
        )
        for index, stage in enumerate(pipeline.enrichment_pipe):
            if type(stage) in {DocumentPictureClassifier, MlxDocumentPictureClassifier}:
                pipeline.enrichment_pipe[index] = cast(GenericEnrichmentModel[Any], replacement)
                break
        else:
            raise RuntimeError("Docling picture classifier was not initialized")

    if isinstance(chart_options, MlxChartExtractionModelOptions):
        pipeline.enrichment_pipe = [
            stage
            for stage in pipeline.enrichment_pipe
            if type(stage) is not MlxGraniteVisionChartExtractionModel
            and type(stage).__name__
            not in {"ChartExtractionModelGraniteVision", "ChartExtractionModelGraniteVisionV4"}
        ]
        pipeline.enrichment_pipe.append(
            MlxGraniteVisionChartExtractionModel(
                enabled=options.do_chart_extraction,
                artifacts_path=pipeline.artifacts_path,
                options=chart_options,
                accelerator_options=options.accelerator_options,
            )
        )

    pipeline.keep_backend = any(
        (
            options.do_formula_enrichment,
            options.do_code_enrichment,
            options.do_picture_classification,
            options.do_picture_description,
            options.do_chart_extraction,
        )
    )


class MlxStandardPdfPipeline(StandardPdfPipeline):
    """StandardPdfPipeline with only the two absent plugin hooks configured."""

    def __init__(self, pipeline_options: Any) -> None:
        from docling_mlx.stages.chart_extraction import MlxChartExtractionModelOptions
        from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifierOptions

        self._mlx_pipeline_options = pipeline_options
        chart_options = pipeline_options.chart_extraction_options
        picture_options = pipeline_options.picture_classification_options
        self._mlx_bootstrap = isinstance(
            picture_options, MlxDocumentPictureClassifierOptions
        ) or isinstance(chart_options, MlxChartExtractionModelOptions)
        if isinstance(chart_options, MlxChartExtractionModelOptions) and not isinstance(
            picture_options, MlxDocumentPictureClassifierOptions
        ):
            raise ValueError("Mlx chart extraction requires MlxDocumentPictureClassifierOptions")
        bootstrap = (
            pipeline_options.model_copy(
                deep=True,
                update={
                    "do_picture_classification": pipeline_options.do_picture_classification
                    and not isinstance(picture_options, MlxDocumentPictureClassifierOptions),
                    "do_chart_extraction": pipeline_options.do_chart_extraction
                    and not isinstance(chart_options, MlxChartExtractionModelOptions),
                },
            )
            if self._mlx_bootstrap
            else pipeline_options
        )
        super().__init__(bootstrap)

    def _init_models(self) -> None:
        super()._init_models()
        if self._mlx_bootstrap:
            self.pipeline_options = self._mlx_pipeline_options
        configure(self)


__all__ = ["MlxStandardPdfPipeline", "configure"]
