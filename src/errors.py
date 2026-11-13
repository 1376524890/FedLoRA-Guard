"""Typed errors. Failures are deliberately never converted into passing metrics."""


class ExperimentError(RuntimeError):
    """A scientific or execution precondition failed."""


class ConfigurationError(ExperimentError):
    """The configuration is invalid, unresolved, or changed."""


class DataValidationError(ExperimentError):
    """Source data or data isolation failed validation."""


class CheckpointError(ExperimentError):
    """A checkpoint is absent, incomplete, corrupt, or incompatible."""


class DetectorInputError(ExperimentError):
    """A detector received forbidden, missing, or degenerate inputs."""


class ExternalServiceError(ExperimentError):
    """An authenticated external service request or schema validation failed."""
