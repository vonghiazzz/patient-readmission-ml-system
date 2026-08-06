"""Privacy-safe exception handling shared by all API routes."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.schemas import APIError, ErrorDetail, ErrorResponse


class ModelUnavailableError(RuntimeError):
    """Raised when a prediction requires unavailable model artifacts."""


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=APIError(
            code=code,
            message=message,
            details=details or [],
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def validation_exception_handler(
    _request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """Return validation metadata without echoing the submitted values."""

    details = [
        ErrorDetail(
            location=list(error.get("loc", ())),
            message=str(error.get("msg", "Invalid value")),
            error_type=str(error.get("type", "validation_error")),
        )
        for error in exception.errors()
    ]
    return error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=details,
    )


async def model_unavailable_exception_handler(
    _request: Request,
    _exception: ModelUnavailableError,
) -> JSONResponse:
    return error_response(
        status_code=503,
        code="model_unavailable",
        message="The prediction model is unavailable.",
    )


async def http_exception_handler(
    _request: Request,
    exception: StarletteHTTPException,
) -> JSONResponse:
    if exception.status_code >= 500:
        return error_response(
            status_code=exception.status_code,
            code="request_failed",
            message="The request could not be completed.",
        )

    return error_response(
        status_code=exception.status_code,
        code="http_error",
        message=str(exception.detail),
    )


async def unhandled_exception_handler(
    _request: Request,
    _exception: Exception,
) -> JSONResponse:
    return error_response(
        status_code=500,
        code="internal_server_error",
        message="An unexpected internal error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the API's stable error response policy."""

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ModelUnavailableError, model_unavailable_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
