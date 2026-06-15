"""
Pydantic schemas for API request/response models.
Defines the contract between the client and the FAQ assistant API.
"""

from typing import Optional

from pydantic import BaseModel, Field


# --- Request Schemas ---

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User's question about a mutual fund scheme",
        examples=["What is the expense ratio of HDFC Mid Cap Fund?"],
    )
    scheme_filter: Optional[str] = Field(
        default=None,
        description="Optional scheme name to filter retrieval (e.g., 'HDFC Mid Cap Fund')",
        examples=["HDFC Mid Cap Fund"],
    )
    document_type_filter: Optional[str] = Field(
        default=None,
        description="Optional document type filter (e.g., 'factsheet', 'scheme_page')",
        examples=["scheme_page"],
    )


# --- Response Schemas ---

class QueryResponse(BaseModel):
    """Successful factual response from the FAQ assistant."""
    answer: str = Field(
        ...,
        description="Generated factual response (max 3 sentences)",
    )
    source_url: str = Field(
        ...,
        description="Primary source URL for the answer",
    )
    last_updated: str = Field(
        ...,
        description="Date when sources were last updated (YYYY-MM-DD)",
    )
    intent: str = Field(
        ...,
        description="Classified intent of the query",
        examples=["factual"],
    )
    scheme: Optional[str] = Field(
        default=None,
        description="Matched scheme name, if applicable",
    )
    context_used: int = Field(
        default=0,
        description="Number of context chunks used for generation",
    )
    latency_ms: float = Field(
        default=0.0,
        description="End-to-end response latency in milliseconds",
    )


class RefusalResponse(BaseModel):
    """Refusal response for advisory or out-of-scope queries."""
    answer: str = Field(
        ...,
        description="Polite refusal message",
    )
    educational_link: str = Field(
        ...,
        description="Link to investor education resource",
    )
    last_updated: str = Field(
        ...,
        description="Date footer",
    )
    intent: str = Field(
        ...,
        description="Classified intent (advisory or out_of_scope)",
    )


class ErrorResponse(BaseModel):
    """Error response for malformed requests or internal failures."""
    error: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    vector_store_count: int = Field(
        default=0,
        description="Number of chunks in the vector store",
    )
    llm_available: bool = Field(
        default=False,
        description="Whether the LLM service is reachable",
    )


class SchemeInfo(BaseModel):
    """Information about a supported mutual fund scheme."""
    name: str
    category: str
    groww_url: str


class SchemeListResponse(BaseModel):
    """Response for the /schemes endpoint."""
    amc_name: str
    scheme_count: int
    schemes: list[SchemeInfo]


class QueryResponseEnvelope(BaseModel):
    """
    Unified response envelope for /query endpoint.
    Contains either a factual answer or a refusal, plus metadata.
    """
    answer: str
    source_url: Optional[str] = None
    educational_link: Optional[str] = None
    last_updated: str
    intent: str
    scheme: Optional[str] = None
    is_refusal: bool = False
    context_used: int = 0
    latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    pii_detected: list[str] = Field(default_factory=list)
