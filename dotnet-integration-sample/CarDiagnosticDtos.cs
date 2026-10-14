using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CarRagService.Client;

/// <summary>
/// Request payload sent to POST /api/v1/diagnose
/// </summary>
public record DiagnoseRequestDto(
    [property: JsonPropertyName("question")] string Question,
    [property: JsonPropertyName("make")] string Make = "BYD",
    [property: JsonPropertyName("model")] string Model = "F3",
    [property: JsonPropertyName("top_k")] int TopK = 2
);

/// <summary>
/// Matched fault record retrieved from the pre-computed vector index.
/// </summary>
public record MatchedFaultDto(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("make")] string Make,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("system")] string System,
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("severity")] string Severity,
    [property: JsonPropertyName("drive_advice")] string DriveAdvice,
    [property: JsonPropertyName("obd_codes")] IReadOnlyList<string> ObdCodes,
    [property: JsonPropertyName("score")] double Score
);

/// <summary>
/// Response payload returned by POST /api/v1/diagnose
/// </summary>
public record DiagnoseResponseDto(
    [property: JsonPropertyName("answer")] string Answer,
    [property: JsonPropertyName("llm_used")] string LlmUsed,
    [property: JsonPropertyName("matched_faults")] IReadOnlyList<MatchedFaultDto> MatchedFaults
);

/// <summary>
/// Response payload returned by GET /health
/// </summary>
public record HealthResponseDto(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("records_loaded")] int RecordsLoaded,
    [property: JsonPropertyName("vectors_ready")] bool VectorsReady
);
