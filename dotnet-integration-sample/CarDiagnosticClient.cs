using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace CarRagService.Client;

/*
==================================================================================
HOW TO REGISTER THIS SERVICE IN .NET (Program.cs / Startup.cs)
==================================================================================

1. Add base URL configuration in appsettings.json:
   {
     "CarRagService": {
       "BaseUrl": "http://localhost:8001"
     }
   }

2. Register the typed client in Program.cs (.NET 6 / 7 / 8 / 9 Minimal APIs or Controllers):

   builder.Services.AddHttpClient<ICarDiagnosticClient, CarDiagnosticClient>(client =>
   {
       var baseUrl = builder.Configuration["CarRagService:BaseUrl"] ?? "http://localhost:8001";
       client.BaseAddress = new Uri(baseUrl);
       client.Timeout = TimeSpan.FromSeconds(60);
   });

3. Inject and use ICarDiagnosticClient wherever needed:

   app.MapPost("/api/car-diagnosis", async (ICarDiagnosticClient diagnosticClient, DiagnoseRequestDto request) =>
   {
       var result = await diagnosticClient.DiagnoseAsync(request);
       return Results.Ok(result);
   });

==================================================================================
*/

/// <summary>
/// Client contract for querying the Car Diagnostic RAG Microservice.
/// </summary>
public interface ICarDiagnosticClient
{
    /// <summary>
    /// Sends a diagnostic question with car make/model to retrieve an AI-generated diagnosis
    /// based on vector similarity search over automotive fault records.
    /// </summary>
    /// <param name="request">Car diagnostic query containing question, make, model, and top_k.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>DiagnoseResponseDto containing answer, llm_used, and matched_faults.</returns>
    Task<DiagnoseResponseDto> DiagnoseAsync(DiagnoseRequestDto request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Queries the health endpoint of the Car Diagnostic RAG service.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>HealthResponseDto containing status, records_loaded, and vectors_ready.</returns>
    Task<HealthResponseDto> GetHealthAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Typed HTTP client implementation for Car Diagnostic RAG Microservice.
/// </summary>
public class CarDiagnosticClient : ICarDiagnosticClient
{
    private readonly HttpClient _httpClient;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public CarDiagnosticClient(HttpClient httpClient)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    /// <inheritdoc />
    public async Task<DiagnoseResponseDto> DiagnoseAsync(DiagnoseRequestDto request, CancellationToken cancellationToken = default)
    {
        if (request == null)
            throw new ArgumentNullException(nameof(request));

        using var response = await _httpClient.PostAsJsonAsync("/api/v1/diagnose", request, JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Car RAG service returned HTTP {(int)response.StatusCode} ({response.ReasonPhrase}): {errorBody}",
                inner: null,
                statusCode: response.StatusCode);
        }

        var result = await response.Content.ReadFromJsonAsync<DiagnoseResponseDto>(JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        return result ?? throw new InvalidOperationException("Failed to deserialize DiagnoseResponseDto: response body was null.");
    }

    /// <inheritdoc />
    public async Task<HealthResponseDto> GetHealthAsync(CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.GetAsync("/health", cancellationToken).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            throw new HttpRequestException(
                $"Car RAG service health check failed with HTTP {(int)response.StatusCode}: {errorBody}",
                inner: null,
                statusCode: response.StatusCode);
        }

        var result = await response.Content.ReadFromJsonAsync<HealthResponseDto>(JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        return result ?? throw new InvalidOperationException("Failed to deserialize HealthResponseDto: response body was null.");
    }
}
