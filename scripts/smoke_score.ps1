# scripts/smoke_score.ps1
# -----------------------------------------------------------------------------
# End-to-end smoke test for the credit scoring API.
#
# Hits every endpoint of a locally-running uvicorn instance and writes the
# full request/response trail to scripts/smoke_score.log so the output can be
# reviewed (or shared) without scrolling back through a terminal session.
#
# Usage (from repo root, with the API already running on :8000):
#   .\scripts\smoke_score.ps1
#
# Start the API in a separate terminal first:
#   uv run uvicorn src.serving.app:app --reload
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Continue"

$LogPath = Join-Path $PSScriptRoot "smoke_score.log"
$BaseUrl = "http://127.0.0.1:8000"

# Reset the log file so each run starts clean.
"" | Set-Content -Path $LogPath -Encoding UTF8

function Write-Log {
    param([string]$Message)
    $Message | Tee-Object -FilePath $LogPath -Append | Out-Host
}

function Invoke-And-Log {
    param(
        [string]$Label,
        [string]$Method,
        [string]$Url,
        [string]$Body = $null
    )
    Write-Log ""
    Write-Log "============================================================"
    Write-Log "$Label"
    Write-Log "$Method $Url"
    Write-Log "============================================================"

    try {
        if ($Body) {
            Write-Log "REQUEST BODY:"
            Write-Log $Body
            Write-Log ""
            $response = Invoke-WebRequest -Uri $Url -Method $Method `
                -ContentType "application/json" -Body $Body `
                -UseBasicParsing -ErrorAction Stop
        } else {
            $response = Invoke-WebRequest -Uri $Url -Method $Method `
                -UseBasicParsing -ErrorAction Stop
        }
        Write-Log "STATUS: $($response.StatusCode) $($response.StatusDescription)"
        Write-Log "RESPONSE BODY:"
        Write-Log $response.Content
    } catch {
        # Non-2xx responses (e.g. 422, 503) land here in PowerShell.
        $err = $_.Exception.Response
        if ($err) {
            $status = [int]$err.StatusCode
            Write-Log "STATUS: $status $($err.StatusCode)"
            try {
                $stream = $err.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $errBody = $reader.ReadToEnd()
                Write-Log "RESPONSE BODY:"
                Write-Log $errBody
            } catch {
                Write-Log "ERROR BODY: (could not read response stream)"
            }
        } else {
            Write-Log "ERROR: $($_.Exception.Message)"
        }
    }
}

Write-Log "Credit Scoring API smoke test"
Write-Log "Run at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Base URL: $BaseUrl"

# ----- 1. Liveness ------------------------------------------------------------
Invoke-And-Log -Label "1. Liveness probe" -Method "GET" -Url "$BaseUrl/healthz"

# ----- 2. Readiness -----------------------------------------------------------
Invoke-And-Log -Label "2. Readiness probe" -Method "GET" -Url "$BaseUrl/readyz"

# ----- 3. Model info (proves feature_names_in_ is wired up) -------------------
Invoke-And-Log -Label "3. Model info (check feature_count > 0)" `
    -Method "GET" -Url "$BaseUrl/model_info"

# ----- 4. Metrics -------------------------------------------------------------
Invoke-And-Log -Label "4. Prometheus metrics" -Method "GET" -Url "$BaseUrl/metrics"

# ----- 5. Score a sample applicant -------------------------------------------
$payload = @{
    dummy_id              = "smoke-applicant-001"
    total_risk_score      = 55.0
    risk_drivers          = 28.0
    risk_mitigators       = 22.0
    product_type          = "personal_loan"
    num_accounts_assess   = 3
    worst_arrears_assess  = 1
    age_oldest_assess     = 60
} | ConvertTo-Json

Invoke-And-Log -Label "5. Score a sample applicant" `
    -Method "POST" -Url "$BaseUrl/v1/score" -Body $payload

Write-Log ""
Write-Log "============================================================"
Write-Log "Smoke test complete. Log written to: $LogPath"
Write-Log "============================================================"
