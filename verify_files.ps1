$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$manifestPath = Join-Path $packageRoot 'FILE_MANIFEST.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$issues = [System.Collections.Generic.List[string]]::new()
foreach ($entry in $manifest.files) {
    $filePath = Join-Path $packageRoot $entry.path
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        $issues.Add("Missing: $($entry.path)")
        continue
    }
    $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) { $issues.Add("Changed: $($entry.path)") }
}
if ($issues.Count -gt 0) {
    $issues | ForEach-Object { Write-Output $_ }
    throw "$($issues.Count) file integrity check(s) failed. Rerunning experiments may legitimately change result files."
}
Write-Output "PASS: $($manifest.files.Count) packaged files match their SHA-256 values."
Write-Output 'FILE_MANIFEST.json and CHECKSUMS.sha256 are excluded from their own file list.'
