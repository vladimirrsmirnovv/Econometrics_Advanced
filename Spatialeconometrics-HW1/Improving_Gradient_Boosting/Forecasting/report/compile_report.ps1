# Build the final report through a wrapper without editing the user's source.
$ErrorActionPreference = 'Stop'
$taskReportRoot = $PSScriptRoot
$taskWorkspaceRoot = [IO.Path]::GetFullPath((Join-Path $taskReportRoot '..\..'))
$taskBuildRoot = [IO.Path]::GetFullPath((Join-Path $taskWorkspaceRoot 'tmp\pdfs\final_forecast_report\build'))
$taskPdfTarget = Join-Path $taskReportRoot 'output\pdf\Прогноз_РТО_финальный_отчёт.pdf'
foreach ($taskTarget in @($taskBuildRoot, $taskPdfTarget)) {
    if (-not $taskTarget.StartsWith($taskWorkspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Build path is outside the workspace: $taskTarget"
    }
}
foreach ($taskTable in @('development.tex', 'confirmation.tex', 'final_individual.tex', 'moran.tex')) {
    if (-not (Test-Path -LiteralPath (Join-Path $taskReportRoot "tables\$taskTable"))) {
        throw "Required table is missing: $taskTable"
    }
}
$taskSourcePath = Join-Path $taskReportRoot 'Прогноз_РТО_финальный_отчёт.tex'
$taskSourceHash = (Get-FileHash -LiteralPath $taskSourcePath -Algorithm SHA256).Hash
New-Item -ItemType Directory -Path $taskBuildRoot -Force | Out-Null
$taskEncoding = [Text.UTF8Encoding]::new($false)
$taskWrapper = @'
\AtBeginDocument{\input{../../tmp/pdfs/final_forecast_report/build/table_input_compat.tex}}
\input{Прогноз_РТО_финальный_отчёт.tex}
'@
$taskCompat = @'
\makeatletter
\let\ReportOriginalIfFileExists\IfFileExists
\expandafter\let\csname ReportTable@tables/development.tex\endcsname\@empty
\expandafter\let\csname ReportTable@tables/confirmation.tex\endcsname\@empty
\expandafter\let\csname ReportTable@tables/final_individual.tex\endcsname\@empty
\expandafter\let\csname ReportTable@tables/moran.tex\endcsname\@empty
\long\def\ReportKnownTable#1#2#3{\@@input #1 }
\long\def\IfFileExists#1{%
  \ifcsname ReportTable@#1\endcsname
    \expandafter\ReportKnownTable
  \else
    \expandafter\ReportOriginalIfFileExists
  \fi{#1}}
\makeatother
'@
$taskWrapperPath = Join-Path $taskBuildRoot 'compile_final.tex'
[IO.File]::WriteAllText($taskWrapperPath, $taskWrapper, $taskEncoding)
[IO.File]::WriteAllText((Join-Path $taskBuildRoot 'table_input_compat.tex'), $taskCompat, $taskEncoding)
Push-Location -LiteralPath $taskReportRoot
try {
    for ($taskPass = 1; $taskPass -le 2; $taskPass++) {
        & pdflatex -disable-installer -disable-write18 -interaction=nonstopmode -halt-on-error -jobname=final_report "-output-directory=$taskBuildRoot" $taskWrapperPath
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $taskPass failed" }
    }
    if ((Get-FileHash -LiteralPath $taskSourcePath -Algorithm SHA256).Hash -ne $taskSourceHash) {
        throw 'Source changed during compilation'
    }
    Move-Item -LiteralPath (Join-Path $taskBuildRoot 'final_report.pdf') -Destination $taskPdfTarget -Force
    Write-Output "Final PDF: $taskPdfTarget"
}
finally { Pop-Location }
