param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$articleDir = Join-Path $repoRoot 'artículo'
$texFile = 'RL_Car_Racing_Grupo4.tex'
$jobName = 'RL_Car_Racing_Grupo4'

if (-not (Test-Path $articleDir)) {
    throw "No se encontró la carpeta del artículo: $articleDir"
}

Push-Location $articleDir
try {
    if ($Clean) {
        $patterns = @(
            "$jobName.aux",
            "$jobName.bbl",
            "$jobName.blg",
            "$jobName.fls",
            "$jobName.log",
            "$jobName.out",
            "$jobName.fdb_latexmk",
            "$jobName.fls",
            "$jobName.synctex.gz",
            "$jobName.pdf"
        )

        foreach ($p in $patterns) {
            if (Test-Path $p) {
                Remove-Item -Force $p
            }
        }
    }

    & pdflatex -interaction=nonstopmode -halt-on-error $texFile
    if ($LASTEXITCODE -ne 0) { throw "pdflatex falló (1er pase)" }

    & bibtex "$jobName.aux"
    if ($LASTEXITCODE -ne 0) { throw "bibtex falló" }

    & pdflatex -interaction=nonstopmode -halt-on-error $texFile
    if ($LASTEXITCODE -ne 0) { throw "pdflatex falló (2do pase)" }

    & pdflatex -interaction=nonstopmode -halt-on-error $texFile
    if ($LASTEXITCODE -ne 0) { throw "pdflatex falló (3er pase)" }

    Write-Host "OK: generado $jobName.pdf en $articleDir" -ForegroundColor Green
}
finally {
    Pop-Location
}
