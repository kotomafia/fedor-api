param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('pull', 'push')]
    [string]$Direction = 'pull'
)

$ErrorActionPreference = 'Stop'

# В fedor-api единственный subtree — ml/ ← fedor-ml.
$prefix = 'ml'
$remote = 'fedor-ml'

Write-Host ""
Write-Host "=== $Direction $prefix <-> $remote/main ===" -ForegroundColor Cyan

if ($Direction -eq 'pull') {
    git subtree pull --prefix=$prefix $remote main --squash
} else {
    git subtree push --prefix=$prefix $remote main
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка при $Direction для $prefix" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Готово." -ForegroundColor Green
