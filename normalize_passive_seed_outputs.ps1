param([int[]]$Seeds = @(202, 303))

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Join-Path $repo "spring-network\models\profile_conditioned_passive_3d"
$tableDir = Join-Path $repo "spring-network\tables\profile_conditioned_passive_3d"

foreach ($seed in $Seeds) {
  $oldStem = "profile_passive_3d_60spring_full_refresh_seed$seed"
  $newStem = "passive_60spring_seed$seed"
  $oldModel = Join-Path $modelDir "$oldStem.npz"
  if (Test-Path -LiteralPath $oldModel) {
    Move-Item -LiteralPath $oldModel -Destination (Join-Path $modelDir "$newStem.npz")
  }
  Get-ChildItem $tableDir -Filter "$oldStem*" -File -ErrorAction SilentlyContinue |
    ForEach-Object {
      $newName = $_.Name -replace "^$oldStem", $newStem
      Move-Item -LiteralPath $_.FullName -Destination (Join-Path $tableDir $newName)
    }
}
