$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$python = "$env:USERPROFILE\anaconda3\envs\adaptive-spring-network\python.exe"
& $python -m pytest `
  spring-network\tests\test_mechanics_3d.py `
  spring-network\tests\test_profile_generator.py `
  spring-network\tests\test_profile_conditioned_passive.py `
  -q
