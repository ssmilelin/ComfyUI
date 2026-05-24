# Poll download progress log + file sizes
$log = "D:\project\ComfyUI\models\wan22_download_progress.json"
$files = @{
  "text_encoder" = "D:\project\ComfyUI\models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "gguf_high"    = "D:\project\ComfyUI\models\unet\Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf"
  "gguf_low"     = "D:\project\ComfyUI\models\unet\Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf"
  "lora"         = "D:\project\ComfyUI\models\loras\wan\Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
}
$exp = @{ text_encoder=6.73; gguf_high=6.51; gguf_low=6.51; lora=0.63 }

while ($true) {
  $ts = Get-Date -Format "HH:mm:ss"
  Write-Output "=== $ts ==="
  if (Test-Path $log) {
    $j = Get-Content $log -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Output "overall: $($j.status)  current: $($j.current)"
    if ($j.error) { Write-Output "error: $($j.error)" }
  }
  foreach ($k in $files.Keys) {
    $p = $files[$k]
    if (Test-Path $p) {
      $gb = [math]::Round((Get-Item $p).Length/1GB, 2)
      $pct = [math]::Round(100 * $gb / $exp[$k], 1)
      Write-Output "$k : ${gb} GB ($pct%)"
    } else {
      $part = Get-ChildItem "D:\project\ComfyUI\models" -Recurse -Filter "*.incomplete" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'umt5|Wan2\.2' } |
        Sort-Object Length -Descending | Select-Object -First 1
      if ($part -and $part.Length -gt 0) {
        $gb = [math]::Round($part.Length/1GB, 2)
        Write-Output "$k : downloading cache ${gb} GB (partial)"
      } else {
        Write-Output "$k : pending"
      }
    }
  }
  if ((Test-Path $log) -and ((Get-Content $log -Raw | ConvertFrom-Json).status -in @("completed","error"))) { break }
  Start-Sleep -Seconds 60
}
