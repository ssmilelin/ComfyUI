# Start 3 parallel Wan 2.2 model downloads
$py = "D:\project\ComfyUI\.venv\Scripts\python.exe"
$script = "D:\project\ComfyUI\scripts\wan22_download_one.py"
$logDir = "D:\project\ComfyUI\models"
$jobs = @("text_encoder", "gguf_high", "gguf_low")

$state = @{
  started_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
  status     = "running"
  jobs       = @{}
  parallel   = $true
}
foreach ($j in $jobs) {
  $state.jobs[$j] = @{ status = "queued" }
}
$state | ConvertTo-Json -Depth 5 | Set-Content "$logDir\wan22_download_progress.json" -Encoding UTF8

foreach ($j in $jobs) {
  $out = "$logDir\wan22_dl_$j.log"
  $err = "$logDir\wan22_dl_$j.err.log"
  Start-Process -FilePath $py -ArgumentList $script, $j -WorkingDirectory "D:\project\ComfyUI" `
    -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
  Write-Output "started $j -> $out"
}
