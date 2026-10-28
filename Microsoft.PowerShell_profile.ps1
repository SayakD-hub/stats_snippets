function prompt {
    $previousExit = $LASTEXITCODE
    $path = $executionContext.SessionState.Path.CurrentLocation
    Write-Host "PS $path" -NoNewline -ForegroundColor Cyan

    $gitBranch = git rev-parse --abbrev-ref HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $gitBranch) {
        Write-Host " ($gitBranch)" -NoNewline -ForegroundColor Green
    }

    Write-Host ">" -NoNewline -ForegroundColor Yellow
    $global:LASTEXITCODE = $previousExit
    return " "
}

function Get-ListingColor {
    param($Item)
    if ($Item.PSIsContainer) {
        return "Cyan"
    }
    switch -Regex ($Item.Extension) {
        "^\.(exe|bat|cmd|ps1|psm1|msi|com)$" { return "Green" }
        "^\.(zip|7z|gz|tar|rar|tgz)$" { return "Red" }
        "^\.(py|pyi|js|ts|tsx|jsx|cs|cpp|c|h|java|json|xml|yml|yaml|md|csv|toml|ini)$" { return "Yellow" }
        default { return "Gray" }
    }
}

function Show-ColorChildItem {
    $passthrough = $MyInvocation.PipelinePosition -lt $MyInvocation.PipelineLength
    $items = @(Get-ChildItem @args)
    if ($passthrough) {
        return $items
    }

    $lastParent = $null
    foreach ($item in $items) {
        if ($item -is [string]) {
            Write-Host $item
            continue
        }

        $parent = ""
        if ($item.PSParentPath) {
            $parent = $item.PSParentPath -replace "^Microsoft\.PowerShell\.Core\\FileSystem::", ""
        }
        if ($parent -ne $lastParent) {
            Write-Host ""
            Write-Host "    Directory: $parent" -ForegroundColor DarkCyan
            Write-Host ""
            Write-Host ("{0,-7}  {1,22}  {2,12}  {3}" -f "Mode", "LastWriteTime", "Length", "Name")
            Write-Host ("{0,-7}  {1,22}  {2,12}  {3}" -f "----", "-------------", "------", "----")
            $lastParent = $parent
        }

        $date = "{0:MM/dd/yyyy  hh:mm tt}" -f $item.LastWriteTime
        $length = ""
        if (-not $item.PSIsContainer) {
            $length = $item.Length
        }
        Write-Host ("{0,-7}  {1,22}  {2,12}  " -f $item.Mode, $date, $length) -NoNewline
        Write-Host $item.Name -ForegroundColor (Get-ListingColor $item)
    }
}

Remove-Item Alias:ls -Force -ErrorAction SilentlyContinue
Remove-Item Alias:dir -Force -ErrorAction SilentlyContinue
Set-Alias ls Show-ColorChildItem
Set-Alias dir Show-ColorChildItem
