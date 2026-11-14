param(
    [string]$Version = $(if ($env:SWIZZLE_VERSION) { $env:SWIZZLE_VERSION } else { "v0.1.0" }),
    [string]$InstallDir = $(if ($env:SWIZZLE_INSTALL_DIR) { $env:SWIZZLE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Swizzle\bin" })
)

$ErrorActionPreference = "Stop"
$Repo = "Swizzle-AI/swizzle-runtime"
$VerNum = $Version.TrimStart("v")
$Archive = "swizzle-rt-$VerNum-windows-x86_64-cuda.zip"
$Url = "https://github.com/$Repo/releases/download/$Version/$Archive"

$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("swizzle-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TmpDir | Out-Null

try {
    $ZipPath = Join-Path $TmpDir $Archive
    Write-Host "Downloading Swizzle Runtime $Version ($Archive)..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
    } catch {
        Write-Warning "Prebuilt Windows archive $Archive was not found for $Version."
        Write-Host "Build from source on Windows with:"
        Write-Host "  git clone https://github.com/$Repo.git; cd swizzle-runtime; cargo build --release --locked -p swizzle-cli"
        exit 1
    }

    Expand-Archive -Path $ZipPath -DestinationPath $TmpDir -Force
    $Bin = Get-ChildItem -Path $TmpDir -Filter "swizzle-rt.exe" -Recurse | Select-Object -First 1
    if (-not $Bin) {
        throw "swizzle-rt.exe not found inside $Archive."
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $Target = Join-Path $InstallDir "swizzle-rt.exe"
    Copy-Item -Path $Bin.FullName -Destination $Target -Force
    Write-Host "Installed swizzle-rt.exe to $Target"

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$InstallDir*") {
        $NewPath = if ($UserPath) { "$UserPath;$InstallDir" } else { $InstallDir }
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        Write-Host "Added $InstallDir to your User PATH (restart your terminal to use 'swizzle-rt')."
    }
    Write-Host "Run 'swizzle-rt --help' to get started."
} finally {
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
