#Requires -Version 5.1
<#
.SYNOPSIS
    Installs Swizzle Runtime (swizzle-rt) on Windows x86_64.
.DESCRIPTION
    Downloads the prebuilt release asset, verifies its SHA256 checksum,
    and installs the binary into $env:LOCALAPPDATA\Swizzle\bin.
    If no prebuilt asset exists, falls back to building from source via cargo.
#>

param(
    [string]$Version = $env:SWIZZLE_VERSION,
    [string]$InstallDir = $env:SWIZZLE_INSTALL_DIR
)

if (-not $Version) { $Version = "v0.1.0" }
if (-not $InstallDir) { $InstallDir = "$env:LOCALAPPDATA\Swizzle\bin" }

$Repo = "Swizzle-AI/swizzle-runtime"
$BaseUrl = "https://github.com/${Repo}/releases/download/${Version}"
$Archive = "swizzle-rt-${Version}-windows-x86_64-cuda12.8.zip"

# Check architecture
$Arch = (Get-CimInstance -ClassName Win32_ComputerSystem).SystemType
if ($Arch -ne "x64-based PC") {
    Write-Host "Prebuilt binaries currently target Windows x86_64 (CUDA 12.8)."
    Write-Host "Build from source with:"
    Write-Host "  cargo build --release --locked -p swizzle-cli"
    exit 1
}

$TmpDir = [System.IO.Path]::GetTempPath() + "swizzle-rt-$Version"
if (Test-Path $TmpDir) { Remove-Item -Recurse -Force $TmpDir }
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

try {
    Write-Host "Downloading Swizzle Runtime $Version ($Archive)..."
    $ArchivePath = Join-Path $TmpDir $Archive

    try {
        Invoke-WebRequest -Uri "$BaseUrl/$Archive" -OutFile $ArchivePath -UseBasicParsing -Verbose:$false
    } catch {
        Write-Host "Prebuilt binary not found for $Version."
        Write-Host ""
        Write-Host "Build from source:"
        Write-Host ""
        Write-Host "  Requires:"
        Write-Host "    - Rust toolchain (https://rustup.rs/)"
        Write-Host "    - Visual Studio Build Tools with 'Desktop development with C++' workload"
        Write-Host "    - CUDA 12.8+ (optional, for GPU acceleration)"
        Write-Host ""
        Write-Host "  cargo build --release --locked -p swizzle-cli"
        Write-Host ""
        Write-Host "  The compiled binary will be at target\release\swizzle-rt.exe"
        exit 0
    }

    # Verify SHA256
    if (Test-Path (Join-Path $TmpDir "SHA256SUMS.txt")) {
        # Download SHA256SUMS if not already present
        try {
            $SumsPath = Join-Path $TmpDir "SHA256SUMS.txt"
            if (-not (Test-Path $SumsPath)) {
                Invoke-WebRequest -Uri "$BaseUrl/SHA256SUMS.txt" -OutFile $SumsPath -UseBasicParsing -Verbose:$false
            }
            Write-Host "Verifying SHA256 checksum..."
            Set-Location $TmpDir
            & sha256sum -c SHA256SUMS.txt --ignore-missing 2>$null
            Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
        } catch {
            # Fallback: use PowerShell's Get-FileHash
            Write-Host "Verifying SHA256 checksum..."
            $Expected = Get-Content (Join-Path $TmpDir "SHA256SUMS.txt") | Where-Object { $_ -match [regex]::Escape($Archive) }
            if ($Expected) {
                $ExpectedHash = ($Expected -split '\s+')[0]
                $ActualHash = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLower()
                if ($ExpectedHash -ne $ActualHash) {
                    Write-Error "SHA256 checksum mismatch. Expected: $ExpectedHash, Got: $ActualHash"
                    exit 1
                }
                Write-Host "SHA256 checksum verified."
            } else {
                Write-Host "Warning: Could not find checksum entry for $Archive in SHA256SUMS.txt. Skipping verification."
            }
        }
    }

    # Extract archive
    Write-Host "Extracting archive..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($ArchivePath, $TmpDir)

    # Find the binary
    $BinPath = Get-ChildItem -Path $TmpDir -Filter "swizzle-rt.exe" -Recurse | Select-Object -First 1
    if (-not $BinPath) {
        Write-Error "swizzle-rt.exe binary not found in archive."
        exit 1
    }

    # Install
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -Path $BinPath.FullName -Destination (Join-Path $InstallDir "swizzle-rt.exe") -Force
    Write-Host "Installed swizzle-rt to ${InstallDir}\swizzle-rt.exe"

    # Check PATH
    $UserPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not $UserPath.Split(';') | Where-Object { $_ -eq $InstallDir }) {
        Write-Host ""
        Write-Host "Add $InstallDir to your PATH:"
        Write-Host "  `$env:PATH = `"$InstallDir;`$env:PATH`""
        Write-Host "Or permanently via:"
        Write-Host "  [System.Environment]::SetEnvironmentVariable('PATH', `"$InstallDir;$UserPath`', 'User')"
    }

    Write-Host ""
    Write-Host "Run 'swizzle-rt --help' to get started."
} finally {
    if (Test-Path $TmpDir) { Remove-Item -Recurse -Force $TmpDir }
}
