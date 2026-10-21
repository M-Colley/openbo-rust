#requires -Version 7.0
<#
.SYNOPSIS
Build a native CLI archive and SHA256 checksum without publishing it.
.EXAMPLE
pwsh -File tools/package-release.ps1
.EXAMPLE
pwsh -File tools/package-release.ps1 -Target x86_64-unknown-linux-musl
.NOTES
Install the requested Rust target and its linker before cross-compiling.
The archive includes the executable, documentation, attribution and examples.
#>
[CmdletBinding()]
param(
    [string]$Target = '',
    [string]$OutputDirectory = '',
    [string]$Tag = '',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $metadataText = (& cargo metadata --locked --no-deps --format-version 1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed.' }
    $metadata = $metadataText | ConvertFrom-Json
    $manifestPath = [IO.Path]::GetFullPath((Join-Path $projectRoot 'Cargo.toml'))
    $packages = @($metadata.packages | Where-Object {
        [IO.Path]::GetFullPath($_.manifest_path) -eq $manifestPath
    })
    if ($packages.Count -ne 1) { throw 'Expected one package at the repository root.' }
    $package = $packages[0]
    $explicitTarget = [bool]$Target

    if ($Tag -and $Tag -cne "v$($package.version)") {
        throw "Tag '$Tag' must match package version 'v$($package.version)'."
    }
    if (-not $Target) {
        $rustcVersion = & rustc --version --verbose
        if ($LASTEXITCODE -ne 0) { throw 'rustc version lookup failed.' }
        $Target = ($rustcVersion | Where-Object { $_ -like 'host: *' }) -replace '^host: ', ''
        if (-not $Target) { throw 'Could not determine the Rust host target.' }
    }
    if ($Target -notmatch '^[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)+$') {
        throw "Invalid Rust target triple '$Target'."
    }
    if (-not $SkipBuild) {
        if ($explicitTarget) {
            & cargo build --locked --release --target $Target --bin openbo
        } else {
            & cargo build --locked --release --bin openbo
        }
        if ($LASTEXITCODE -ne 0) { throw "Release build failed for $Target." }
    }

    $binaryName = if ($Target -like '*-windows-*') { 'openbo.exe' } else { 'openbo' }
    $binaryPath = if ($explicitTarget) { "$Target/release/$binaryName" } else { "release/$binaryName" }
    $binary = Join-Path $metadata.target_directory $binaryPath
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw "Release executable not found at '$binary'."
    }
    if (-not $OutputDirectory) {
        $OutputDirectory = Join-Path $metadata.target_directory 'distribution'
    }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    # A fresh staging directory prevents old files entering a later archive.
    $stage = Join-Path $OutputDirectory ('.staging/' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Copy-Item -LiteralPath $binary -Destination (Join-Path $stage $binaryName)
    $fileNames = @($binaryName)
    foreach ($name in @('LICENSE', 'README.md', 'CITATION.cff', 'UPSTREAM_CITATION.cff', 'PORTING.md', 'VALIDATION.md', 'CHANGELOG.md')) {
        $source = Join-Path $projectRoot $name
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $stage $name)
            $fileNames += $name
        }
    }
    foreach ($name in @('examples', 'relevant_papers')) {
        $source = Join-Path $projectRoot $name
        if (Test-Path -LiteralPath $source -PathType Container) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $stage $name) -Recurse
            $fileNames += $name
        }
    }
    $baseName = "$($package.name)-$($package.version)-$Target"
    if ($Target -like '*-windows-*') {
        $archive = Join-Path $OutputDirectory "$baseName.zip"
        $files = @($fileNames | ForEach-Object { Join-Path $stage $_ })
        Compress-Archive -LiteralPath $files -DestinationPath $archive -Force
    } else {
        $archive = Join-Path $OutputDirectory "$baseName.tar.gz"
        & tar -czf $archive -C $stage -- @fileNames
        if ($LASTEXITCODE -ne 0) { throw 'Creating the tar archive failed.' }
    }
    $checksum = "$archive.sha256"
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    $utf8 = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($checksum, "$hash  $([IO.Path]::GetFileName($archive))`n", $utf8)
    if ($env:GITHUB_OUTPUT) {
        [IO.File]::AppendAllText($env:GITHUB_OUTPUT, "archive=$archive`nchecksum=$checksum`n", $utf8)
    }
    [pscustomobject]@{
        package = $package.name
        version = $package.version
        target = $Target
        archive = $archive
        checksum = $checksum
    } | ConvertTo-Json
} finally {
    Pop-Location
}
