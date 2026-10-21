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

function Update-StagedStudentGuide {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$PackageVersion,
        [Parameter(Mandatory)][string]$BuildRevision
    )
    $guidePath = Join-Path $Directory 'docs/index.html'
    if (-not (Test-Path -LiteralPath $guidePath -PathType Leaf)) { return }
    $guide = [IO.File]::ReadAllText($guidePath)
    $guide = $guide.Replace('@PACKAGE_VERSION@', $PackageVersion).Replace('@BUILD_REVISION@', $BuildRevision)
    # Native archives do not contain rustdoc's openbo/ directory.
    $guide = [regex]::Replace($guide, '(?i)(\bhref\s*=\s*["''])openbo/', '${1}https://m-colley.github.io/openbo-rust/openbo/')
    [IO.File]::WriteAllText($guidePath, $guide, [Text.UTF8Encoding]::new($false))
}

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
    $rustcVersion = & rustc --version --verbose
    if ($LASTEXITCODE -ne 0) { throw 'rustc version lookup failed.' }
    $hostTarget = ($rustcVersion | Where-Object { $_ -like 'host: *' }) -replace '^host: ', ''
    if (-not $hostTarget) { throw 'Could not determine the Rust host target.' }
    if (-not $Target) { $Target = $hostTarget }
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
    if ($Target -ceq $hostTarget) {
        $binaryVersion = (& $binary --version) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw 'Release executable version check failed.' }
        if ($binaryVersion.Trim() -cne "openbo $($package.version)") {
            throw "Release executable reports '$($binaryVersion.Trim())'; expected 'openbo $($package.version)'. Rebuild before packaging."
        }
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
    foreach ($name in @('examples', 'relevant_papers', 'docs')) {
        $source = Join-Path $projectRoot $name
        if (Test-Path -LiteralPath $source -PathType Container) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $stage $name) -Recurse
            $fileNames += $name
        }
    }
    $buildRevision = "v$($package.version)"
    # Do not inherit a parent checkout's revision when packaging a source archive.
    if ((Test-Path -LiteralPath (Join-Path $projectRoot '.git')) -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $revision = (& git -C $projectRoot rev-parse --verify HEAD 2>$null) -join ''
        if ($LASTEXITCODE -eq 0 -and $revision -match '^[a-fA-F0-9]{40,64}$') {
            $buildRevision = $revision
        }
    }
    Update-StagedStudentGuide -Directory $stage -PackageVersion $package.version -BuildRevision $buildRevision
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
