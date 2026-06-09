# nix/gpucloud-agent.nix — Overridable GPUCLOUD Agent package
#
# callPackage auto-wires nixpkgs args; flake inputs are passed explicitly.
# Users override via:
#   pkgs.gpucloud-agent.override { extraPythonPackages = [...]; }
#   pkgs.gpucloud-agent.override { extraDependencyGroups = [ "hindsight" ]; }
{
  lib,
  stdenv,
  makeWrapper,
  callPackage,
  python312,
  nodejs_22,
  electron,
  ripgrep,
  git,
  openssh,
  ffmpeg,
  tirith,

  # linux-only deps
  wl-clipboard,
  xclip,

  # Flake inputs — passed explicitly by packages.nix and overlays.nix
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  npm-lockfile-fix,
  # Locked git revision of the flake source — embedded so banner.py can
  # check for updates without needing a local .git directory. Null for
  # impure / dirty builds where flakes can't determine a rev.
  rev ? null,
  # Overridable parameters
  extraPythonPackages ? [ ],
  extraDependencyGroups ? [ ],
}:
let
  nodejs = nodejs_22;
  gpucloudVenv = callPackage ./python.nix {
    inherit uv2nix pyproject-nix pyproject-build-systems;
    dependency-groups = [ "all" ] ++ extraDependencyGroups;
  };

  gpucloudNpmLib = callPackage ./lib.nix {
    inherit npm-lockfile-fix nodejs;
  };

  gpucloudTui = callPackage ./tui.nix {
    inherit gpucloudNpmLib;
  };

  gpucloudWeb = callPackage ./web.nix {
    inherit gpucloudNpmLib;
  };

  bundledSkills = lib.cleanSourceWith {
    src = ../skills;
    filter = path: _type: !(lib.hasInfix "/index-cache/" path);
  };

  # Import bundled plugins (memory, context_engine, platforms/*).  Keeping
  # them out of the Python site-packages keeps import semantics identical
  # to a dev checkout — the loader reads them from GPUCLOUD_BUNDLED_PLUGINS.
  bundledPlugins = lib.cleanSourceWith {
    src = ../plugins;
    filter = path: _type: !(lib.hasInfix "/__pycache__/" path);
  };

  # i18n locale catalogs (locales/*.yaml). Shipped into the store and pointed
  # at by GPUCLOUD_BUNDLED_LOCALES so the wrapped binary always resolves human
  # strings instead of raw i18n keys (#23943 / #27632 / #35374).
  #
  # Defense-in-depth, not load-bearing: the wheel already declares locales/ as
  # setuptools data-files, so uv2nix materializes them into the venv's data
  # scheme and agent/i18n.py resolves them with no env var. The wrapper override
  # pins the store path so a future uv2nix change that drops data-files can't
  # silently ship raw keys via `nix build` (checks don't run on a plain build).
  # The bundled-locales flake check verifies BOTH paths independently.
  #
  # Plain cleanSource (no __pycache__ filter): locales/ is bare *.yaml, never
  # compiled, so it never carries a __pycache__ dir to exclude.
  bundledLocales = lib.cleanSource ../locales;

  runtimeDeps = [
    nodejs
    ripgrep
    git
    openssh
    ffmpeg
    tirith
  ]
  ++ lib.optionals stdenv.isLinux [
    wl-clipboard
    xclip
  ];

  runtimePath = lib.makeBinPath runtimeDeps;

  sitePackagesPath = python312.sitePackages;

  # Walk propagatedBuildInputs to include transitive Python deps in PYTHONPATH.
  # Without this, a plugin listing e.g. requests as a dep would fail at runtime
  # if requests isn't already in the sealed uv2nix venv.
  allExtraPythonPackages = python312.pkgs.requiredPythonModules extraPythonPackages;

  pythonPath = lib.makeSearchPath sitePackagesPath allExtraPythonPackages;

  pyprojectHash = builtins.hashString "sha256" (builtins.readFile ../pyproject.toml);
  uvLockHash =
    if builtins.pathExists ../uv.lock then
      builtins.hashString "sha256" (builtins.readFile ../uv.lock)
    else
      "none";
  checkPackageCollisions = ''
    import pathlib, sys, re

    def canonical(name):
        return re.sub(r'[-_.]+', '-', name).lower()

    # Collect core venv package names
    core = set()
    venv_sp = pathlib.Path('${gpucloudVenv}/${sitePackagesPath}')
    for di in venv_sp.glob('*.dist-info'):
        meta = di / 'METADATA'
        if meta.exists():
            for line in meta.read_text().splitlines():
                if line.startswith('Name:'):
                    core.add(canonical(line.split(':', 1)[1].strip()))
                    break

    # Check each extra package for collisions
    extras_dirs = [${lib.concatMapStringsSep ", " (p: "'${toString p}'") allExtraPythonPackages}]
    for edir in extras_dirs:
        sp = pathlib.Path(edir) / '${sitePackagesPath}'
        if not sp.exists():
            continue
        for di in sp.glob('*.dist-info'):
            meta = di / 'METADATA'
            if not meta.exists():
                continue
            for line in meta.read_text().splitlines():
                if line.startswith('Name:'):
                    pkg = canonical(line.split(':', 1)[1].strip())
                    if pkg in core:
                        print(f'ERROR: plugin package \"{pkg}\" collides with a package in gpucloud sealed venv', file=sys.stderr)
                        print(f'  from: {di}', file=sys.stderr)
                        print(f'  Remove this dependency from extraPythonPackages.', file=sys.stderr)
                        sys.exit(1)
                    break

    print('No collisions found.')
  '';
in
stdenv.mkDerivation (finalAttrs: {
  pname = "gpucloud-agent";
  version = (fromTOML (builtins.readFile ../pyproject.toml)).project.version;

  dontUnpack = true;
  dontBuild = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/gpucloud-agent $out/bin
    cp -r ${bundledSkills} $out/share/gpucloud-agent/skills
    cp -r ${bundledPlugins} $out/share/gpucloud-agent/plugins
    cp -r ${bundledLocales} $out/share/gpucloud-agent/locales
    cp -r ${gpucloudWeb} $out/share/gpucloud-agent/web_dist

    mkdir -p $out/ui-tui
    cp -r ${gpucloudTui}/lib/gpucloud-tui/* $out/ui-tui/

    ${lib.concatMapStringsSep "\n"
      (name: ''
        makeWrapper ${gpucloudVenv}/bin/${name} $out/bin/${name} \
          --suffix PATH : "${runtimePath}" \
          --set GPUCLOUD_BUNDLED_SKILLS $out/share/gpucloud-agent/skills \
          --set GPUCLOUD_BUNDLED_PLUGINS $out/share/gpucloud-agent/plugins \
          --set GPUCLOUD_BUNDLED_LOCALES $out/share/gpucloud-agent/locales \
          --set GPUCLOUD_WEB_DIST $out/share/gpucloud-agent/web_dist \
          --set GPUCLOUD_TUI_DIR $out/ui-tui \
          --set GPUCLOUD_PYTHON ${gpucloudVenv}/bin/python3 \
          --set GPUCLOUD_NODE ${lib.getExe nodejs} \
          ${lib.optionalString (rev != null) ''--set GPUCLOUD_REVISION ${rev} \''}
          ${lib.optionalString (extraPythonPackages != [ ]) ''--suffix PYTHONPATH : "${pythonPath}"''}
      '')
      [
        "gpucloud"
        "gpucloud-agent"
        "gpucloud-acp"
      ]
    }

    ${lib.optionalString (extraPythonPackages != [ ]) ''
      echo "=== Checking for plugin/core package collisions ==="
      ${gpucloudVenv}/bin/python3 -c "${checkPackageCollisions}"
      echo "=== No collisions ==="
    ''}

    runHook postInstall
  '';

  passthru = {
    inherit
      gpucloudTui
      gpucloudWeb
      gpucloudNpmLib
      gpucloudVenv
      ;

    # `gpucloudDesktop` references `finalAttrs.finalPackage` (this whole
    # derivation, after all overrides are applied) so the desktop wrapper
    # can prepend its `/bin` to PATH.  The desktop's resolver step 4
    # ("existing gpucloud on PATH") then picks up the fully wrapped
    # `gpucloud` binary — venv with all deps, bundled skills/plugins,
    # runtime PATH (ripgrep/git/ffmpeg/etc).  No re-implementation
    # of the agent resolution in the desktop wrapper.
    gpucloudDesktop = callPackage ./desktop.nix {
      inherit gpucloudNpmLib electron;
      gpucloudAgent = finalAttrs.finalPackage;
    };

    devShellHook = ''
      STAMP=".nix-stamps/gpucloud-agent"
      STAMP_VALUE="${pyprojectHash}:${uvLockHash}"
      if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
        echo "gpucloud-agent: installing Python dependencies..."
        uv venv .venv --python ${python312}/bin/python3 2>/dev/null || true
        source .venv/bin/activate
        uv pip install -e ".[all]"
        [ -d mini-swe-agent ] && uv pip install -e ./mini-swe-agent 2>/dev/null || true
        mkdir -p .nix-stamps
        echo "$STAMP_VALUE" > "$STAMP"
      else
        source .venv/bin/activate
        export GPUCLOUD_PYTHON=${gpucloudVenv}/bin/python3
      fi
    '';
  };

  meta = with lib; {
    description = "AI agent with advanced tool-calling capabilities";
    homepage = "https://github.com/NousResearch/gpucloud-agent";
    mainProgram = "gpucloud";
    license = licenses.mit;
    platforms = platforms.unix;
  };
})
